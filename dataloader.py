
from dataclasses import dataclass
import os
from typing import Iterator
import nibabel as nib
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from functools import lru_cache

BRAIN_FRAMES = 155
IMAGE_SIZE = 128
CHANNELS = 4
X_DTYPE = np.float32
Y_DTYPE = np.uint8
LABEL_MAPPING_PATTERN = {0: 0, 2: 1, 4: 2, 1: 3}

@dataclass
class Brain:
    t1: nib.Nifti1Image
    t1ce: nib.Nifti1Image
    t2: nib.Nifti1Image
    flair: nib.Nifti1Image
    seg: nib.Nifti1Image

def get_directory_paths(directory_path: str):
    return [f.path for f in os.scandir(directory_path) if f.is_dir()]

def load_dataset_paths(labeled_path: str, unlabeled_path: str, validation_size_from_train: float = 0.2, test_size: float = 0.15, random_state = 42):
    hgg_path = os.path.join(labeled_path, 'HGG')
    lgg_path = os.path.join(labeled_path, 'LGG')

    hgg_brains = get_directory_paths(hgg_path)
    hgg_train_val_brains, hgg_test_brains = train_test_split(hgg_brains, test_size=test_size, random_state=random_state)
    lgg_brains = get_directory_paths(lgg_path)
    lgg_train_val_brains, lgg_test_brains = train_test_split(lgg_brains, test_size=test_size, random_state=random_state)

    test_brains = hgg_test_brains + lgg_test_brains

    hgg_train_brains, hgg_val_brains = train_test_split(hgg_train_val_brains, test_size=validation_size_from_train, random_state=random_state)
    lgg_train_brains, lgg_val_brains = train_test_split(lgg_train_val_brains, test_size=validation_size_from_train, random_state=random_state)

    train_brains = hgg_train_brains + lgg_train_brains
    val_brains = hgg_val_brains + lgg_val_brains
    
    unlabeled_brains = get_directory_paths(unlabeled_path)

    return (train_brains, val_brains, test_brains, unlabeled_brains), (len(hgg_train_brains), len(lgg_train_brains), len(hgg_val_brains), len(lgg_val_brains))

def extend_path_from_last_part(path: str) -> str:
    return os.path.join(path, os.path.basename(path))


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_paths: list[str], max_value: int, batch_size: int = 32, brain_slices: int = 8, X_dtype = X_DTYPE, Y_dtype = Y_DTYPE, bootstrap: bool = True, hgg_size: int = None, lgg_size: int = None):
        self.dataset_paths = [extend_path_from_last_part(path).decode('ASCII') for path in dataset_paths]
        self.brain_slices = brain_slices
        self.batch_size = batch_size
        self.sample_size = self.batch_size//self.brain_slices
        self.batches_per_brain = BRAIN_FRAMES//self.batch_size
        self.cutted_frames = self.batches_per_brain * self.batch_size
        rest = BRAIN_FRAMES - self.cutted_frames
        self.offset = rest//2 + 1
        self.slice_size = self.cutted_frames//brain_slices
        self.cur_brain: Brain = None
        self.X_dtype = X_dtype
        self.Y_dtype = Y_dtype
        self.max_value = max_value
        self.len = self.__len__()
        self.bootstrap = bootstrap
        if bootstrap:
            self.hgg_size = hgg_size
            self.lgg_size = lgg_size

    @lru_cache(maxsize=32)
    def _load_brain(self, idx) -> Brain:
        path = self.dataset_paths[idx]
        t1 = nib.load(f'{path}_t1.nii')
        t1ce = nib.load(f'{path}_t1ce.nii')
        t2 = nib.load(f'{path}_t2.nii')
        flair = nib.load(f'{path}_flair.nii')
        seg = nib.load(f'{path}_seg.nii')

        return Brain(t1, t1ce, t2, flair, seg)
    
    def _map_labels(self, arr):
        u, inv = np.unique(arr, return_inverse=True)
        return np.array([LABEL_MAPPING_PATTERN[x] for x in u], dtype=self.Y_dtype)[inv].reshape(arr.shape)

    def __len__(self):
        return len(self.dataset_paths) * self.batches_per_brain

    def __getitem__(self, idx):
        if self.bootstrap:

            batch_X = np.zeros((self.batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=self.X_dtype)
            batch_Y = np.zeros((self.batch_size, IMAGE_SIZE, IMAGE_SIZE), dtype=self.Y_dtype)

            ceil = BRAIN_FRAMES - self.offset
            batch_index = 0
            for slice in range(self.brain_slices):
                low = max(slice * self.slice_size + self.offset - 1, 0)
                high = min(low + self.slice_size + 1, ceil)

                for sample in range(self.sample_size):
                    brain_scan_index = tf.random.uniform(shape=(), maxval=self.hgg_size, dtype=tf.int32).numpy() if sample < self.sample_size//2 else tf.random.uniform(shape=(), minval=self.hgg_size, maxval=(self.hgg_size + self.lgg_size), dtype=tf.int32).numpy()
                    brain_scan = self._load_brain(brain_scan_index)
                    sample_index = tf.random.uniform(shape=(), minval=low, maxval=high, dtype=tf.int32).numpy()

                    batch_X[batch_index, :, :, 0] = cv2.resize(brain_scan.t1.get_fdata(dtype=self.X_dtype)[:, :, sample_index], (IMAGE_SIZE, IMAGE_SIZE))
                    batch_X[batch_index, :, :, 1] = cv2.resize(brain_scan.t1ce.get_fdata(dtype=self.X_dtype)[:, :, sample_index], (IMAGE_SIZE, IMAGE_SIZE))
                    batch_X[batch_index, :, :, 2] = cv2.resize(brain_scan.t2.get_fdata(dtype=self.X_dtype)[:, :, sample_index], (IMAGE_SIZE, IMAGE_SIZE))
                    batch_X[batch_index, :, :, 3] = cv2.resize(brain_scan.flair.get_fdata(dtype=self.X_dtype)[:, :, sample_index], (IMAGE_SIZE, IMAGE_SIZE))
                    y_map = self._map_labels(brain_scan.seg.get_fdata(dtype=self.X_dtype)[:, :, sample_index])
                    
                    batch_Y[batch_index, :, :] = cv2.resize(y_map, (IMAGE_SIZE, IMAGE_SIZE))

                    batch_index += 1

            return batch_X/self.max_value, tf.one_hot(batch_Y, 4, dtype=self.Y_dtype)

        step = idx % self.batches_per_brain

        if step == 0:

            self.cur_brain = self._load_brain(idx // self.batches_per_brain)

        batch_X = np.zeros((self.batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=self.X_dtype)
        batch_Y = np.zeros((self.batch_size, IMAGE_SIZE, IMAGE_SIZE), dtype=self.Y_dtype)

        batch_index = 0
        ceil = BRAIN_FRAMES - self.offset
        for slice in range(self.brain_slices):
            low = max(slice * self.slice_size + step * self.sample_size + self.offset - 1, 0)
            high = min(low + self.sample_size, ceil)
            t1_cut = self.cur_brain.t1.get_fdata(dtype=self.X_dtype)[:, :, low : high]
            t1ce_cut = self.cur_brain.t1ce.get_fdata(dtype=self.X_dtype)[:, :, low : high]
            t2_cut = self.cur_brain.t2.get_fdata(dtype=self.X_dtype)[:, :, low : high]
            flair_cut = self.cur_brain.flair.get_fdata(dtype=self.X_dtype)[:, :, low : high]
            seg_cut = self.cur_brain.seg.get_fdata(dtype=self.X_dtype)[:, :, low : high]
            X_cuts = (t1_cut, t1ce_cut, t2_cut, flair_cut)

            for i in range(self.sample_size):

                for channel in range(CHANNELS):
                    batch_X[batch_index, :, :, channel] = cv2.resize(X_cuts[channel][:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                
                batch_Y[batch_index, :, :] = cv2.resize(seg_cut[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                batch_index += 1

        return batch_X/self.max_value, tf.one_hot(batch_Y, 4, dtype=self.Y_dtype)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._load_brain.cache_clear()

def build_data_generator(dataset_paths: list[str], max_value: int, batch_size: int = 32, brain_slices: int = 8, X_dtype = X_DTYPE, Y_dtype = Y_DTYPE, bootstrap: bool = True, hgg_size: int = None, lgg_size: int = None) -> Iterator[tuple[np.ndarray, tf.Tensor]]:
    with DataGenerator(dataset_paths, max_value, batch_size, brain_slices, X_dtype, Y_dtype, bootstrap, hgg_size, lgg_size) as generator:

        for i in range(generator.len):
            yield generator.__getitem__(i)

def find_max_per_channel(paths: list[str]) -> dict[str, float]:

    t1_max, t1ce_max, t2_max, flair_max = 0, 0, 0, 0

    for path in paths:
        extended = extend_path_from_last_part(path)
        t1_max = max(t1_max, np.max(nib.load(f'{extended}_t1.nii').get_fdata()))
        t1ce_max = max(t1ce_max, np.max(nib.load(f'{extended}_t1ce.nii').get_fdata()))
        t2_max = max(t2_max, np.max(nib.load(f'{extended}_t2.nii').get_fdata()))
        flair_max = max(flair_max, np.max(nib.load(f'{extended}_flair.nii').get_fdata()))

    return {'t1': t1_max, 't1ce': t1ce_max, 't2': t2_max, 'flair': flair_max}


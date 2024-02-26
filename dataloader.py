
from dataclasses import dataclass
import os
from typing import Iterator
import nibabel as nib
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from functools import lru_cache

# folder treningowy zawiera HGG i LGG. Należy stworzyć generator łączący oba zbiory w jeden potasowany w losowej kolejności. Każdy element to oddzielny mózg. Każdy mózg to (240, 240, 155)
# jedna próbka będzie jednym przekrojem w kilku kanałach, a więc będzie tensorem o wymiarach (240, 240, channels). Takich próbek będzie tyle ile we wsadzie
# należy opracować metodę, aby pojedynczy wsad był jak najbardziej reprezentatywny dla procesu optymalizacji. Nie może być tak, że cały wsad będzie składał się z początkowych slice'ów pojedynczego mózgu
# batch_size/brain_slices to ile klatek pobieramy z każdego slicea, 155/batch_size = wynik to ile wsadów na jeden mózg, 155 - wynik * batch_size = reszta,
# reszta/2 = offset, [offset-1:155-offset]
# przykład: batch_size = 32, wynik = 155/32 = 4 = steps, 4*32 = 128, 155-128 = 27, 27/2=13, [12:142], brain_slices = 8, 32/8=4 = sample_size, 155/8 = 19 = slice_size
# for step in range(steps): for slice in range(brain_slices): low = slice*slice_size (if slice != 0: -1) + step*sample_size 
# [low + offset : low + sample_size - offset]

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

class FrameCutter:

    def __init__(self, frame_radius: int = 5):
        middle = BRAIN_FRAMES/2
        offset = frame_radius/2
        self.low =  int(middle - offset)
        self.high = int(middle + offset)
        self.w_low_index, self.w_high_index, self.h_low_index, self.h_high_index = IMAGE_SIZE - 1, 0, IMAGE_SIZE - 1, 0

    def fit(self, paths: list[str]):

        for path in paths:
            extended = extend_path_from_last_part(path)
            clip = nib.load(f'{extended}_t1.nii').get_fdata(dtype=X_DTYPE)[:, :, self.low:self.high]
            for i in range(clip.shape[-1]):
                resized = cv2.resize(clip[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                h_low_index, h_high_index, w_low_index, w_high_index = self._find_limit_values(resized)

                self.h_low_index = min(self.h_low_index, h_low_index)
                self.h_high_index = max(self.h_high_index, h_high_index)

                self.w_low_index = min(self.w_low_index, w_low_index)
                self.w_high_index = max(self.w_high_index, w_high_index)


    def transform(self, array: np.ndarray) -> np.ndarray:
        if self.w_low_index == IMAGE_SIZE - 1 and self.w_high_index == 0 and self.h_low_index == IMAGE_SIZE - 1 and self.h_high_index == 0:
            raise FrameCutter.NotFittedError()
        
        return array[self.h_low_index:self.h_high_index, self.w_low_index:self.w_high_index]

    def _find_limit_values(self, array: np.ndarray) -> tuple[int, int, int, int]:
        
        def get_through_axis(array, mask, axis):
            low_indexes = mask.argmax(axis=axis)
            low_index = low_indexes[low_indexes.nonzero()].min()
            high_indexes = np.flip(mask, axis=axis).argmax(axis=axis)
            high_index = (array.shape[0] - high_indexes[high_indexes.nonzero()] - 1).max()

            return low_index, high_index

        mask = array != 0
        h_low_index, h_high_index = get_through_axis(array, mask, axis=0)
        w_low_index, w_high_index = get_through_axis(array, mask, axis=1)

        return h_low_index, h_high_index, w_low_index, w_high_index
        
    
    class NotFittedError(Exception):

        def __init__(self):
            super('Frame cutter was not fitted! Use .fit method or fill index fields manually')

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
        print(idx)
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
            extended = extend_path_from_last_part(self.dataset_paths[idx // self.batches_per_brain])
            t1 = nib.load(f'{extended}_t1.nii').get_fdata(dtype=self.X_dtype)
            t1ce = nib.load(f'{extended}_t1ce.nii').get_fdata(dtype=self.X_dtype)
            t2 = nib.load(f'{extended}_t2.nii').get_fdata(dtype=self.X_dtype)
            flair = nib.load(f'{extended}_flair.nii').get_fdata(dtype=self.X_dtype)
            seg = nib.load(f'{extended}_seg.nii').get_fdata(dtype=self.X_dtype)


            self.cur_brain = Brain(t1, t1ce, t2, flair, seg)

        batch_X = np.zeros((self.batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=self.X_dtype)
        batch_Y = np.zeros((self.batch_size, IMAGE_SIZE, IMAGE_SIZE), dtype=self.Y_dtype)

        batch_index = 0
        ceil = BRAIN_FRAMES - self.offset
        for slice in range(self.brain_slices):
            low = max(slice * self.slice_size + step * self.sample_size + self.offset - 1, 0)
            high = min(low + self.sample_size, ceil)
            t1_cut = self.cur_brain.t1[:, :, low : high]
            t1ce_cut = self.cur_brain.t1ce[:, :, low : high]
            t2_cut = self.cur_brain.t2[:, :, low : high]
            flair_cut = self.cur_brain.flair[:, :, low : high]
            seg_cut = self.cur_brain.seg[:, :, low : high]
            X_cuts = (t1_cut, t1ce_cut, t2_cut, flair_cut)

            for i in range(self.sample_size):

                for channel in range(CHANNELS):
                    batch_X[batch_index, :, :, channel] = cv2.resize(X_cuts[channel][:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                
                batch_Y[batch_index, :, :] = cv2.resize(seg_cut[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                batch_index += 1

        return batch_X/self.max_value, tf.one_hot(batch_Y)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self._load_brain.cache_info())
        self._load_brain.cache_clear()

def build_data_generator(dataset_paths: list[str], max_value: int, hgg_size: int, lgg_size: int, batch_size: int = 32, brain_slices: int = 8, X_dtype = X_DTYPE, Y_dtype = Y_DTYPE, bootstrap: bool = True) -> Iterator[tuple[np.ndarray, tf.Tensor]]:
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


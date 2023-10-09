
from dataclasses import dataclass
import os
import nibabel as nib
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split

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

@dataclass
class Brain:
    t1: np.memmap
    t1ce: np.memmap
    t2: np.memmap
    flair: np.memmap
    seg: np.memmap

def get_directory_paths(directory_path: str):
    return [f.path for f in os.scandir(directory_path) if f.is_dir()]

def load_dataset_paths(train_path: str, test_path: str, validation_size: float = 0.2) -> tuple[list[str], list[str], list[str]]:
    hgg_path = os.path.join(train_path, 'HGG')
    lgg_path = os.path.join(train_path, 'LGG')

    brains = get_directory_paths(hgg_path)
    lgg_brains = get_directory_paths(lgg_path)

    brains.extend(lgg_brains)

    train_brains, val_brains = train_test_split(brains, test_size=validation_size, random_state=42, shuffle=True)
    
    test_brains = get_directory_paths(test_path)

    return train_brains, val_brains, test_brains

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_paths: list[str], batch_size: int = 32, brain_slices: int = 8, X_dtype = np.float32, Y_dtype = np.int8):
        self.dataset_paths = dataset_paths
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

    def __len__(self):
        return self.dataset_paths * self.batches_per_brain

    def __getitem__(self, idx):

        step = idx % self.batches_per_brain

        if step == 0:
            extended = self._extend_path_from_last_part(self.dataset_paths[idx])
            t1 = nib.load(f'{extended}_t1.nii').get_fdata(dtype=self.X_dtype)
            t1ce = nib.load(f'{extended}_t1ce.nii').get_fdata(dtype=self.X_dtype)
            t2 = nib.load(f'{extended}_t2.nii').get_fdata(dtype=self.X_dtype)
            flair = nib.load(f'{extended}_flair.nii').get_fdata(dtype=self.X_dtype)
            seg = nib.load(f'{extended}_seg.nii').get_fdata(dtype=self.X_dtype)

            self.cur_brain = Brain(t1, t1ce, t2, flair, seg)

        batch_X = np.zeros((self.batch_size, IMAGE_SIZE, IMAGE_SIZE, 4), dtype=self.X_dtype)
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

            for i in range(self.sample_size):
                batch_X[batch_index, :, :, 0] = cv2.resize(t1_cut[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                batch_X[batch_index, :, :, 1] = cv2.resize(t1ce_cut[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                batch_X[batch_index, :, :, 2] = cv2.resize(t2_cut[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                batch_X[batch_index, :, :, 3] = cv2.resize(flair_cut[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                batch_Y[batch_index, :, :] = cv2.resize(seg_cut[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                batch_index += 1

        return batch_X/np.max(batch_X), batch_Y
    
    def on_epoch_end(self):
        self.dataset_paths = tf.random.shuffle(self.dataset_paths)

    def _extend_path_from_last_part(self, path):
        return os.path.join(path, os.path.basename(path))

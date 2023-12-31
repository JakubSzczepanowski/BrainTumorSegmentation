
from dataclasses import dataclass
import os
from typing import Iterator
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
CHANNELS = 4
X_DTYPE = np.float32
Y_DTYPE = np.int8

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
            start = IMAGE_SIZE//2
            for i in range(clip.shape[-1]):
                resized = cv2.resize(clip[:, :, i], (IMAGE_SIZE, IMAGE_SIZE))
                horizontal_strip = resized[start, :]
                low_index, high_index = self._find_limit_values(horizontal_strip)
                self.w_low_index = min(self.w_low_index, low_index)
                self.w_high_index = max(self.w_high_index, high_index)

                vertical_strip = resized[:, start]
                low_index, high_index = self._find_limit_values(vertical_strip)
                self.h_low_index = min(self.h_low_index, low_index)
                self.h_high_index = max(self.h_high_index, high_index)

    def transform(self, array: np.ndarray) -> np.ndarray:
        if self.w_low_index == IMAGE_SIZE - 1 and self.w_high_index == 0 and self.h_low_index == IMAGE_SIZE - 1 and self.h_high_index == 0:
            raise FrameCutter.NotFittedError()
        
        return array[self.h_low_index:self.h_high_index, self.w_low_index:self.w_high_index]
    
    def get_indexes(self) -> dict[str, int]:
        return {'h_low_index': self.h_low_index, 'h_high_index': self.h_high_index, 'w_low_index': self.w_low_index, 'w_high_index': self.w_high_index}

    def _find_limit_values(self, array: np.ndarray) -> tuple[int, int]:
        low_index, high_index = 0, 0
        is_brain = False
        for index, elem in enumerate(array):
            if elem and not is_brain:
                low_index = index
                is_brain = True
            elif not elem and is_brain:
                high_index = index
                break

        return low_index, high_index
    
    class NotFittedError(Exception):

        def __init__(self):
            super('Frame cutter was not fitted! Use .fit method or fill index fields manually')

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_paths: list[str], h_low_index: int, h_high_index: int, w_low_index: int, w_high_index: int, batch_size: int = 32, brain_slices: int = 8, X_dtype = X_DTYPE, Y_dtype = Y_DTYPE):
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
        self.h_low_index = h_low_index
        self.h_high_index = h_high_index
        self.w_low_index = w_low_index
        self.w_high_index = w_high_index
        self.len = self.__len__()

    def __len__(self):
        return len(self.dataset_paths) * self.batches_per_brain

    def __getitem__(self, idx):

        step = idx % self.batches_per_brain

        if step == 0:
            path = self.dataset_paths[idx]
            extended = extend_path_from_last_part(path if type(path) == str else path.decode('ASCII'))
            t1 = nib.load(f'{extended}_t1.nii').get_fdata(dtype=self.X_dtype)
            t1ce = nib.load(f'{extended}_t1ce.nii').get_fdata(dtype=self.X_dtype)
            t2 = nib.load(f'{extended}_t2.nii').get_fdata(dtype=self.X_dtype)
            flair = nib.load(f'{extended}_flair.nii').get_fdata(dtype=self.X_dtype)
            seg = nib.load(f'{extended}_seg.nii').get_fdata(dtype=self.X_dtype)

            self.cur_brain = Brain(t1, t1ce, t2, flair, seg)

        new_height = self.h_high_index - self.h_low_index
        new_width = self.w_high_index - self.w_low_index
        batch_X = np.zeros((self.batch_size, new_height, new_width, CHANNELS), dtype=self.X_dtype)
        batch_Y = np.zeros((self.batch_size, new_height, new_width), dtype=self.Y_dtype)

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
                    batch_X[batch_index, :, :, channel] = self._cut_frame(cv2.resize(X_cuts[channel][:, :, i], (IMAGE_SIZE, IMAGE_SIZE)))
                
                batch_Y[batch_index, :, :] = self._cut_frame(cv2.resize(seg_cut[:, :, i], (IMAGE_SIZE, IMAGE_SIZE)))
                batch_index += 1

        return batch_X/np.max(batch_X), batch_Y
    
    def on_epoch_end(self):
        self.dataset_paths = tf.random.shuffle(self.dataset_paths)

    def _cut_frame(self, array: np.ndarray) -> np.ndarray:
        return array[self.h_low_index:self.h_high_index, self.w_low_index:self.w_high_index]

def build_data_generator(dataset_paths: list[str], h_low_index: int, h_high_index: int, w_low_index: int, w_high_index: int, batch_size: int = 32, brain_slices: int = 8, X_dtype = X_DTYPE, Y_dtype = Y_DTYPE) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    generator = DataGenerator(dataset_paths, h_low_index, h_high_index, w_low_index, w_high_index, batch_size, brain_slices, X_dtype, Y_dtype)

    for i in range(generator.len):
        yield generator.__getitem__(i)

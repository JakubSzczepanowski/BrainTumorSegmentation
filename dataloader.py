import os
import nibabel as nb


def load_images(path: str):
    hgg_path = os.path.join(path, 'hgg')
    lgg_path = os.path.join(path, 'lgg')

    hgg_brains = os.listdir(hgg_path)
    lgg_brains = os.listdir(lgg_path)

    print(hgg_brains)
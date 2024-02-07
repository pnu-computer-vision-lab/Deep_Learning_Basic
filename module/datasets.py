from typing import Union, List

import numpy as np
import cv2
import pickle
from PIL import Image

from torch.utils.data import Dataset
from data.cifar10.cifar10 import load_train_data, load_test_data

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

PICKLED_DATA_DIR = r"../data/cifar10/"

def get_dataset(name: str, **kwargs):
    if name == "custom":
        return CustomDataset(**kwargs)
    else:
        raise ValueError("Incorrect Name")

class CustomDataset(Dataset):
    """
    CustomDataset for Image Data
    """

    def __init__(
            self,
            imgs: np.ndarray,
            labels: np.ndarray,
            transforms=None
    ):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        # Hint :
        # get image by using opencv-python or pillow library
        # return image and label(you can return as tuple or dictionary type)

        ## TODO ##
        # ----- Modify Example Code -----
        # img_array = self.imgs[index]
        img = self.imgs[index]

        # todo. Solve the error below
        # if self.transforms is not None:
        #     # img = self.transforms(image=img)["image"]

        return {
            "img": img,
            "label": self.labels[index]
        }
        # -------------------------------

    def __len__(self):
        # Hint : return labels or img_paths length
        return len(self.imgs)


if __name__ == "__main__":
    # Check if CustomDataset is working
    pass

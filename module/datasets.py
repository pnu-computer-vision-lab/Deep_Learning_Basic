"""
Defines a 'CustomDataset' class, which inherits from PyTorch's 'Dataset' class, intended for image data

To support image data loading and preprocessing
"""

from typing import Union, List

import numpy as np
import cv2
# cv2; powerful library for working with images in python

from torch.utils.data import Dataset

import os
from PIL import Image

def get_dataset(name: str, **kwargs):
    if name == "custom":
        return CustomDataset(**kwargs)
    else:
        raise ValueError("Incorrect Name")
    

class CustomDataset(Dataset):
    """
    CustomDataset for Image Data
    """

    # Takes paths to images corresponding labels and optional transformations
    # It stores these information as instance variables
    def __init__(
            self,
            images: np.ndarray,                 # Union type; Union[X,Y] means either X or Y
            labels: Union[List, np.ndarray],    # List; type hinting to indicate the types of variables
            transforms = None                   #       ,function parameters and return values
        ):

        self.images = images
        self.labels = labels
        self.transforms = transforms
    
    #
    def __getitem__(self, index):
        """Retrieves an [images-label] pair from the dataset
           Ensuring the image is in the correct format (convert into PIL image)

        1) Convert to PIL image if necessary
        2) Apply any specified transformations
        3) Return the processed image along with its corresponding label

        cf) Original purpose:
        It reads the image from the file system using OpenCV,
        it converts into RGB and applies some optional transformations

        Return:
            a dictionary with the image and label
        """

        # Fetching the image
        # self.images ~ a collection (ie. NumPy array) where each element is an image 
        img = self.images[index]

        # Check if img is a PIL Image, if not convert it
        # If not, assumes 'img' is a NumPy array representing an image
        # => Many transformation libraries (ie. albumentations) can work directly with PIL images
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img.astype('uint8'), 'RGB')

        # 'albumentations' works iwth the numpy arrays
        # ensure that the image is converted to a nympy array before applying the transformations
        if self.transforms is not None:
            img = self.transforms(image=np.array(img))["image"]
        
        # Fetching the label
        # self.labels ~ a collection where each element is the label corresponding to each image in 'self.images'
        label = self.labels[index]

        return {"img": img,
                "label": label}
        # # Hint :
        # # get image by using opencv-python or pillow library
        # # return image and label(you can return as tuple or dictionary type)

        # ## TODO ##
        # # ----- Modify Example Code -----

        # # Load the image
        # img_path = self.img_paths[index]

        # if not isinstance(img_path, str) or not os.path.isfile(img_path):
        #     raise ValueError(f"Invalid image path: {img_path}")

        # # load the image file from the given path
        # img = cv2.imread(img_path)
        # if img is None:
        #     raise ValueError(f"Unable to read image at path: {img_path}")
        # # OpenCV, the color image is saved as 'BGR' order
        # # convert the order in 'RGB' so as to display well with matplotlib
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if self.transforms is not None:
        #     img = self.transforms(image=img)["image"]

        # # apply the transformations
        # if self.transforms is not None:
        #     img = self.transforms(image=img)["image"]
        
        # return {
        #     "img": img,
        #     "label": self.labels[index]
        # }
        # # -------------------------------

    def __len__(self):
        """Returns the length of the dataset

        the length of the dataset == the number of image paths provided
        """
        # Hint : return labels or img_paths length
        return len(self.images)


if __name__ == "__main__":
    # Check if CustomDataset is working
    pass

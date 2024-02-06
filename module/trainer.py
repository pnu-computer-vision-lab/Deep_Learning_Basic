"""Designed to handle the training process for a neural network using PyTorch
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tarfile
import pickle
import numpy as np

# [glob];
#   - Finds all the path_names matching a specified pattern according to the rules used by the Unix shell

# [glob.glob(pathname, *, root_dir=None, dir_fd=None, recursive=False, include_hidden=False)];
#   - Return a list of path names that match pathname
from glob import glob

# [tqdm];
#   - Provides a simple and convenient way to add progress bars to loops and iterable objects

# [tqdm.tqdm];
#   - Decorate an iterable object
#   - Return an iterator which acts exactly like the original iterable
from tqdm import tqdm

# [sklearn]; Package that offers several ML, DL and data analysis tools
# [sklearn.model_selection.train_test_split]; 
#   - Split arrays or matrices into random [train / test] subsets
from sklearn.model_selection import train_test_split

# [albumentations];
#   - library for image augmentation to increase the quality of trained models
#   - the purpose of image augmentation is to create the new training samples from the existing data

# +) image augmentation ~ the process of creating new images from an existing image data set
import albumentations as A

# To use the 'albumentation' in pytorch, convert the data type into 'torch'
# Convert image and mask to 'torch.Tensor'
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn

# [torch.optim];
#   - A package implementing various optimization algorithms
#   - to update the model's weight and bias
import torch.optim as optim
import torch.nn.functional as F

# [torch.utils.data.DataLoader];
#   - PyTorch data loading utility class
#   - It represents a Python iterable over a dataset
from torch.utils.data import DataLoader


from module.datasets import CustomDataset, get_dataset
from module.models import get_model
from module.utils import Config, seed_everything
from module.log import get_logger

class Trainer:
    def __init__(self, config: Config):
        self.config = config

        # path of CIFAR-10 dataset to be uploaded
        self.cifar10_tar_path = './data/cifar-10-python.tar.gz'
        self.extract_dir = './data/cifar10_extracted'

        self.model = get_model("custom")
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.lr_scheduler = None

    # Extract the '.tar.gz' file if not already extracted before
    def extract_and_load_cifar10(self):
        if not os.path.exists(self.extract_dir):
            with tarfile.open(self.cifar10_tar_path) as tar:
                tar.extractall(path=self.extract_dir)

        # Unpickle the CIFAR-10 dataset files
        # 'pickle' module:
        #   - datatype like list, class which are not text
        #   - to save or load these kinds of datatypes
        #   - 'unpickle' is opposite operation
        def unpickle(file):
            with open(file, 'rb') as un:
                dict = pickle.load(un, encoding='bytes')
            return dict
    
        # Load the CIFAR-10's 5 batches
        images = []
        labels = []

        # CIFAR-10 has 5 batches named data_batch_1 to data_batch_5
        for i in range(1, 6):  
            batch_file = os.path.join(self.extract_dir, 'cifar-10-batches-py', f'data_batch_{i}')
            batch = unpickle(batch_file)

            batch_images = batch[b'data']
            batch_labels = batch[b'labels']

            batch_images = batch_images.reshape(len(batch_images), 3, 32, 32).transpose(0, 2, 3, 1)

            images.append(batch_images)
            labels += batch_labels
        
        # The image data (in the form of numpy arrays) are appended to each 'images' and 'labels' lists
            
        # Merge the list of arrays into a single numpy array
        # size ~ (total_num_of_images_in all batches, 32, 32, 3)
        images = np.concatenate(images)

        # Labels are collected in a list, converted to a numpy array
        # for consistency and usage in pytorch data models or loaders
        labels = np.array(labels)

        # images ~ the array of image data
        # labels ~ the array of labels
        return images, labels
    
    def setup(self, mode="train"):
        """
        you need to code how to get data
        and define dataset, dataloader, transform in this function

        1) Sets a random seed
        2) Initialize the logger for tracking - optional
        3) Contains placeholders for loading data
            (ie. split a dataset into training and validation sets)
        4) Defines the data transformations for augmentation and preprocessing
            (ie. normalization, conversion to PyTorch tensors)
        5) Initialize the datasets and data-loaders for both training and validation data
        """
        if mode == "train":
            
            # Sets a random seed
            seed_everything(self.config.seed)

            # Initialize the logger for tracking via using tensorboard
            self.logger = get_logger(
                name="tensorboard",
                log_dir=f"{self.config.log_dir}",    
            )
            # Extract and load CIFAR-10 dataset
            # Get each arrays for images data and labels
            train_images, train_labels = self.extract_and_load_cifar10()

            # [train_test_split]
            #   - Split arrays or matrices into random train and test subsets

            #   - Params;
            #       * test_size: represent the proportion of the dataset to include in the test split
            #       * random_state: controls the shuffling

            # in this code, use this method to split the dataset into training and validation sets
            train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
                train_images, train_labels, test_size=0.2, random_state=self.config.seed)
            
            # Initialize the model / loss function / optimizer
            self.model = get_model("custom")
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

            # Train
            # Defining data transformation

            # [albumentations.Compose]
            #   - Compose transforms
            #   - Handle all transformations regarding bounding boxes
            train_transform = A.Compose([
                # add augmentation

                # # After set padding=4 the image become 40 x 40, random crop operator can get more result
                # A.RandomCrop(32, padding=4),

                # Flip the input horizontally around the y-axis common data augmentation technique
                A.HorizontalFlip(),

                # Normalization (mean, std, max_pixel_value)
                # A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),

                # Convert image and mask to 'torch.Tensor' (numpy array -> pytorch tensor)
                ToTensorV2()                                        
            ])

            # Validation
            val_transform = A.Compose([
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2()
            ])

            # Creating the dataset objects

            #   - creates instances of 'CustomDataset' class for training and validation datasets
            #   - passes the respective images, labels and transformations

            train_dataset = CustomDataset(
                images=train_imgs,
                labels=train_lbls,
                transforms=train_transform
            )

            val_dataset = CustomDataset(
                images=val_imgs,
                labels=val_lbls,
                transforms=val_transform
            )

            # Creating the dataloader objects

            #   - used to load the data in batches and optionally shuffle it
            #   cf) for the validation dataloader
            #       - the batch size is doubled
            #       - to speed up the validation, as no back-propagation is needed

            self.train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
            )
 
            self.val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=self.config.batch_size * 2,
                num_workers=self.config.num_workers,
                shuffle=False,
            )

        elif mode == "test":
            pass
    
    def train(self):

        # Moves the model to the specified device (ie. CPU or GPU)
        # To ensure the model computations are performed on the correct device
        self.model.to(self.config.device)

        best_acc = 0

        # Loops through each epoch for the number of epochs
        for epoch in range(self.config.epochs):
            self.model.train()

            # 'running_loss' ~ to accumulate the loss over the epoch
            running_loss = 0.0

            # Iterates over each batch of data
            # [tqdm] ~ to display a progress bar
            for batch in tqdm(self.train_dataloader):

                # Loads images and labels form the current batch
                # Moves them to the configured device
                imgs = batch["img"].to(self.config.device)

                # cf) Ensures that the labels are of type 'long'
                #       -> required for CrossEntropyLoss in PyTorch
                labels = batch["label"].to(self.config.device).long()
                
                # Clears old gradients from the previous step
                self.optimizer.zero_grad()

                # Pass the batch of images through the model
                outputs = self.model(imgs)

                # Compute the loss between 
                # the model predictions('outputs') || the true labels ('labels')
                loss = self.loss_fn(outputs, labels)

                # Perform the back_propagation starting from the loss
                loss.backward()

                # Updates the model parameters based on the current gradients
                self.optimizer.step()

                # Accumulates the loss over the epoch
                running_loss += loss.item()
            
            # Computes the average loss for the epoch
            avg_loss = running_loss / len(self.train_dataloader)

            # Calls the '_valid' method, compute the 'validation accuracy'
            val_acc = self._valid()
            
            # Update the best model so far with the best validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f'best_model.pth')

            # Logging
            self.logger.add_scalar('Training Loss', avg_loss, epoch)
            self.logger.add_scalar('Validation Accuracy', val_acc, epoch)

            print(f'Epoch [{epoch+1}/{self.config.epochs}], Loss: {avg_loss:.4f}, , Validation Acc: {val_acc:.4f}')

    def _valid(self):
        """Evaluating the model on the validation dataset
        """

        # Set the module in evaluation mode
        self.model.eval()

        # Set some variables for computing the accuracy of the validation set
        correct = 0
        total = 0

        # [torch.no_grad()];
        #   - Context-manager that disables gradient calculation
        #   - This reduces memory usage and speeds up computation
        #   - since gradients are not needed for model evaluation
        with torch.no_grad():

            # Iterates over each batch of data
            for batch in tqdm(self.val_dataloader):

                # Loads images and labels from the current batch
                # Moves them to the configured device.
                imgs = batch["img"].to(self.config.device)
                labels = batch["label"].to(self.config.device)

                # Pass the batch of images through the model
                outputs = self.model(imgs)

                # [torch.max];
                #   - returns the indices of the maximum values along dimension 1 (ie. class dimension)
                #   - represents ""the models' predicted classes"
                _, predicted = torch.max(outputs.data, 1)

                # Updates the count by adding the number of samples in the current batch
                total += labels.size(0)

                # Track the total number of correct predictions
                correct += (predicted == labels).sum().item()

        # Calculates the validation accuracy by dividing
        val_acc = correct / total
        return val_acc

import sys, os
# sys.path.append(os.pardir)
# sys.path.insert(0, "..")

from glob import glob
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# from base.module.datasets import get_dataset
from module.datasets import get_dataset
from module.models import get_model
from module.utils import Config, seed_everything
from module.log import get_logger
from data.cifar10.cifar10 import load_train_data, load_test_data


class Trainer:
    def __init__(self, config: Config):
        self.config = config
    
    def setup(self, mode="train"):
        """
        you need to code how to get data
        and define dataset, dataloader, transform in this function
        """
        if mode == "train":

            seed_everything(self.config.seed) # for reproducible result

            self.logger = get_logger(
                name="tensorboard",
                log_dir=f"{self.config.log_dir}",    
            )

            ## TODO ##
            # Hint : get data by using pandas or glob 
            X, y = load_train_data(self.config.data_path)

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)


            # Train
            train_transform = A.Compose([
                # add augmentation
                A.Normalize(),
                ToTensorV2()
            ])

            train_dataset = get_dataset(
                "custom",
                imgs=X_train,
                labels=y_train,
                transforms=train_transform
            )

            self.train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
            )
            
            # Validation
            val_transform = A.Compose([
                A.Normalize(),
                ToTensorV2()
            ])

            val_dataset = get_dataset(
                "custom",
                imgs=X_valid,
                labels=y_valid,
                transforms=val_transform
            )         

            self.val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=self.config.batch_size * 2,
                num_workers=self.config.num_workers,
                shuffle=False,
            )

            # Model
            self.model = get_model("custom")
            
            # load model
            
            # Loss function
            self.loss_fn = nn.CrossEntropyLoss() # Softmax 포함하고 있음

            # Optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

            # LR Scheduler
            self.lr_scheduler = None

        elif mode == "test":
            X_test, y_test = load_test_data()


    def train(self):
        self.model.to(self.config.device)
        
        # early stopping
        early_stopping = 0

        # metric
        best_acc = 0
        best_f1 = 0

        best_model = None
        total_batch = len(self.train_dataloader)

        for epoch in range(1, self.config.epochs+1):
            self.model.train()
            train_loss = 0
            train_correct = 0

            for batch in tqdm(self.train_dataloader):
                
                ## TODO ##
                # ----- Modify Example Code -----
                # following code is pesudo code
                # modify the code to fit your task
                img = batch["img"].to(self.config.device)
                label = batch["label"].to(self.config.device).long()
                
                self.optimizer.zero_grad()
                pred = self.model(img)

                loss = self.loss_fn(pred, label)
                loss.backward()
                
                # calculate metric
                
                self.optimizer.step()
                # -------------------------------

                with torch.no_grad():
                    classfication = torch.argmax(pred, dim=1) == label
                    train_correct += classfication.float().mean()
                    train_loss += loss.item()


            print(f'Epoch : {epoch}, train loss = {train_loss/len(self.train_dataloader):.6f}, train accuracy = {train_correct/len(self.train_dataloader):.6f}')

            self._valid()

            # logging

            # save model
            os.makedirs(f"./model/cifar10", exist_ok=True)
            torch.save(self.model.state_dict(), f"./model/cifar10/{epoch}_cifar10.pth")
            
            if early_stopping >= 5:
                break
            
            
    def _valid(self):
        # metric
        valid_loss = 0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader):
                img = batch["img"].to(self.config.device)
                label = batch["label"].to(self.config.device).long()

                pred = self.model(img)
                loss = self.loss_fn(pred, label)

                valid_loss += loss.item()
                correct_prediction = torch.argmax(pred, 1) == label
                accuracy = correct_prediction.float().mean()

            print(f"valid loss = {valid_loss/len(self.val_dataloader):.6f},  valid accuracy : {accuracy.item()}")


                # logging

    def test(self):
        pass

        
    
if __name__ == '__main__':
    exit()
    pass
    # trainer = Trainer()
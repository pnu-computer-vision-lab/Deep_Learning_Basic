import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def get_model(name: str, **kwargs):
    if name == "custom":
        return CustomModel(**kwargs)
    else:
        raise ValueError("Incorrect Name")


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # super(CustomModel, self).__init__() # 뭔 차이지?_?
        # define DL layer
        # Similar to AlexNet
        # Image input size : cifar-10(3, 32, 32)
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64x15x15
        )
        # input : 64x15x15
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(), # 192x15x15
            nn.MaxPool2d(kernel_size=3, stride=2) # 192x7x7
        )
        # input : # 192x7x7
        self.convLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU() # 384x7x7
        )
        # input : # 384x7x7
        self.convLayer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        # input : # 384x7x7
        self.convLayer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(), # 256x7x7
            nn.MaxPool2d(kernel_size=2, stride=1)  # 256x6x6
        )

        self.fc6 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU()
        )
        self.fc7 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.fc8 = nn.Linear(4096, 10)

        # nn.init.kaiming_uniform(self.lastLayer.weight)

    def forward(self, x):
        # using module that you define in __init__ and check whether it's right sequence

        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = self.convLayer4(out)
        out = self.convLayer5(out)

        out = out.view(out.size(0), -1) # Flatten the for FClayers

        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)

        return out


if __name__ == "__main__":
    # Check the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomModel()
    model.to(device)
    summary(model, input_size=(1, 3, 32, 32))
    # pass

# 'timm' module:
# to use the pre-trained model in the pytorch
import timm

# 'torch.nn' module:
# helps to create and train the neural network
import torch.nn as nn

# 'torch.nn.functional' module:
#   used to train and build the layers of neural networks (eg. input, hidden and output)
import torch.nn.functional as F


# 'torchinfo' module:
# the 'torchinfo' provides information complementary to what is provided by print(model) in PyTorch
from torchinfo import summary

def get_model(name: str, **kwargs):
    """Returns an instance of 'CustomModel'
    """
    # 'custom' is passed as the name parameter
    if name == "custom":
        return CustomModel(**kwargs)
    else:
        raise ValueError("Incorrect Name")


class CustomModel(nn.Module):
    """A subclass of 'torch.nn.Module'

    intended to represent a custom deep learning model

    +) [nn.Module]
        
        - Base class for all neural network modules
        - Each own models should be also subclass this class
    """
    def __init__(self):
        """Calls the superclass initializer
        """
        super(CustomModel, self).__init__()
        # Define DL layer

        # Define the layers of the CNN
        # Convolutional layers

        # [nn.Conv2d];
        #   - Applies a 2D-convolution over an input signal
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max-pooling

        # [nn.MaxPool2d];
        #   - Applies a 2D-max pooling over an input signal composed of several input planes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Batch normalization

        # [nn.BatchNorm2d];
        #   - Applies batch normalization over a 4D input
        #   - a mini-batch of 2D inputs with additional channel dimension
        #   - Args denotes the 'number of features', especially an expected input of size
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Dropout

        # [nn.Dropout];
        #   - During the training, randomly 'zeroes' some of the elements of the input tensor with probability
        #   - The zeroed elements are chosen independently for each forward call
        #   - being sampled from a Bernoulli distribution

        #   - Args: p ~ probability of an element to be zeroed
        self.dropout = nn.Dropout(0.5)


        # Fully connected layers

        # [nn.Linear];
        #   - Applies a linear transformation to the incoming data

        #   - Args:
        #       * in_features: size of each input sample
        #       * out_features: size of each output sample
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)

        # Final fully-connected layer resulting the final 10 labels
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        # using module that you define in __init__ and check whether it's right sequence

        # Apply convolutional layers with ReLU and max pooling

        # [F.relu()];
        #   - Applies the rectified linear unit function element-wise
        #   - uses 'torch.nn.ReLU'

        #   - Args:
        #       * input: Tensor
        #       * bool: can optionally do the operation in-place
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the feature maps

        # [torch.tensor.view];
        #   - method in PyTorch, defined in PyTorch's 'Tensor' class
        #   - used to reshape a tensor without changing its data
        #   - returns a new tensor with the same data as the self tensor but of a different shape

        #   - Args:
        #       * torch.Size or int: the desired size
        #       * -1: must be divisible by the ration between the element sizes of the d-types

        # the desired size '128*4*4': size of the second dimension of the new tensor
        #   - 128: the number of output channels from the third convolution layer
        #   - 4*4: the size of the feature map after three pooling operations with 2x2 window on 32x32 input image

        # - 32/2/2/2 = 4 ~ after the 3 pooling layers, the 32x32 image is down-sampled into 4x4 image
        # - 128 = there are 128 such feature maps
        x = x.view(-1, 128 * 4 * 4)

        # Apply fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))

        # Apply the dropout to the result of the ReLU activation to regularize the network and prevent overfitting
        # by randomly zeroing some elements with probability 0.5
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        # the output is passed through the third fully connected layer
        # to produce the final output logits
        x = self.fc3(x)
        return x
    

if __name__ == "__main__":
    # Check the model
    
    # model = CustomModel()
    # summary(model, input_size=(1, 3, 32, 32))

    # [torchinfo.summary()];
    #   - summarize the given PyTorch model

    #   1) Layer names,
    #   2) input/output shapes,
    #   3) kernel shape,
    #   4) # of parameters,
    #   5) # of operations (Mult-Adds),
    #   6) whether layer is trainable

    #   - Args;
    #       * model: nn.Module
    #       * input_size: INPUT_SIZE_TYPE (; Sequence[Union[int, Sequence[Any], torch.Size]])

    #   1 - Batch size: the number of samples that will be passed through the network at one time
    #   3 - Channels: the number of channels in the input images (ie. RGB channels)
    #   32x32 - Height x Width: CIFAR-10 image pixel size
    pass

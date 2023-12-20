
## Separable Conv = Depthwise + Pointwise + Eliminate 5 layers + Last Conv Channel Reduction + max pool
import torch
import torch.nn as nn
from torchsummary import summary

class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1)
        )

        #self.bn = nn.BatchNorm2d(output_channels)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        #x = self.bn(x)
        #x = self.relu(x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin 
                         a network uniformly at each layer. For a given 
                         layer and width multiplier α, the number of 
                         input channels M becomes αM and the number of 
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=10):
       super().__init__()
    # change the input channel size from 3 to 1
       alpha = width_multiplier
       self.stem = nn.Sequential(
           BasicConv2d(1, int(32 * alpha), 3, padding=1, bias=False),
           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               padding=1,
               bias=False
           ),
           nn.BatchNorm2d(int(64 * alpha)),
           nn.ReLU(inplace=True)
       )

       #downsample with stride=2 #nn.MaxPool2d
       self.conv1 = nn.Sequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=1,#stride=2
               padding=1,
               bias=False
           ),
           nn.BatchNorm2d(int(128 * alpha)),
           nn.MaxPool2d(3, 2),#output_ratio = 1 / stride
           nn.ReLU(inplace=True),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           ),
           nn.BatchNorm2d(int(128 * alpha)),
           nn.ReLU(inplace=True)
       )

       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=1,#stride=2
               padding=1,
               bias=False
           ),
           nn.BatchNorm2d(int(256 * alpha)),
           nn.MaxPool2d(3, 2),
           nn.ReLU(inplace=True),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           ),
           nn.BatchNorm2d(int(256 * alpha)),
           nn.ReLU(inplace=True)
       )

       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=1,#stride=2
               padding=1,
               bias=False
           ),
           nn.BatchNorm2d(int(512 * alpha)),
           nn.MaxPool2d(3, 2),
           nn.ReLU(inplace=True)
        )

       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=1,#stride=2
               padding=1,
               bias=False
           ),
           nn.BatchNorm2d(int(1024 * alpha)),
           nn.ReLU(inplace=True),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(512 * alpha), #1024 -> 512
               3,
               padding=1,
               bias=False
           ),
           nn.BatchNorm2d(int(512 * alpha)),
           nn.ReLU(inplace=True)
       )

       self.fc = nn.Linear(int(512 * alpha), class_num) #1024 -> 512
       self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet(alpha=1, class_num=10):
    return MobileNet(alpha, class_num)

# check model summary
# "Model size" refers to the number of trainable parameters in the model?
net=mobilenet()
print(summary(net, (1,28,28)))
# (3,32,32) -> grayscale (1,28,28)

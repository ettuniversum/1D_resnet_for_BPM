import torch
import torch.nn as nn

class ResNetBlock1D(nn.Module):
    """
    1D Residual Block with two 1D convolutional layers, batch normalization,
    and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock1D, self).__init__()
        # Large kernel size (15) is typical for filtering PPG/ECG signals
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=15,
            stride=stride,
            padding=7,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=15,
            stride=1,
            padding=7,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """
    1D Residual Network for regression tasks (e.g., estimating BPM from PPG segments).
    Inputs: shape (batch, 1, 1000)
    Outputs: shape (batch, 1)
    """
    def __init__(self, block, layers, num_classes=1):
        super(ResNet1D, self).__init__()
        self.in_channels = 32
        
        # Initial projection & downsampling
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=15,
            stride=2,
            padding=7,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Stages
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        # Output layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch, 1, 1000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_1d(num_classes=1):
    """Returns a 1D ResNet-18 model."""
    return ResNet1D(ResNetBlock1D, [2, 2, 2, 2], num_classes=num_classes)


def resnet10_1d(num_classes=1):
    """Returns a 1D ResNet-10 model (lighter version for edge devices)."""
    return ResNet1D(ResNetBlock1D, [1, 1, 1, 1], num_classes=num_classes)


if __name__ == "__main__":
    # Quick sanity check
    model = resnet10_1d()
    test_input = torch.randn(2, 1, 1000)
    test_output = model(test_input)
    print("ResNet-1D architecture initialized successfully.")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")

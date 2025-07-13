import torch.nn as nn
import torch.nn.init as init

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, first_block=False):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=bias)      
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)     

        self.first_block = first_block

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        identity = x        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if not self.first_block:
            x += identity
        x = self.relu(x)
        
        return x

class ECGResNet(nn.Module):
    def __init__(self, feature_dim=256):
        super(ECGResNet, self).__init__()
        
        self.conv = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.layer1 = self.make_layer(32, 32, 4)
        self.layer2 = self.make_layer(32, 64, 4)
        self.layer3 = self.make_layer(64, 128, 4)
        self.layer4 = self.make_layer(128, feature_dim, 4)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.conv.weight)

    def make_layer(self, in_channels, out_channels, blocks):
        layers = []
        
        # First block
        layers.append(ResidualBlock1D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, first_block=True))

        # Remaining blocks
        for _ in range(0, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
    
        x = self.layer1(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        x = self.maxpool(x)

        x = self.layer3(x)
        x = self.maxpool(x)

        x = self.layer4(x)
        x = self.avgpool(x)  # Global average pooling

        return x

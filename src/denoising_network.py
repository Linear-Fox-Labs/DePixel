import torch
import torch.nn as nn

class DualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(DualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out + residual

class RecursiveResidualGroup(nn.Module):
    def __init__(self, channels):
        super(RecursiveResidualGroup, self).__init__()
        self.dab1 = DualAttentionBlock(channels)
        self.dab2 = DualAttentionBlock(channels)

    def forward(self, x):
        out = self.dab1(x)
        out = self.dab2(out)
        return out + x

class DenoisingNetwork(nn.Module):
    def __init__(self, channels=64):
        super(DenoisingNetwork, self).__init__()
        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.rrg1 = RecursiveResidualGroup(channels)
        self.rrg2 = RecursiveResidualGroup(channels)
        self.rrg3 = RecursiveResidualGroup(channels)
        self.rrg4 = RecursiveResidualGroup(channels)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.rrg1(out)
        out = self.rrg2(out)
        out = self.rrg3(out)
        out = self.rrg4(out)
        out = self.conv_out(out)
        return out
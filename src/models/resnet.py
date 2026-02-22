import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.
    Uses batch normalization and ReLU activation.
    """
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Helps the network focus on important feature channels.
    """
    def __init__(self, num_filters, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_filters, num_filters // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters // reduction, num_filters, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch, channels, _, _ = x.size()
        
        y = self.squeeze(x).view(batch, channels)
        
        y = self.excitation(y).view(batch, channels, 1, 1)
        
        return x * y.expand_as(x)


class ResidualBlockSE(nn.Module):
    """
    Residual block with Squeeze-and-Excitation for better performance.
    """
    def __init__(self, num_filters):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SEBlock(num_filters)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out


class ChessResNet(nn.Module):
    """
    ResNet architecture for chess position evaluation.
    
    Input: (batch, 18, 8, 8) - 18 bitplanes representing chess position
    Output: (batch, 1) - tanh-normalized evaluation (-1 to 1)
    
    Args:
        num_blocks: Number of residual blocks (default: 15 for 16M positions)
        num_filters: Number of convolutional filters (default: 256)
        use_se: Whether to use Squeeze-and-Excitation blocks (default: True)
        value_head_hidden: Hidden layer size in value head (default: 256)
    """
    def __init__(self, num_blocks=15, num_filters=256, use_se=True, value_head_hidden=256):
        super(ChessResNet, self).__init__()
        
        
        self.input_conv = nn.Conv2d(18, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        
        block_class = ResidualBlockSE if use_se else ResidualBlock
        self.residual_blocks = nn.ModuleList([
            block_class(num_filters) for _ in range(num_blocks)
        ])
        
        
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, value_head_hidden)
        self.value_fc2 = nn.Linear(value_head_hidden, 1)
        
    def forward(self, x):
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        # v = torch.tanh(v)
        
        return v

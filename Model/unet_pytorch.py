"""
U-Net Implementation for Porosity Detection in X-ray Tomography Images
Based on the original U-Net architecture (Ronneberger et al., 2015)
Adapted for materials science applications

References:
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks 
  for Biomedical Image Segmentation. arXiv:1505.04597.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DoubleConv(nn.Module):
    """Double convolution block: (convolution => BN => ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # Use bilinear upsampling or transpose convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for porosity segmentation
    
    Args:
        n_channels: Number of input channels (1 for grayscale X-ray images)
        n_classes: Number of output classes (2 for binary porosity segmentation)
        bilinear: Use bilinear upsampling instead of transpose convolutions
    """
    
    def __init__(self, n_channels: int = 1, n_classes: int = 2, bilinear: bool = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (expansive path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        self.inc = torch.utils.checkpoint.checkpoint_sequential(self.inc, 1)
        self.down1 = torch.utils.checkpoint.checkpoint_sequential(self.down1, 1)
        self.down2 = torch.utils.checkpoint.checkpoint_sequential(self.down2, 1)
        self.down3 = torch.utils.checkpoint.checkpoint_sequential(self.down3, 1)
        self.down4 = torch.utils.checkpoint.checkpoint_sequential(self.down4, 1)


def create_unet_for_porosity(input_size: Tuple[int, int] = (512, 512), 
                            device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> UNet:
    """
    Create a U-Net model specifically configured for porosity detection
    
    Args:
        input_size: Expected input image size (height, width)
        device: Device to run the model on
    
    Returns:
        Configured U-Net model
    """
    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    model = model.to(device)
    
    print(f"Created U-Net model for porosity detection:")
    print(f"- Input size: {input_size}")
    print(f"- Input channels: 1 (grayscale)")
    print(f"- Output classes: 2 (background, pore)")
    print(f"- Device: {device}")
    print(f"- Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_unet_for_porosity()
    
    # Test with dummy input
    x = torch.randn(1, 1, 512, 512)
    if torch.cuda.is_available():
        x = x.cuda()
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test successful!")

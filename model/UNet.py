import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp

from data import DRIVEDataset

class SampleUNet(nn.Module):
    """
    # Example usage:
    # Create an instance of UNet
    model = SampleUNet(in_channels=3, out_channels=1)
    # Print the model architecture
    print(model)
    """

    def __init__(self, in_channels, out_channels):
        super(SampleUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Down-sampling path
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Up-sampling path
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Down-sampling
        # Ensure input tensor x is 4D (batch_size, channels, height, width)
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor with 4 dimensions, but got {x.dim()} dimensions.")

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Up-sampling
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Final convolution
        x = self.final_conv(x)
        x = F.sigmoid(x)  # Use sigmoid activation for binary segmentation
        
        # Convert to binary segmentation output
        # x = (x[:, 0, :, :] > x[:, 1, :, :]).float()  # Compare the two channels and convert to float


        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        # Adjust dimensions of x1 to match x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class ResUNet(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super(ResUNet, self).__init__()
        
        # Load ResNet backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            encoder_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            encoder_channels = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported. Choose from ['resnet18', 'resnet34', 'resnet50', 'resnet101']")
        
        self.encoder1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool, base_model.layer1)
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4
        
        # Up-sampling path with dynamic channel adjustment
        if backbone in ['resnet18', 'resnet34']:
            self.up1 = UpBlock(encoder_channels[4], encoder_channels[3])
            self.up2 = UpBlock(encoder_channels[3], encoder_channels[2])
            self.up3 = UpBlock(encoder_channels[2], encoder_channels[1])
            self.up4 = UpBlock(encoder_channels[1], encoder_channels[0])
        elif backbone in ['resnet50', 'resnet101']:
            self.up1 = UpBlock(encoder_channels[3], encoder_channels[2])
            self.up2 = UpBlock(encoder_channels[2], encoder_channels[1])
            self.up3 = UpBlock(encoder_channels[1], encoder_channels[0])
            self.up4 = UpBlock(encoder_channels[0], encoder_channels[0])
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported.")
        
        self.final_conv = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)
    
    def forward(self, x0):
        # Encoder
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # Final convolution
        x = self.final_conv(x)
        x = F.sigmoid(x)  # Use sigmoid activation for binary segmentation

        return x
        
        
class ResNet34Unet(smp.Unet):
    def __init__(self):
        super(ResNet34Unet, self).__init__()

        self.model = smp.Unet(encoder_name='resnet34',
                              in_channels=3, classes=1, activation=None)

    def forward(self, x):
        return self.model.forward(x)

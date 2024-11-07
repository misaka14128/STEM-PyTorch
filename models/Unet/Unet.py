import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(1, 32)  # 256*256*1 to 256*256*32
        self.conv2 = DoubleConv(32, 64)  # 128*128*32 to 128*128*64
        self.conv3 = DoubleConv(64, 128)  # 64*64*64 to 64*64*128
        self.conv4 = DoubleConv(128, 256)  # 32*32*128 to 32*32*256
        self.upsample1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)  # 64*64*128
        self.upsample2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 128*128*64
        self.upsample3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 256*256*32
        self.upconv1 = DoubleConv(128+128, 128)
        self.upconv2 = DoubleConv(64+64, 64)
        self.upconv3 = DoubleConv(32+32, 32)
        self.finalconv = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv2(x)
        x = self.maxpool(conv2)

        conv3 = self.conv3(x)
        x = self.maxpool(conv3)

        conv4 = self.conv4(x)
        x = self.upsample1(conv4)
        x = torch.cat([x, conv3], dim=1)

        upconv1 = self.upconv1(x)
        x = self.upsample2(upconv1)
        x = torch.cat([x, conv2], dim=1)

        upconv2 = self.upconv2(x)
        x = self.upsample3(upconv2)
        x = torch.cat([x, conv1], dim=1)

        x = self.upconv3(x)
        out = self.finalconv(x)

        return out


class Unet_2x(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(1, 32)  # 256*256*1 to 256*256*32
        self.conv2 = DoubleConv(32, 64)  # 128*128*32 to 128*128*64
        self.conv3 = DoubleConv(64, 128)  # 64*64*64 to 64*64*128
        self.conv4 = DoubleConv(128, 256)  # 32*32*128 to 32*32*256
        self.upsample1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)  # 64*64*128
        self.upsample2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 128*128*64
        self.upsample3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 256*256*32
        self.upsample4 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)  # 512*512*32
        self.upconv1 = DoubleConv(128+128, 128)
        self.upconv2 = DoubleConv(64+64, 64)
        self.upconv3 = DoubleConv(32+32, 32)
        self.finalconv = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        origin = x
        conv1 = self.conv1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv2(x)
        x = self.maxpool(conv2)

        conv3 = self.conv3(x)
        x = self.maxpool(conv3)

        conv4 = self.conv4(x)
        x = self.upsample1(conv4)
        x = torch.cat([x, conv3], dim=1)

        upconv1 = self.upconv1(x)
        x = self.upsample2(upconv1)
        x = torch.cat([x, conv2], dim=1)

        upconv2 = self.upconv2(x)
        x = self.upsample3(upconv2)
        x = torch.cat([x, conv1], dim=1)
        x = self.upconv3(x)

        x = self.upsample4(x)
        out = self.finalconv(x)

        return out

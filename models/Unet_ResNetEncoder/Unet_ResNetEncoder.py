import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class UNetWithResNetEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 使用ResNet34作为编码器
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # 使用ResNet的前几层作为编码器
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 输入到ResNet的第一部分
        self.encoder1 = resnet.layer1  # ResNet block 1
        self.encoder2 = resnet.layer2  # ResNet block 2
        self.encoder3 = resnet.layer3  # ResNet block 3
        self.encoder4 = resnet.layer4  # ResNet block 4

        # 解码器部分
        self.upsample1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)  # 上采样 ResNet最后的输出
        self.upsample2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.upsample4 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.upsample5 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

        # 上采样后的卷积操作 (Up Convolutions)
        self.upconv1 = DoubleConv(512, 256)
        self.upconv2 = DoubleConv(256, 128)
        self.upconv3 = DoubleConv(128, 64)
        self.upconv4 = DoubleConv(128, 64)

        # 最后的输出层
        self.finalconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器（ResNet部分）
        enc0 = self.encoder0(x)  # 输入通过ResNet的第一部分 64*64*64
        enc1 = self.encoder1(enc0)  # ResNet block 1 64*64*64
        enc2 = self.encoder2(enc1)  # ResNet block 2 128*32*32
        enc3 = self.encoder3(enc2)  # ResNet block 3 256*16*16
        enc4 = self.encoder4(enc3)  # ResNet block 4 512*8*8

        # 解码器
        x = self.upsample1(enc4)  # 上采样 256*16*16
        x = torch.cat([x, enc3], dim=1)  # 跳跃连接 512*16*16
        x = self.upconv1(x)  # 解码卷积 256*16*16

        x = self.upsample2(x)  # 上采样 128*32*32
        x = torch.cat([x, enc2], dim=1)  # 跳跃连接 256*32*32
        x = self.upconv2(x)  # 解码卷积 128*32*32

        x = self.upsample3(x)  # 上采样 64*64*64
        x = torch.cat([x, enc1], dim=1)  # 跳跃连接 128*64*64
        x = self.upconv3(x)  # 解码卷积 64*64*64

        x = torch.cat([x, enc0], dim=1)  # 跳跃连接 128*64*64
        x = self.upconv4(x)  # 解码卷积 64*64*64

        x = self.upsample4(x)  # 上采样 64*128*128
        x = self.upsample5(x)  # 上采样 64*256*256

        # 最后的卷积输出层
        out = self.finalconv(x)
        return out

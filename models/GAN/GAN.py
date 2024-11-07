import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),  # 256*256*1 to 256*256*32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.donwconv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 256*256*32 to 128*128*32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),  # 128*128*32 to 128*128*64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.donwconv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 128*128*64 to 64*64*64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),  # 64*64*64 to 64*64*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.downconv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 64*64*128 to 32*32*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),  # 32*32*128 to 32*32*256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),  # 32*32*256 to 64*64*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),  # 64*64*128 to 128*128*64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),  # 128*128*64 to 256*256*32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.upconv1 = nn.Sequential(
            nn.Conv2d(128+128, 128, kernel_size=3, padding=1, bias=False),  # 64*64*256 to 64*64*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1, bias=False),  # 128*128*128 to 128*128*64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.upconv3 = nn.Conv2d(32+32, 1, kernel_size=3, padding=1, bias=False)  # 256*256*64 to 256*256*1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)  # 256*256*1 to 256*256*32
        downconv1 = self.donwconv1(conv1)  # 256*256*32 to 128*128*32
        conv2 = self.conv2(downconv1)  # 128*128*32 to 128*128*64
        downconv2 = self.donwconv2(conv2)  # 128*128*64 to 64*64*64
        conv3 = self.conv3(downconv2)  # 64*64*64 to 64*64*128
        downconv3 = self.downconv3(conv3)  # 64*64*128 to 32*32*128
        conv4 = self.conv4(downconv3)  # 32*32*128 to 32*32*256
        upsample1 = self.upsample1(conv4)  # 32*32*256 to 64*64*128
        cat1 = torch.cat([upsample1, conv3], dim=1)  # 64*64*256
        upconv1 = self.upconv1(cat1)  # 64*64*256 to 64*64*128
        upsample2 = self.upsample2(upconv1)  # 64*64*128 to 128*128*64
        cat2 = torch.cat([upsample2, conv2], dim=1)  # 128*128*128
        upconv2 = self.upconv2(cat2)  # 128*128*128 to 128*128*64
        upsample3 = self.upsample3(upconv2)  # 128*128*64 to 256*256*32
        cat3 = torch.cat([upsample3, conv1], dim=1)  # 256*256*64
        upconv3 = self.upconv3(cat3)  # 256*256*64 to 256*256*1
        out = self.sigmoid(upconv3)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 256*256*1 to 128*128*64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 128*128*64 to 64*64*128
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 64*64*128 to 32*32*256
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),  # 32*32*256 to 32*32*512
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=False),  # 32*32*512 to 16*16*1
            nn.Sigmoid()
        )

    def forward(self, input, target):
        x = torch.cat([input, target], dim=1)
        out = self.model(x)
        return out

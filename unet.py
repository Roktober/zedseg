import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, layers=2):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            *sum([[
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ] for i in range(layers)], [])
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, layers=2):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(in_ch, out_ch, layers)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, layers=2, bilinear=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = Conv(in_ch, out_ch, layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


UNetLayerInfo = namedtuple('UNetLayerInfo', 'down_channels down_layers up_channels up_layers')
DownInfo = namedtuple('DownInfo', 'in_ch out_ch layers')
UpInfo = namedtuple('UpInfo', 'in_ch out_ch layers')


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, downs, ups):  #  =[(64, 64), (128, ), (256, ), (512, )]):
        super(UNet, self).__init__()
        # down_channels = list([li.down_channels for li in arch])
        self.inc = Conv(n_channels, downs[0][0])
        self.downs = [Down(pr_c, nx_c, pr_l) for (pr_c, pr_l), (nx_c, _) in zip(downs[:-1], downs[1:])]

        self.ups = [
            Up(downs[i][0] + (ups[i + 1][0] if i < len(ups) - 1 else downs[-1][0]), out_c, layers)
            for i, (out_c, layers) in enumerate(ups)
        ]

#        self.down1 = Down(64, 128)
#        self.down2 = Down(128, 256)
#        self.down3 = Down(256, 512)
#        self.down4 = Down(512, 512)
#        self.up1 = Up(1024, 256)
#        self.up2 = Up(512, 128)
#        self.up3 = Up(256, 64)
#        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(ups[0][0], n_classes, 1)

    def forward(self, x):
        xs = [None] * (len(self.downs) + 1)
        xs[0] = self.inc(x)
        for i, down in enumerate(self.downs):
            xs[i + 1] = down(xs[i])
        x = xs[-1]
        for i, up in reversed(list(enumerate(self.ups))):
            x = up(x, xs[i])

#        x1 = self.inc(x)
#        x2 = self.down1(x1)
#        x3 = self.down2(x2)
#        x4 = self.down3(x3)
#        x5 = self.down4(x4)
#        x = self.up1(x5, x4)
#        x = self.up2(x, x3)
#        x = self.up3(x, x2)
#        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

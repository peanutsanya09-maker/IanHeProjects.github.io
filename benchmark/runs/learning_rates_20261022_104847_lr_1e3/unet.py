import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace = True),

            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 沿通道拼接编码器特征和上采样特征。
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """基础 U-Net：输入的高、宽须为 16 的倍数。"""

    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # 返回 logits，交叉熵损失直接使用这些分数。
        return self.outc(x)


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)

    model = UNet(in_channels=3, num_classes=2)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        masks = logits.argmax(dim=1)

    print("原始输入：", x.shape)
    print("类别分数：", logits.shape)
    print("预测掩码：", masks.shape)

    

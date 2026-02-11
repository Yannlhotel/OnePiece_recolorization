import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False) if down else
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        # Encoder (downsampling path)
        self.down1 = UNetBlock(in_channels, 64, down=True)
        self.down2 = UNetBlock(64, 128, down=True)
        self.down3 = UNetBlock(128, 256, down=True)
        self.down4 = UNetBlock(256, 512, down=True)
        self.down5 = UNetBlock(512, 512, down=True)
        self.down6 = UNetBlock(512, 512, down=True)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (upsampling path)
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True) # Skip connection: 512+512=1024
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True) # Skip connection: 512+512=1024
        self.up4 = UNetBlock(1024, 256, down=False) # 512+512=1024
        self.up5 = UNetBlock(512, 128, down=False)   # 256+256=512
        self.up6 = UNetBlock(256, 64, down=False)    # 128+128=256

        # Final convolution
        self.final_up = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1) # 64+64=128
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        # Bottleneck
        bottleneck = self.bottleneck(d6)

        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        
        # Final output
        final_out = self.final_up(torch.cat([u6, d1], 1))
        return self.tanh(final_out)
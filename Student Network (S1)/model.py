import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_small_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 16)  # Reduced channels
        self.e2 = encoder_block(16, 32)  # Reduced channels

        """ Bottleneck """
        self.b = conv_block(32, 64)  # Reduced channels

        """ Decoder """
        self.d1 = decoder_block(64, 32)  # Reduced channels
        self.d2 = decoder_block(32, 16)  # Reduced channels

        """ Classifier """
        self.outputs = nn.Conv2d(16, 1, kernel_size=1, padding=0)  # Reduced channels

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        d1 = self.d1(b, s2)
        d2 = self.d2(d1, s1)

        outputs = self.outputs(d2)

        return outputs

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = build_small_unet()
    y = f(x)
    print(y.shape)

import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
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
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class build_unet(nn.Module):
    def __init__(self, num_classes_segmentation=1, num_channels_reconstruction=3):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Segmentation Decoder """
        self.d1_seg = decoder_block(1024, 512)
        self.d2_seg = decoder_block(512, 256)
        self.d3_seg = decoder_block(256, 128)
        self.d4_seg = decoder_block(128, 64)

        """ Reconstruction Decoder """
        self.d1_recon = decoder_block(1024, 512)
        self.d2_recon = decoder_block(512, 256)
        self.d3_recon = decoder_block(256, 128)
        self.d4_recon = decoder_block(128, 64)

        """ Classifiers """
        self.segmentation_output = nn.Conv2d(64, num_classes_segmentation, kernel_size=1, padding=0)
        self.reconstruction_output = nn.Conv2d(64, num_channels_reconstruction, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Segmentation Decoder """
        d1_seg = self.d1_seg(b, s4)
        d2_seg = self.d2_seg(d1_seg, s3)
        d3_seg = self.d3_seg(d2_seg, s2)
        d4_seg = self.d4_seg(d3_seg, s1)

        """ Reconstruction Decoder """
        d1_recon = self.d1_recon(b, s4)
        d2_recon = self.d2_recon(d1_recon, s3)
        d3_recon = self.d3_recon(d2_recon, s2)
        d4_recon = self.d4_recon(d3_recon, s1)

        segmentation_output = self.segmentation_output(d4_seg)
        reconstruction_output = self.reconstruction_output(d4_recon)

        return segmentation_output, reconstruction_output
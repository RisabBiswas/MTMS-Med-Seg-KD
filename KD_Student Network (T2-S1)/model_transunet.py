import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import repeat, rearrange
import os
import random
from typing import List, Tuple
import numpy as np
from einops import repeat
import math

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x,mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=8, heads=12, dim_head=None, dim_linear_block=2048, dropout=0, prenorm=False):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head,
                                            dim_linear_block, dropout, prenorm=prenorm) for _ in range(blocks)]
        print("Block/Layers: ", blocks)
        print("Heads: ", heads)
        print("Dimension: ", dim_linear_block)
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=2048, dropout=0.1, activation=nn.GELU, gf = None,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if self.prenorm:
            y = self.drop(self.mhsa(self.norm_1(x), mask)) + x      
            out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
            out = self.norm_2(self.linear(y) + y)
        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 3) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=3, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class SignleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(SignleConv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(SignleConv(in_ch, out_ch, norm_layer),
                                  SignleConv(out_ch, out_ch, norm_layer))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """
    Doubles spatial size with bilinear upsampling
    Skip connections and double convs
    """

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        mode = "bilinear"
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        """
        Args:
            x1: [b,c, h, w]
            x2: [b,c, 2*h,2*w]

        Returns: 2x upsampled double conv reselt
        """
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)



def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

def init_random_seed(seed, gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if gpu:
        torch.backends.cudnn.deterministic = True

# from https://huggingface.co/transformers/_modules/transformers/modeling_utils.html
def get_module_device(parameter: nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device

class ViT(nn.Module):
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=256,
                 blocks=8,
                 heads=12,
                 dim_linear_block=2048,
                 dim_head=None,
                 dropout=0, transformer=None, classification=True):
        """
        Minimal re-implementation of ViT
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token that we will index in the final classification layer
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
        self.p = patch_dim
        self.classification = classification
        # tokens = number of patches
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(self.dim / heads)) if dim_head is None else dim_head

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, self.dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, self.dim))

        if self.classification:
            self.mlp_head = nn.Linear(self.dim, num_classes)

        if transformer is None:
            self.transformer = TransformerEncoder(self.dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def forward(self, img, mask=None):
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)
        num_patches = (img.shape[2] // self.p) * (img.shape[3] // self.p)

        batch_size, tokens, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(y[:, 0, :]) if self.classification else y[:, 1:, :]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class TransUnet(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_blocks=8,
                 vit_heads=12,
                 vit_dim_linear_mhsa_block=2048,
                 patch_size=16,
                 vit_transformer_dim=256,
                 vit_transformer=None,
                 vit_channels=None,
                 ):
        """
        My reimplementation of TransUnet based on the paper:
        https://arxiv.org/abs/2102.04306
        Badly written, many details missing and significantly differently
        from the authors official implementation (super messy code also :P ).
        My implementation doesnt match 100 the authors code.
        Basically I wanted to see the logic with vit and resnet backbone for
        shaping a unet model with long skip connections.

        Args:
            img_dim: the img dimension
            in_channels: channels of the input
            classes: desired segmentation classes
            vit_blocks: MHSA blocks of ViT
            vit_heads: number of MHSA heads
            vit_dim_linear_mhsa_block: MHSA MLP dimension
            vit_transformer: pass your own version of vit
            vit_channels: the channels of your pretrained vit. default is 128*8
            patch_dim: for image patches of the vit
        """
        super().__init__()
        self.inplanes = 128
        self.patch_size = patch_size
        self.vit_transformer_dim = vit_transformer_dim
        vit_channels = self.inplanes * 8 if vit_channels is None else vit_channels
        in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                             bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.conv3 = Bottleneck(self.inplanes * 4, vit_channels, stride=2)

        self.img_dim_vit = img_dim // 16

        assert (self.img_dim_vit % patch_size == 0), "Vit patch_dim not divisible"

        self.vit = ViT(img_dim=self.img_dim_vit,
                       in_channels=vit_channels,  # input features' channels (encoder)
                       patch_dim=patch_size,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        # to project patches back - undoes vit's patchification
        token_dim = vit_channels * (patch_size ** 2)
        self.project_patches_back = nn.Linear(vit_transformer_dim, token_dim)
        # upsampling path
        self.vit_conv = SignleConv(in_ch=vit_channels, out_ch=512)
        self.dec1 = Up(vit_channels, 256)
        self.dec2 = Up(512, 128)
        self.dec3 = Up(256, 64)
        self.dec4 = Up(64, 16)

        self.dec1_rec = Up(vit_channels, 256)
        self.dec2_rec = Up(512, 128)
        self.dec3_rec = Up(256, 64)
        self.dec4_rec = Up(64, 16)
        self.segmentation_output = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.reconstruction_output = nn.Conv2d(16, 3, kernel_size=1, padding=0)

    def forward(self, x):
        #print("Input Image shape: ", x.shape)
        # ResNet 50-like encoder
        x2 = self.init_conv(x)
        x4 = self.conv1(x2)
        x8 = self.conv2(x4)
        x16 = self.conv3(x8)  # out shape of 1024, img_dim_vit, img_dim_vit
        y = self.vit(x16)  
        y = self.project_patches_back(y)

        y = rearrange(y, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                      x=self.img_dim_vit // self.patch_size, y=self.img_dim_vit // self.patch_size,
                      patch_x=self.patch_size, patch_y=self.patch_size)
        y = self.vit_conv(y)
        y_seg = self.dec1(y, x8)
        y_seg = self.dec2(y_seg, x4)
        y_seg = self.dec3(y_seg, x2)
        y_seg_ = self.dec4(y_seg)
        y_seg = self.segmentation_output(y_seg_)

        y_rec = self.dec1_rec(y, x8)
        y_rec = self.dec2_rec(y_rec, x4)
        y_rec = self.dec3_rec(y_rec, x2)
        y_rec = self.dec4_rec(y_rec)
        y_rec = self.reconstruction_output(y_rec)

        return x4 , y, y_seg_, y_seg, y_rec

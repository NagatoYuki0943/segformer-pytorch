# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)    # [B, C, H, W] -> [B, C, P] -> [B, P, C]
        x = self.proj(x)                    # [B, P, C] -> [B, P, embed_dim]
        return x

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

#-----------------------------------------------------------------------#
#   SegFormer解码头,对backbone的每个输出使用一个mlp调整到相同的维度,
#   然后经过一个1x1Conv降低维度,在通过一个1x1Conv将通道数调整为num_classes
#-----------------------------------------------------------------------#
class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # MixVisionTransformer的四个输出都接一个mlp调整到指定维度
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # 上面4个输入在通道拼接后经过1x1Conv降低维度
        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        # 将拼接后经过1x1Conv的输出再经过1一个1x1Conv将通道数转换为需要的num_classes
        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        # [B, 32,128, 128]
        # [B, 64,  64, 64]
        # [B, 160, 32, 32]
        # [B, 256, 16, 16]
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        # [B, 256, 16, 16] -> [B, 16*16, 256] -> [B, 256, 16*16] -> [B, 256, 16, 16] -> [B, 256, 128, 128]
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # [B, 160, 32, 32] -> [B, 32*32, 256] -> [B, 256, 32*32] -> [B, 256, 32, 32] -> [B, 256, 128, 128]
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # [B, 64,  64, 64] -> [B, 64*64, 256] -> [B, 256, 64*64] -> [B, 256, 64, 64] -> [B, 256, 128, 128]
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # [B, 32,128, 128] -> [B, 128*128, 256] -> [B, 256, 128*128] -> [B, 256, 128, 128]
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # cat([B, 256, 128, 128] * 4) -> [B, 1024, 128, 128] -> [B, 256, 128, 128]
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x) # [B, 256, 128, 128] -> [B, num_classes, 128, 128]

        return x

class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)   # [B, 3, 512, 512]

        # x = [[B, 32,128, 128],
        #      [B, 64,  64, 64],
        #      [B, 160, 32, 32],
        #      [B, 256, 16, 16]]
        x = self.backbone(inputs)
        # x -> [B, num_classes, 128, 128]
        x = self.decode_head(x)

        # 还原到原图大小 [B, num_classes, 128, 128] -> [B, num_classes, 512, 512]
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == "__main__":
    x = torch.ones(1, 3, 512, 512)
    model = SegFormer()
    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 21, 512, 512]

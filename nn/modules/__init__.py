# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""
from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, ScConv, GSConv, VoVGSCSP, SE, PDC2f, LSKblock, PDConv, PDBottleneck,
                    PDM, VoVGS, GAMAttention,
                    SimAM, space_to_depth, CoordAttention, SPDConv, AKC2f, C2f_faster, Coo_C2f, CooVoVGSCSP, SPPFCSPC)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, BiFPN_Concat2, BiFPN_Concat3)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .CPSAM import CPSAM, MSCAM, PYSAM
from .AKConv import AKConv
from .MixConv import *
from .ShuffleNetv2 import *
from .GSAttention import GSA

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'ScConv',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'BiFPN_Concat2', 'BiFPN_Concat3', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI', 'SimAM', 'GAMAttention',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'GSA', 'VoVGS',
           'GSConv', 'VoVGSCSP', 'space_to_depth', 'CoordAttention', 'PYSAM', 'SE', 'SPDConv', 'PDC2f', 'PDBottleneck',
           'AKC2f', 'AKConv', 'C2f_faster', 'Coo_C2f', 'MixBlock', 'CooVoVGSCSP', 'SPPFCSPC', 'LSKblock', 'PDConv', 'PDM')

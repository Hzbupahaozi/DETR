# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# 初始化父类
class BackboneBase(nn.Module):

    #                  resnet50             True                  2048               false
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            # 如果不微调 backbone 的话（train_backbone=False），就把所有参数的require_grad 都设置为false
            # 如果微调的话，只微调'layer2''layer3''layer4'层里处了bn层以外的的参数
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            # 只返回最后一层的输出，字典
            return_layers = {'layer4': "0"}
        # 获取网络中间层的输出，对于目标检测任务这里获取layer4层的输出
        # 给self.body一个输入，就会返回layer4这一层的输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    # 输入的tensor_list就是dataloader输出的部分，其中tensor_list.tensors就是图片的信息，一个batch里的两张图片的数据
    def forward(self, tensor_list: NestedTensor):
        # 输出的xs就是一个字典，只有一个键值对，key:'0',value:layer4的输出 （2，2048，27，27）
        xs = self.body(tensor_list.tensors)
        # 空字典把xs的键值对拿出来
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # mask是指明图像中哪一些是padding（True）哪一些是图像（False）
            # 把空图取出来，然后通过双线性插值的方式，缩放为和
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,   # resnet50
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):   #
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],  # 最经典的resnet50
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)  # 初始化父类


# 这里是传给了nn.Sequential，可以通过索引来获取模块
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # self[0]就是backbone，然后把tensor_list传入到backbone中
        # 这里获得的xs其实就是经过了layer4之后的输出
        xs = self[0](tensor_list)
        # 一个放backbone输出，一个放位置编码
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    # position_encoding是在backbone之后才开始使用
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0  # 是否要微调backbone的权重
    return_interm_layers = args.masks       # false，不用返回backbone中间层的输出
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 用Joiner把backbone和position_embedding放在一起，因为在做分割的时候需要backbone中layer1，2，3的输出，也就需要对应大小的位置编码
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

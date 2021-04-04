import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from .nf_base import WSConv2d, ScaledStdConv2d

NF_RESO_CONFIG = {
    'nf_0': {'train_reso':192, 'inference_reso':256},
    'nf_1': {'train_reso':224, 'inference_reso':320},
    'nf_2': {'train_reso':256, 'inference_reso':352},
    'nf_3': {'train_reso':320, 'inference_reso':416},
    'nf_4': {'train_reso':384, 'inference_reso':512},
    'nf_5': {'train_reso':416, 'inference_reso':544},
    'nf_6': {'train_reso':448, 'inference_reso':576}
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return base_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Conv2d:
    """1x1 convolution"""
    return base_conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            base_conv: nn.Conv2d = ScaledStdConv2d
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, base_conv=base_conv)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, base_conv=base_conv)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class NFResNet(nn.Module):

    def __init__(
            self,
            block: Type[BasicBlock],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            base_conv: nn.Conv2d = ScaledStdConv2d,
            dropout_p: float = 0.2
    ) -> None:
        super(NFResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = base_conv(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], base_conv=base_conv)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], base_conv=base_conv)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], base_conv=base_conv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], base_conv=base_conv)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout_p)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                        stride, base_conv=base_conv),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, base_conv=base_conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                base_conv=base_conv))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _nf_resnet(
        arch: str,
        block: Type[BasicBlock],
        layers: List[int],
        pretrained: bool,
        base_conv: nn.Conv2d,
        dropout_p: float,
        **kwargs: Any
) -> NFResNet:
    model = NFResNet(block, layers, base_conv=base_conv, dropout_p=dropout_p, **kwargs)
    return model

def nf_light_v1_0(base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs):
    return _nf_resnet('f0', BasicBlock, [1, 2, 6, 3], False, base_conv=base_conv, dropout_p=0.2, **kwargs)

def nf_light_v1_1(base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs):
    return _nf_resnet('f1', BasicBlock, [2, 4, 12, 6], False, base_conv=base_conv, dropout_p=0.3, **kwargs)

def nf_light_v1_2(base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs):
    return _nf_resnet('f1', BasicBlock, [3, 6, 18, 9], False, base_conv=base_conv, dropout_p=0.4, **kwargs)

def nf_light_v1_3(base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs):
    return _nf_resnet('f1', BasicBlock, [4, 8, 24, 12], False, base_conv=base_conv, dropout_p=0.5, **kwargs)

def nf_light_v1_4(base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs):
    return _nf_resnet('f1', BasicBlock, [5, 10, 30, 15], False, base_conv=base_conv, dropout_p=0.5, **kwargs)

def nf_light_v1_5(base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs):
    return _nf_resnet('f1', BasicBlock, [6, 12, 36, 18], False, base_conv=base_conv, dropout_p=0.5, **kwargs)

def nf_light_v1_6(base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs):
    return _nf_resnet('f1', BasicBlock, [7, 14, 42, 21], False, base_conv=base_conv, dropout_p=0.5, **kwargs)
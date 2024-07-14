import torch
import torch.nn as nn
from typing import Dict, Type, List, Optional
from functools import partial
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass

from .layers3d import (
    DoubleConv_3d,
    Down_3d,
    Up_3d,
    OutConv_3d,
    Bottleneck,
    downsample_basic_block,
)
from .losses3d import get_loss_criterion

import debugpy

# try:
#     debugpy.listen(("localhost", 1925))
#     print("Waiting for debugger attach")
#     print("the python code is unet3d.py")
#     print("the host is: localhost, the port is: 1925")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


class UNet_3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        bilinear: bool = True,
        base_c: int = 64,
    ):
        super(UNet_3d, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv_3d(in_channels, base_c)
        self.down1 = Down_3d(base_c, base_c * 2)
        self.down2 = Down_3d(base_c * 2, base_c * 4)
        self.down3 = Down_3d(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down_3d(base_c * 8, base_c * 16 // factor)
        self.up1 = Up_3d(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up_3d(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up_3d(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up_3d(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv_3d(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits


class UNet_3d_resnet_encoder(nn.Module):
    """
    3D U-Net网络结合了ResNet编码器，用于图像分割任务。

    参数:
    - block: 残差块类型，用于构建ResNet编码器。
    - layers: 每个残差块层的块数量列表。
    - in_channels: 输入通道数，默认为1。
    - num_classes: 分割任务的类别数，默认为2。
    - bilinear: 是否使用双线性上采样，默认为True。
    - base_c: 基础通道数，默认为64。
    - shortcut_type: 残差连接的类型，"A"或"B"，默认为"B"。
    - no_cuda: 是否禁用CUDA，默认为False。
    """

    def __init__(
        self,
        block: Optional[Bottleneck],
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 2,
        bilinear: bool = True,
        base_c: int = 64,
        shortcut_type: str = "B",
        no_cuda: bool = False,
    ) -> None:
        super(UNet_3d_resnet_encoder, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inplanes = 64
        self.no_cuda = no_cuda

        # unet
        factor = 2 if bilinear else 1
        self.down4 = Down_3d(base_c * 8, base_c * 16 // factor)

        self.up1 = Up_3d(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up_3d(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up_3d(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up_3d(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv_3d(base_c, num_classes)

        # resnet
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(1, 1, 1),  # stride=(2, 2, 2)-> (1, 1, 1)
            padding=(3, 3, 3),
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block: Type[Bottleneck],  # 残差块的类型
        planes: int,  # 每个残差块的输出通道数
        blocks: int,  # 该层中残差块的数量
        shortcut_type: str,  # 快捷连接的类型，"A" 或其他值
        stride: int = 1,  # 卷积层的步长
    ) -> nn.Sequential:
        """
        创建并返回一个由多个残差块组成的序列层。

        参数:
        - block: 残差块的类类型。
        - planes: 每个残差块的输出通道数。
        - blocks: 该层中残差块的数量。
        - shortcut_type: 快捷连接的类型，"A" 或其他值，影响下采样层的构造。
        - stride: 卷积层的步长，默认为1。

        返回:
        - 一个由多个残差块组成的nn.Sequential层。
        """
        downsample = None
        # 如果步长不为1或当前平面数不等于扩展后的平面数，则需要下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                # 使用函数partial定义下采样快捷连接，适用于Shortcut-A
                downsample = partial(
                    downsample_basic_block,  # 假设这是定义好的下采样函数
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,  # 假设self.no_cuda是类的一个属性
                )
            else:
                # 定义下采样层，通常用于Shortcut-B
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []  # 用于存储每一层残差块
        # 添加第一个残差块，它可能包含下采样层
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion  # 更新当前平面数
        # 添加剩余的残差块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)  # 返回一个序列层

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)

        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits


class UNet3dConfig(PretrainedConfig):
    def __init__(
        self,
        in_channels=1,
        num_channels=1,
        base_c=64,
        layers=[3, 4, 6, 3],
        unet_type="UNet_3d",
        loss_config: dict = {},
        **kwargs,
    ):
        """ "
        unet_type: UNet_3d or UNet_3d_resnet_encoder
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.unet_type = unet_type
        self.base_c = base_c
        self.layers = layers

        self.loss_config = loss_config
     
        self.shortcut_type = "B"    # 保证总是使用Shortcut-B实现残差连接，也就是用卷积网络实现下采样
        self.no_cuda = False    # 保证总是使用cuda加速
        self.bilinear = False   # 保证总是使用 反卷积 来上采样，而不是使用 双线性 上采样

        self.label_names = "labels"
        self.main_input_name = "volumes"
        self.keys_to_ignore_at_inference = ["labels"]


@dataclass
class UNet3DModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    labels: torch.LongTensor = None


class UNet3DModel(PreTrainedModel):
    def __init__(self, config: UNet3dConfig):
        super().__init__(config)

        if config.unet_type == "UNet_3d":
            print("use the UNet_3d")
            self.unet = UNet_3d(
                in_channels=config.in_channels,
                num_classes=config.num_channels,
                bilinear=config.bilinear,
                base_c=config.base_c,
            )
        elif config.unet_type == "UNet_3d_resnet_encoder":
            print("use the UNet_3d_resnet_encoder")
            self.unet = UNet_3d_resnet_encoder(
                block=Bottleneck,   # 使用确定的残差块类型
                layers=config.layers,
                in_channels=config.in_channels,
                num_classes=config.num_channels,
                bilinear=config.bilinear,
                base_c=config.base_c,
                shortcut_type=config.shortcut_type,
                no_cuda=config.no_cuda,
            )
        else:
            raise ValueError("unet_type must be UNet_3d or UNet_3d_resnet_encoder")
        self.criterion = get_loss_criterion(config.loss_config)

    def forward(
        self, volumes: torch.Tensor, labels: torch.Tensor = None
    ) -> UNet3DModelOutput:
        unet3d_output = self.unet(volumes)
        if labels is not None:
            loss = self.criterion(unet3d_output, labels)
        return UNet3DModelOutput(loss=loss, logits=unet3d_output, labels=labels)


if __name__ == "__main__":
    loss_config = {
        "loss": {
            "name": "BCEWithLogitsLoss",
        }
    }
    model_config = UNet3dConfig(
        in_channels=1,
        num_channels=1,
        base_c=64,
        layers=[3, 4, 6, 3],
        unet_type="UNet_3d",
        loss_config=loss_config,
    )
    model = UNet3DModel(model_config)
    x = torch.randn(4, 1, 64, 64, 64)
    y = torch.randint(0, 2, (4, 1, 64, 64, 64)).float()
    output = model(**{"volumes": x, "labels": y})
    # output = model(x, y)
    print(output)

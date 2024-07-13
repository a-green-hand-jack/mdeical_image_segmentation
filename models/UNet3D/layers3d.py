import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional


class DoubleConv_3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_3d, self).__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down_3d(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down_3d, self).__init__(
            nn.MaxPool3d(2, stride=2), DoubleConv_3d(in_channels, out_channels)
        )


class Up_3d(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_3d, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv_3d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv_3d(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W, Z]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        diff_z = x2.size()[4] - x1.size()[4]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_z // 2,
                diff_z - diff_z // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv_3d(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv_3d, self).__init__(
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(
    x: torch.Tensor, planes: int, stride: int, no_cuda: Optional[bool] = False
) -> torch.Tensor:
    """
    对输入的3D张量进行下采样，并根据需要在通道维度上进行零填充。

    参数:
    - x: 输入的3D特征张量。
    - planes: 期望的输出通道数。
    - stride: 下采样的步长。
    - no_cuda: 是否禁用CUDA，如果设置为True，则不会将零填充张量移动到CUDA设备。

    返回:
    - 下采样并可能经过零填充的张量。
    """
    # 使用平均池化进行下采样，kernel_size=1意味着只进行步长为stride的下采样而不改变特征图大小
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)

    # 计算需要添加的零填充大小
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    )

    # 如果没有禁用CUDA并且输入张量在CUDA设备上，则将零填充张量也移动到CUDA设备
    if not no_cuda and isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    # 沿着通道维度拼接原始输出和零填充张量
    out = torch.cat([out.data, zero_pads], dim=1)

    return out


class Bottleneck(nn.Module):
    """
    定义了一个残差块，用于3D残差网络的构建。

    参数:
    - inplanes: 输入通道数。
    - planes: 卷积层的通道数。
    - stride: 卷积的步长，默认为1。
    - dilation: 卷积的扩张率，默认为1。扩张率用于控制卷积核覆盖的输入特征图的范围。扩张率为1时，卷积核覆盖的输入与常规卷积相同。扩张率大于1时，卷积核覆盖的输入范围会增大，但不会改变特征图的尺寸。
    - downsample: 用于调整输入维度以匹配输出维度的可选下采样层。
    """

    expansion = 2  # 特征图扩展倍数

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        # 第一个卷积层，1x1卷积用于降维
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        # 第二个卷积层，3x3x3卷积用于特征提取；但是这个部分不会改变特征图的大小，也就是说，没有使用下采样
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        # 第三个卷积层，1x1卷积用于升维
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
        - x: 输入的张量。

        返回:
        - 经过残差块处理后的张量。
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果存在下采样层，则对输入进行下采样以匹配维度
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接
        out += residual
        out = self.relu(out)

        return out

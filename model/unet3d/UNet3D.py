import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

from model import UNet3D, ResidualUNet3D, ResidualUNetSE3D
from model.unet3d.losses import get_loss_criterion
logger = logging.getLogger(__name__)




class UNet3DForMedicalSegmentationConfig(PretrainedConfig):
    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid,
        basic_module,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
        loss_config:dict = {"loss": {"name":"BCEDiceLoss"}},
        unet_type="UNet3D",  # UNet3D, ResidualUNet3D, ResidualUNetSE3D
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.basic_module = basic_module
        self.f_maps = f_maps
        self.layer_order = layer_order
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.is_segmentation = is_segmentation
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.conv_padding = conv_padding
        self.conv_upscale = conv_upscale
        self.upsample = upsample
        self.dropout_prob = dropout_prob
        self.is3d = is3d
        self.unet_type = unet_type
        self.loss_config = loss_config


@dataclass
class UNet3DForMedicalSegmentationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None


class UNet3DForMedicalSegmentation(PreTrainedModel):
    def __init__(self, config: UNet3DForMedicalSegmentationConfig):
        super().__init__(config)
        if config.unet_type == "UNet3D":
            self.model = UNet3D(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                final_sigmoid=config.final_sigmoid,
                f_maps=config.f_maps,
                layer_order=config.layer_order,
                num_groups=config.num_groups,
                num_levels=config.num_levels,
                is_segmentation=config.is_segmentation,
                conv_padding=config.conv_padding,
                conv_upscale=config.conv_upscale,
                upsample=config.upsample,
                dropout_prob=config.dropout_prob,
            )

        elif config.unet_type == "ResidualUNet3D":
            self.model = ResidualUNet3D(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                final_sigmoid=config.final_sigmoid,
                f_maps=config.f_maps,
                layer_order=config.layer_order,
                num_groups=config.num_groups,
                num_levels=config.num_levels,
                is_segmentation=config.is_segmentation,
                conv_padding=config.conv_padding,
                conv_upscale=config.conv_upscale,
                upsample=config.upsample,
                dropout_prob=config.dropout_prob,
            )
        elif config.unet_type == "ResidualUNetSE3D":
            self.model = ResidualUNetSE3D(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                final_sigmoid=config.final_sigmoid,
                f_maps=config.f_maps,
                layer_order=config.layer_order,
                num_groups=config.num_groups,
                num_levels=config.num_levels,
                is_segmentation=config.is_segmentation,
                conv_padding=config.conv_padding,
                conv_upscale=config.conv_upscale,
                upsample=config.upsample,
                dropout_prob=config.dropout_prob,
            )
        self.chose_activation(config)
        self.loss_criterion = get_loss_criterion(config.loss_config)
        
    

    def chose_activation(self, config):
        if config.is_segmentation and config.final_sigmoid:
            logger.info("Using sigmoid activation for segmentation task")
            self.activation = nn.Sigmoid()
        elif config.is_segmentation and not config.final_sigmoid:
            logger.info("Using softmax activation for segmentation task")
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None
            logger.info("Using no activation for regression task")
        
    def forward(
        self,
        volume: Optional[torch.Tensor],
        target: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor | None] = None,
    ):
        unet3d_output = self.model(volume)  # 这里没有使用激活函数
        activated_output = (
            self.activation(unet3d_output)
            if self.activation is not None
            else unet3d_output
        )

        if weight is None:
            loss = self.loss_criterion(activated_output, target)
        else:
            loss = self.loss_criterion(activated_output, target, weight)

        return UNet3DForMedicalSegmentationOutput(
            loss=loss, logits=activated_output, labels=target
        )

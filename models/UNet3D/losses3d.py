import torch
import torch.nn.functional as F
from torch import nn as nn
from monai.losses import DiceCELoss

from typing import Optional,Any,Dict

def compute_per_channel_dice(
    input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
    weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算多通道输入和目标张量之间的Dice系数。
    https://arxiv.org/abs/1606.04797
    参数:
    - input (torch.Tensor): NxCxSpatial 输入张量，其中N是批次大小，C是通道数，Spatial是空间维度。
    - target (torch.Tensor): NxCxSpatial 目标张量，与输入具有相同的形状。
    - epsilon (float): 用于防止除以零的小常数。
    - weight (torch.Tensor, 可选): 每个通道/类别的权重，形状为Cx1。
    
    返回:
    - torch.Tensor: 每个通道的Dice系数。
    """

    # 输入和目标的形状必须匹配
    assert input.size() == target.size(), "输入和目标必须具有相同的形状"

    # 展平除了通道维度以外的所有维度
    input = input.view(input.size(0), input.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    target = target.float()  # 确保target是浮点类型

    # 计算每个通道的交集
    intersect = (input * target).sum(-1)
    if weight is not None:
        # 如果提供了权重，则按权重计算加权交集
        intersect = weight * intersect

    # 计算分子和分母
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    # 计算Dice系数，使用clamp确保分母不为零
    dice_coefficient = 2 * (intersect / denominator.clamp(min=epsilon))

    return dice_coefficient


class _MaskingLossWrapper(nn.Module):
    """
    损失包装器，用于防止在目标等于 `ignore_index` 的位置处计算损失梯度。
    
    Attributes:
        loss (nn.Module): 被包装的原始损失函数。
        ignore_index (int): 用于忽略梯度计算的目标值索引。
    """

    def __init__(self, loss: nn.Module, ignore_index: int):
        """
        初始化损失包装器。
        
        参数:
            loss (nn.Module): 被包装的损失函数。
            ignore_index (int): 目标张量中用于忽略梯度计算的索引值。
        """
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index 不能为 None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播，计算损失，并应用掩码以忽略指定索引处的梯度。
        
        参数:
            input (torch.Tensor): 输入张量，通常是模型的预测结果。
            target (torch.Tensor): 目标张量，通常是真实标签。
        
        返回:
            torch.Tensor: 计算得到的损失值。
        """
        # 创建掩码，`ignore_index` 对应位置设为False
        mask = target.clone().ne_(self.ignore_index)
        
        # 确保掩码不需要梯度
        mask.requires_grad = False

        # 应用掩码到输入和目标，忽略掩码为False的位置
        input = input * mask
        target = target * mask

        # 将掩码后的输入和目标传递给原始损失函数计算损失
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    损失包装器，用于在计算损失前去除目标张量的额外通道。
    
    Attributes:
        loss (nn.Module): 被包装的原始损失函数。
        squeeze_channel (bool): 是否在去除最后一个通道后压缩目标张量的通道维度。
    """

    def __init__(self, loss: nn.Module, squeeze_channel: bool = False):
        """
        初始化损失包装器。
        
        参数:
            loss (nn.Module): 被包装的损失函数。
            squeeze_channel (bool): 是否在去除最后一个通道后压缩目标张量的通道维度。
        """
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行前向传播，去除目标张量的最后一个通道，并计算损失。
        
        参数:
            input (torch.Tensor): 输入张量，通常是模型的预测结果。
            target (torch.Tensor): 目标张量，通常是真实标签。
            weight (torch.Tensor, 可选): 用于加权损失计算的权重张量。
        
        返回:
            torch.Tensor: 计算得到的损失值。
        """
        # 确保目标张量的通道维度不是单一的，这样才能去除最后一个通道
        assert target.size(1) > 1, "目标张量的通道维度是单一的，无法去除通道"

        # 如果需要，去除最后一个目标通道
        target = target[:, :-1, ...]

        # 如果设置了squeeze_channel，压缩目标张量的通道维度
        if self.squeeze_channel:
            target = torch.squeeze(target, dim=1)

        # 计算损失，根据weight参数是否存在，调用损失函数的方式不同
        if weight is not None:
            return self.loss(input, target, weight)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    不同实现的Dice损失的基类。
    
    Attributes:
        weight (torch.Tensor, 可选): 用于加权不同通道或类别的权重张量。
        normalization (str): 指定用于归一化输出的函数，可以是 'sigmoid'、'softmax' 或 'none'。
    """
    
    def __init__(self, weight: Optional[torch.Tensor] = None, normalization: str = 'sigmoid'):
        """
        初始化抽象Dice损失基类。
        
        参数:
            weight (torch.Tensor, 可选): 用于加权不同通道或类别的权重张量。
            normalization (str): 指定用于归一化输出的函数，可以是 'sigmoid'、'softmax' 或 'none'。
        """
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        
        # 根据normalization参数选择归一化函数
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:  # normalization == 'none'
            self.normalization = lambda x: x

    def dice(self, input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]):
        """
        实际的Dice分数计算方法，需要在子类中实现。
        
        参数:
            input (torch.Tensor): 模型的原始输出（未归一化的预测）。
            target (torch.Tensor): 目标张量（通常是真实的标签或掩码）。
            weight (torch.Tensor, 可选): 用于加权计算Dice分数的权重张量。
        
        返回:
            float: 计算得到的Dice分数。
        """
        raise NotImplementedError("Dice score computation must be implemented by the subclass")

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        执行前向传播，计算Dice损失。
        
        参数:
            input (torch.Tensor): 模型的原始输出（未归一化的预测）。
            target (torch.Tensor): 目标张量（通常是真实的标签或掩码）。
        
        返回:
            torch.Tensor: 计算得到的Dice损失值。
        """
        # 使用指定的归一化函数处理输入
        input = self.normalization(input)

        # 计算每个通道的Dice系数
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # 在所有通道/类别上平均Dice分数
        return 1. - torch.mean(per_channel_dice)



class DiceLoss(_AbstractDiceLoss):
    """
    计算Dice损失，用于衡量预测分割和真实分割之间的相似度。
    https://arxiv.org/abs/1606.04797.
    
    参数:
        weight (torch.Tensor, 可选): 一个张量，用于在多分类问题中为每个类别分配不同的权重。
        normalization (str): 指定用于归一化输出的函数，可以是 'sigmoid' 或 'softmax'。
    """
    
    def __init__(self, weight: Optional[torch.Tensor] = None, normalization: str = 'sigmoid'):
        """
        初始化Dice损失实例。
        
        参数:
            weight (torch.Tensor, 可选): 一个张量，用于在多分类问题中为每个类别赋予不同权重。
            normalization (str): 指定用于归一化输出的函数，可以是 'sigmoid' 或 'softmax'。
        """
        super().__init__(weight, normalization)

    def dice(self, input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算Dice系数。
        
        参数:
            input (torch.Tensor): 输入张量，通常是模型的原始输出（未经归一化的预测）。
            target (torch.Tensor): 目标张量，通常是真实的标签或掩码。
            weight (torch.Tensor, 可选): 用于加权Dice系数的权重张量。
        
        返回:
            torch.Tensor: 计算得到的Dice系数。
        """
        # 调用 compute_per_channel_dice 函数来计算每个通道的Dice系数
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """
    计算广义Dice损失（Generalized Dice Loss，GDL），用于处理高度不平衡的数据集，如医学图像分割问题。
    
    Attributes:
        epsilon (float): 用于数值稳定性的小常数，防止除以零。
    """

    def __init__(self, normalization: str = 'sigmoid', epsilon: float = 1e-6):
        """
        初始化GeneralizedDiceLoss实例。
        
        参数:
            normalization (str): 指定用于归一化输出的函数，可以是 'sigmoid' 或 'softmax'。
            epsilon (float): 用于数值稳定性的小常数。
        """
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算广义Dice系数。
        
        参数:
            input (torch.Tensor): 输入张量，模型的原始输出（未归一化的预测）。
            target (torch.Tensor): 目标张量，通常是真实的标签或掩码。
            weight (torch.Tensor, 可选): 用于加权计算Dice分数的权重张量。
        
        返回:
            torch.Tensor: 计算得到的广义Dice系数。
        """
        assert input.size() == target.size(), "输入和目标必须具有相同的形状"

        input = input.view(-1)  # 展平输入张量
        target = target.view(-1).float()  # 展平目标张量并转换为浮点数

        # 当只有一个通道时，将前景和背景像素分到不同的通道
        if input.size(0) == 1:
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # 计算权重，每个标签的贡献通过其体积的倒数来校正
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        # 计算交并比的交集部分
        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        # 计算交并比的分母部分
        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        # 计算并返回广义Dice系数
        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = nominator / denominator
        return class_weights.detach()


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target[mask] = 0
        else:
            mask = torch.zeros_like(target)
        # add channel dimension and invert the mask
        mask = 1 - mask.unsqueeze(1)
        # convert target to one-hot encoding
        target = F.one_hot(target.long())
        if target.ndim == 5:
            # permute target to (NxCxDxHxW)
            target = target.permute(0, 4, 1, 2, 3).contiguous()
        else:
            target = target.permute(0, 3, 1, 2).contiguous()
        # apply the mask on the target
        target = target * mask
        # add channel dimension to the weights
        weights = weights.unsqueeze(1)
        # compute the losses
        result = -weights * target * log_probabilities
        return result.mean()


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    # assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        weight = torch.tensor(weight)

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss


#######################################################################################################################


def _create_loss(
    name: str,
    loss_config: Dict[str, Any],
    weight: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
    pos_weight: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    根据提供的名称和配置创建并返回一个损失函数实例。
    
    参数:
        name (str): 损失函数的名称。
        loss_config (Dict[str, Any]): 损失函数的配置参数字典。
        weight (torch.Tensor, 可选): 用于加权损失计算的权重张量。
        ignore_index (int, 可选): 在计算损失时忽略的索引。
        pos_weight (torch.Tensor, 可选): 正样本的权重张量。
    
    返回:
        nn.Module: PyTorch损失函数实例。
    
    抛出:
        RuntimeError: 如果提供的损失函数名称不受支持。
    """
    
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alpha', 1.)
        beta = loss_config.get('beta', 1.)
        # 假设 BCEDiceLoss 是一个已经定义好的类
        return BCEDiceLoss(alpha, beta)
    elif name == 'CrossEntropyLoss':
        ignore_index = -100 if ignore_index is None else ignore_index
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        ignore_index = -100 if ignore_index is None else ignore_index
        # 假设 WeightedCrossEntropyLoss 是一个已经定义好的类
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        # 假设 PixelWiseCrossEntropyLoss 是一个已经定义好的类
        return PixelWiseCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return GeneralizedDiceLoss(normalization=normalization)
    elif name == 'DiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return DiceLoss(weight=weight, normalization=normalization)
    elif name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'SmoothL1Loss':
        return nn.SmoothL1Loss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(
            threshold=loss_config['threshold'],
            initial_weight=loss_config['initial_weight'],
            apply_below_threshold=loss_config.get('apply_below_threshold', True)
        )
    elif name == 'DiceCELoss':  # 参考https://github.com/a-green-hand-jack/SAM-Med3D/blob/main/train.py 中的loss_fn的选择
        return DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")



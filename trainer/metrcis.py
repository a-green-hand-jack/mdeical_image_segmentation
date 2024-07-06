import numpy as np
from transformers import EvalPrediction

import numpy as np
import matplotlib.pyplot as plt
from medpy import metric
import torch


def plot_first_element(array1, array2, threshold=0.5):
    """
    绘制两个形状为 (N, H, W) 的 NumPy 数组的第一个元素的灰阶图。

    :param array1: 第一个 NumPy 数组
    :param array2: 第二个 NumPy 数组
    # 示例数据
    array1 = np.random.rand(10, 512, 512)
    array2 = np.random.rand(10, 512, 512)

    # 调用函数绘制灰阶图
    plot_first_element(array1, array2)

    """
    # 确保输入是 NumPy 数组
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)

    # 检查输入数组的形状
    assert array1.ndim == 3, "array1 必须是一个 3D 数组"
    assert array2.ndim == 3, "array2 必须是一个 3D 数组"
    assert array1.shape == array2.shape, "两个数组的形状必须相同"

    array1 = (array1 > threshold).astype(np.float32)
    array2 = (array2 > threshold).astype(np.float32)

    # 提取第一个元素
    first_element_array1 = array1[0]
    first_element_array2 = array2[0]

    # 创建一个图形窗口
    plt.figure(figsize=(10, 5))

    # 绘制第一个数组的第一个元素的灰阶图
    plt.subplot(1, 2, 1)
    plt.title("Array 1 - First Element")
    plt.imshow(first_element_array1, cmap="gray")
    plt.axis("off")  # 隐藏坐标轴

    # 绘制第二个数组的第一个元素的灰阶图
    plt.subplot(1, 2, 2)
    plt.title("Array 2 - First Element")
    plt.imshow(first_element_array2, cmap="gray")
    plt.axis("off")  # 隐藏坐标轴
    plt.savefig("output.png")

    # 显示图像
    plt.show()


def compute_iou(preds, labels, threshold=0.5):
    """
    计算 Intersection over Union (IoU)

    :param preds: 预测张量，形状为 (N, H, W)，其中 N 是样本数量，H 是高度，W 是宽度
    :param labels: 标签张量，形状为 (N, H, W)
    :param threshold: 二值化阈值，默认值为 0.5
    :return: IoU
    """
    preds = (preds > threshold).astype(np.float32)
    labels = (labels > threshold).astype(np.float32)

    intersection = np.sum(preds * labels, axis=(1, 2))
    union = np.sum(preds, axis=(1, 2)) + np.sum(labels, axis=(1, 2)) - intersection
    # 避免除以零
    epsilon = 1e-6
    union = np.maximum(union, epsilon)

    iou = intersection / union
    iou_mean = np.mean(iou)  # 计算所有样本的IoU平均值
    return iou_mean


def compute_dice(preds, labels, threshold=0.5):
    """
    计算 Dice coefficient

    :param preds: 预测张量，形状为 (N, H, W)
    :param labels: 标签张量，形状为 (N, H, W)
    :param threshold: 二值化阈值，默认值为 0.5
    :return: Dice coefficient
    """
    preds = (preds > threshold).astype(np.float32)
    labels = (labels > threshold).astype(np.float32)

    intersection = np.sum(preds * labels, axis=(1, 2))
    sum_pred = np.sum(preds, axis=(1, 2))
    sum_lab = np.sum(labels, axis=(1, 2))

    # 避免除以零
    epsilon = 1e-6
    denominator = (2.0 * intersection) + epsilon
    sum_pred += epsilon
    sum_lab += epsilon

    dice = denominator / (sum_pred + sum_lab)

    dice = np.mean(dice)  # 计算所有样本的Dice系数平均值
    return dice


def calculate_metric_percase(pred, gt):
    """
    计算分割任务的评估指标

    :param pred: 预测结果的二值化图像
    :param gt: 真实标注的二值化图像
    :return: 返回四个评估指标：Dice coefficient, Jaccard coefficient, HD95, ASD
    """
    dice = metric.binary.dc(pred, gt)  # 计算 Dice coefficient
    jc = metric.binary.jc(pred, gt)  # 计算 Jaccard coefficient
    hd = metric.binary.hd95(pred, gt)  # 计算 95th percentile Hausdorff distance
    asd = metric.binary.asd(pred, gt)  # 计算 Average Surface Distance
    return dice, jc, hd, asd


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)


def compute_metrics(p: EvalPrediction):
    logits, labels = p.predictions, p.label_ids
    # 去掉维度为 1 的通道维度, (N,C,H,W) where C=1 -> (N,H,W)
    preds = np.squeeze(logits, axis=1).astype(np.float32)
    labels = np.squeeze(labels, axis=1).astype(np.float32)

    # preds = np.argmax(logits, axis=1)
    epsilon = 1e-6
    preds = 1 / (1 + np.exp(-preds) + epsilon)
    # 计算 preds 的全局平均值作为阈值
    threshold = np.mean(preds)
    plot_first_element(array1=preds, array2=labels, threshold=threshold)

    iou = compute_iou(preds=preds, labels=labels, threshold=threshold)
    dice = compute_dice(preds=preds, labels=labels, threshold=threshold)
    return {"iou": iou, "dice": dice}

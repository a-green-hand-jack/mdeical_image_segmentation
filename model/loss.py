import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from pytorch_msssim import MS_SSIM
import torch.nn.functional as F

"""
    
By combining focal loss ( ℓp: ) [10], MS-SSIM loss
(ℓa;6;;,a) and IoU loss (ℓ,qr) [11], we develop a hybrid loss 
for segmentation in three-level hierarchy – pixel-, patch- and 
map-level, which is able to capture both large-scale and fine
structures with clear boundaries. The hybrid segmentation 
loss (ℓ;#d)is defined as

"""




class MSSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return 1 - self.ms_ssim(inputs, targets)



class IoULoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(IoULoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        iou = (intersection + self.epsilon) / (union + self.epsilon)
        return 1 - iou



class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = (inputs * targets).sum()
        precision = TP / (inputs.sum() + self.epsilon)
        recall = TP / (targets.sum() + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        return 1 - f1

class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.f1_loss = F1Loss()
        self.ms_ssim_loss = MSSSIMLoss()
        self.iou_loss = IoULoss()

    def forward(self, inputs, targets):
        loss_f1 = self.f1_loss(inputs, targets)
        loss_msssim = self.ms_ssim_loss(inputs, targets)
        loss_iou = self.iou_loss(inputs, targets)
        return loss_f1 + loss_msssim + loss_iou

if __name__ == '__main__':
    # 示例用法
    pred = torch.randn((4, 1, 256, 256), requires_grad=True)  # 预测值
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()  # 真实值

    criterion = SegmentationLoss()
    loss = criterion(pred, target)
    print("Segmentation Loss:", loss.item())

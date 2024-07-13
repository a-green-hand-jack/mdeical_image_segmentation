import matplotlib.pyplot as plt
import os
import torch
import numpy as np

from datasets import DRIVEDataset, DRIVEDataCollator
from models import UNet
import debugpy

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9925))
    print("Waiting for debugger attach")
    print("the python code is test_model.py")
    print("the host is: localhost, the port is: 9925")
    debugpy.wait_for_client()
except Exception as e:
    pass


if __name__ == "__main__":
    # 定义增强变换

    # 使用增强方法创建数据集
    data_path = "../Dataset/DRIVE/training"
    dataset = DRIVEDataset(data_path, augmentations=None)
    # 示例用法
    sample_data = dataset[15]
    image = sample_data["image"]
    mask = sample_data["mask"]

    # 创建模型实例
    # model = SampleUNet(in_channels=3, out_channels=2)
    # model = ResUNet(num_classes=1,)
    model = UNet(
        in_channels=3,
        out_channels=1,
    )
    # 打印模型结构
    # print(model)

    model.eval()

    # Perform forward pass
    with torch.no_grad():
        output_tensor = model(
            images=torch.unsqueeze(
                image, 0
            ),  # 添加批处理维度，假设 image 是三维张量 (C, H, W)
            masks=torch.unsqueeze(mask, 0),
        )  # 添加批处理维度，假设 mask 是三维张量 (C, H, W)
    # output_array = torch.squeeze(output_tensor, 0).numpy()  # 去除批处理维度，假设 output_tensor 形状为 (1, 32, 32)

    # Convert output tensor to numpy array and squeeze to remove batch dimension
    # output_array = output_tensor.squeeze(0).squeeze(0).numpy()
    # # Display original image and model output
    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1)
    # plt.imshow(image.numpy().transpose(1, 2, 0))
    # plt.title("Original Image")
    # plt.axis("off")

    # plt.subplot(1, 3, 2)
    # plt.imshow(output_array, cmap="gray")  # Assuming output is grayscale
    # plt.title("UNet Output")
    # plt.axis("off")

    # plt.subplot(1, 3, 3)
    # plt.imshow(mask.squeeze(0).numpy(), cmap="gray")  # Assuming output is grayscale
    # plt.title("Mask")
    # plt.axis("off")
    # # save_path = "./model/ResUNet_visualization.png"
    # save_path = "./model/ResNet34Unet_visualization.png"
    # # plt.show()
    # if save_path:
    #     # Ensure directory exists
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     # Save figure
    #     plt.savefig(save_path)
    #     print(f"Saved visualization to {save_path}")
    # else:
    #     plt.show()

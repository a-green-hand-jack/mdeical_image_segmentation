import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9525))
#     print("Waiting for debugger attach")
#     print("the python code is dataset.py")
#     print("the host is: localhost, the port is: 9525")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def show(dataset, index, save_path=None):
    sample = dataset[index]
    image = sample["image"].numpy().transpose(1, 2, 0)
    mask = sample["mask"].squeeze(0).numpy()
    # Clip image data to [0, 1]
    image = np.clip(image, 0, 1)
    mask = np.clip(mask, 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save figure
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


class DRIVEDataset(Dataset):
    def __init__(
        self,
        data_path,
        augmentations=None,
        mode="train",
        train_ratio=0.7,
        eval_ratio=0.2,
        random_seed=42,
    ):
        super().__init__()

        self.images_path = sorted(glob(os.path.join(data_path, "images", "*")))
        self.masks_path = sorted(glob(os.path.join(data_path, "labels", "*")))

        if not len(self.images_path) == len(self.masks_path):
            raise ValueError("The number of images and masks do not match.")

        # Split the dataset into train, eval, test sets
        train_images, temp_images, train_masks, temp_masks = train_test_split(
            self.images_path,
            self.masks_path,
            test_size=(1 - train_ratio),
            random_state=random_seed,
        )
        eval_size = eval_ratio / (1 - train_ratio)
        eval_images, test_images, eval_masks, test_masks = train_test_split(
            temp_images, temp_masks, test_size=(1 - eval_size), random_state=random_seed
        )

        if mode == "train":
            self.images_path, self.masks_path = train_images, train_masks
            self.init_augmentations(augmentations=augmentations)
        elif mode == "eval":
            self.images_path, self.masks_path = eval_images, eval_masks
            self.augmentations = A.Compose(
                [
                    A.Resize(512, 512, interpolation=Image.Resampling.NEAREST),
                    ToTensorV2(),
                ]
            )
        elif mode == "test":
            self.images_path, self.masks_path = test_images, test_masks
            self.augmentations = A.Compose(
                [
                    A.Resize(512, 512, interpolation=Image.Resampling.NEAREST),
                    ToTensorV2(),
                ]
            )
        else:
            raise ValueError("Mode should be 'train', 'eval', or 'test'.")

        self.n_samples = len(self.images_path)

        for i in self.masks_path:
            if not os.path.exists(i):
                print(f"file {i} does not exist.")

    def init_augmentations(self, augmentations=None):
        if augmentations is None:
            self.augmentations = A.Compose(
                [
                    A.Resize(512, 512, interpolation=Image.Resampling.NEAREST),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    ToTensorV2(),
                ]
            )
        else:
            self.augmentations = augmentations

    def __getitem__(self, index):
        image = Image.open(self.images_path[index]).convert("RGB")
        mask = Image.open(self.masks_path[index]).convert("L")
        image = np.array(image)
        mask = np.array(mask)

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.float() / 255.0
        mask = mask.float().unsqueeze(0) / 255.0

        return {"image": image, "mask": mask}

    def __len__(self):
        return self.n_samples


class BUSIDataset(DRIVEDataset):
    def __init__(self, data_path, augmentations=None):
        super().__init__(data_path, augmentations)
        self.images_path = sorted(glob(os.path.join(data_path, "images", "*")))
        self.masks_path = sorted(glob(os.path.join(data_path, "mask", "0", "*")))
        self.n_samples = len(self.images_path)
        # self.augmentations = augmentations
        self.init_augmentations(augmentations=augmentations)

        for i in self.masks_path:
            if not os.path.exists(i):
                print(f"file {i} does not exist.")

    def __getitem__(self, index):
        # print(self.images_path[index])
        image = Image.open(self.images_path[index]).convert("L")
        mask = Image.open(self.masks_path[index]).convert("L")
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        # Normalize and convert to tensor
        image = image.float() / 255.0
        mask = mask.float().unsqueeze(0) / 255.0

        return {"image": image, "mask": mask}


if __name__ == "__main__":
    # 定义增强变换
    augmentations = None

    # 使用增强方法创建数据集
    # data_path = "../../Dataset/DRIVE/training"
    # dataset = DRIVEDataset(data_path, augmentations=augmentations)
    data_path = "../../Dataset/BUSI/"
    dataset = BUSIDataset(data_path, augmentations=augmentations)
    # 示例用法
    show(dataset, 15, "./BUSI_visualization.png")

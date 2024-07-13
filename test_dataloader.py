import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from PIL import Image
import os
import numpy as np
from datasets import DRIVEDataCollator, DRIVEDataset

import debugpy

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9725))
    print("Waiting for debugger attach")
    print("the python code is test.py")
    print("the host is: localhost, the port is: 9725")
    debugpy.wait_for_client()
except Exception as e:
    pass


if __name__ == "__main__":
    print("Hello world")

    # Example usage:
    dataset = DRIVEDataset(data_path="../Dataset/DRIVE/training")
    data_collator = DRIVEDataCollator()
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)

    counter_num = 0
    for batch in dataloader:
        print(
            f"Batch image shape: {batch['image'].shape}, Batch mask shape: {batch['mask'].shape}"
        )
        counter_num += 1
        if counter_num >= 10:
            break

import matplotlib.pyplot as plt
import os
import torch
import numpy as np

from data import DRIVEDataset, DRIVEDataCollator
from model import UNet

from transformers import TrainingArguments, Trainer
from trainer import CustomTrainer, compute_metrics

import debugpy

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 7925))
    print("Waiting for debugger attach")
    print("the python code is test_trainer.py")
    print("the host is: localhost, the port is: 7925")
    debugpy.wait_for_client()
except Exception as e:
    pass

# 假设你已经有加载和预处理后的图像和标签数据
train_dataset = DRIVEDataset(data_path="../Dataset/DRIVE/training")
eval_dataset = DRIVEDataset(data_path="../Dataset/DRIVE/test")
data_collator = DRIVEDataCollator()
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=10,
    logging_dir="./logs",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
)

model = UNet(in_channels=3, out_channels=1)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

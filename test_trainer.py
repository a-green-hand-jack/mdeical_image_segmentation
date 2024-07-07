import os
import torch
import numpy as np

from data import DRIVEDataset, DRIVEDataCollator, BUSIDataset, BUSIDataCollator
from model import UNetModel, UNetConfig

from transformers import TrainingArguments, Trainer, LlamaModel
from trainer import CustomTrainer, compute_metrics
from torch.utils.data import Dataset, DataLoader, Subset
import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 7925))
#     print("Waiting for debugger attach")
#     print("the python code is test_trainer.py")
#     print("the host is: localhost, the port is: 7925")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# 假设你已经有加载和预处理后的图像和标签数据
# train_dataset = BUSIDataset(data_path="../Dataset/BUSI")
# eval_dataset = BUSIDataset(data_path="../Dataset/BUSI")
# data_collator = BUSIDataCollator()
train_dataset = DRIVEDataset(data_path="../Dataset/DRIVE/training")
eval_dataset = DRIVEDataset(data_path="../Dataset/DRIVE/training", mode="eval")
# 建立对应的测试数据集
test_dataset = DRIVEDataset(data_path="../Dataset/DRIVE/test", mode="test")
data_collator = DRIVEDataCollator()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir="./logs",
    logging_steps=100,
    num_train_epochs=5000,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_steps=1000,
    save_total_limit=5,
    remove_unused_columns=False,
    label_names=["labels"]
)

config = UNetConfig(in_channels=3, out_channels=1, unet_type="UNet")
model = UNetModel(config)

# model = UNet(in_channels=3, out_channels=1)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()
# 评估模型
trainer.evaluate()
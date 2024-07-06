import os
import torch
import numpy as np

from data import DRIVEDataset, DRIVEDataCollator, BUSIDataset, BUSIDataCollator
from model import UNet, UNetModel, UNetConfig

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
train_eval_dataset = DRIVEDataset(data_path="../Dataset/DRIVE/training")
# 假设数据集有N个样本，我们想要拆分为80%的训练集和20%的验证集
N = len(train_eval_dataset)
indices = list(range(N))
np.random.shuffle(indices)  # 随机打乱索引

split_idx = int(N * 0.8)  # 计算拆分点

# 创建训练集和验证集的索引
train_indices, eval_indices = indices[:split_idx], indices[split_idx:]
# 使用Subset来创建训练集和验证集
train_dataset = Subset(train_eval_dataset, train_indices)
eval_dataset = Subset(train_eval_dataset, eval_indices)

# 建立对应的测试数据集
test_dataset = DRIVEDataset(data_path="../Dataset/DRIVE/test")
data_collator = DRIVEDataCollator()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    eval_steps=1,
    logging_dir="./logs",
    logging_steps=1,
    num_train_epochs=5000,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    save_steps=1,
    save_total_limit=5,
    remove_unused_columns=False,
    label_names=["labels"]
)

config = UNetConfig(in_channels=3, out_channels=1)
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
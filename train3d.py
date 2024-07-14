import torchio as tio
from transformers import TrainingArguments, Trainer
from datetime import datetime
from pathlib import Path

from datasets import AMOSDataCollator, AMOSDatasetTrain, AMOSDatasetVal
from models import UNet3dConfig, UNet3DModel

import argparse

import debugpy

try:
    debugpy.listen(("localhost", 4325))
    print("Waiting for debugger attach")
    print("the python code is train3d.py")
    print("the host is: localhost, the port is: 4325")
    debugpy.wait_for_client()
except Exception as e:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define parser and add arguments
    parser = argparse.ArgumentParser(description="Training arguments for your model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/unet3d",
        help="Directory to save the training results UNet3D",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../Dataset/AMOS/amos22",
        help="Directory to data",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy (steps or epoch)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Number of steps between logging"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5000,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU/CPU for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU/CPU for evaluation",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between model saves",
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=5, help="Total number of saved models"
    )
    parser.add_argument(
        "--remove_unused_columns",
        action="store_true",
        help="Whether to remove unused columns from input data",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.001,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.005, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="iou",
        help="Metric to determine best model",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Number of input channels for the model",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=1,
        help="Number of output channels for the model",
    )
    parser.add_argument(
        "--unet_type",
        type=str,
        default="UNet_3d",
        help="Chose unet types:UNet, UNet_3Plus, UNet_3Plus_DeepSup ",
    )
    # Add more arguments as needed

    args = parser.parse_args()

    time_map = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).joinpath(time_map).joinpath(args.unet_type)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_dataset = AMOSDatasetTrain(
        paths=[args.data_path],
        transform=tio.Compose(
            transforms=[
                tio.ToCanonical(),
                tio.CropOrPad(target_shape=(32, 32, 32)),
            ]
        ),
        threshold=500,
        pcc=False,
    )
    # 建立对应的训练数据集
    eval_dataset = AMOSDatasetVal(paths=[args.data_path])
    # 建立对应的测试数据集

    data_collator = AMOSDataCollator()

    training_args = TrainingArguments(
        output_dir=output_dir.joinpath("results"),
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_dir=output_dir.joinpath("logs"),
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        label_names=["labels"],
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        metric_for_best_model=args.metric_for_best_model,
    )

    config = UNet3dConfig(
        in_channels=args.in_channels,
        num_channels=args.out_channels,
        unet_type=args.unet_type,
        loss_config={
            "name": "DiceCELoss",
        },
    )
    model = UNet3DModel(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 训练模型
    trainer.train()
    # 评估模型
    trainer.evaluate()

#!/bin/bash

# Define paths and settings
DATA_PATH="../Dataset/DRIVE/training"
OUTPUT_DIR="./results"
EVAL_STRATEGY="steps"
EVAL_STEPS=100
LOGGING_STEPS=100
NUM_TRAIN_EPOCHS=5000
PER_DEVICE_TRAIN_BATCH_SIZE=5
PER_DEVICE_EVAL_BATCH_SIZE=5
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=5
WARMUP_RATIO=0.001
LEARNING_RATE=0.005
WEIGHT_DECAY=0.001
METRIC_FOR_BEST_MODEL="iou"
IN_CHANNELS=3
OUT_CHANNELS=1
UNET_TYPE="UNet_3Plus"

# Run the Python script with predefined arguments
python3 train2d.py \
    --output_dir "$OUTPUT_DIR" \
    --evaluation_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --remove_unused_columns \
    --warmup_ratio "$WARMUP_RATIO" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --metric_for_best_model "$METRIC_FOR_BEST_MODEL" \
    --in_channels "$IN_CHANNELS" \
    --out_channels "$OUT_CHANNELS" \
    --unet_type "$UNET_TYPE" \
    --data_path "$DATA_PATH"


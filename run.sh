#!/bin/bash
set -e

# --------------------------
# Basic Path Configuration
# --------------------------
MODEL_ID="/path/to/model"
QUANT_SAVE_DIR="/path/to/quantized_model"
QUANT_MODEL_WITH_ADAPTER="/path/to/quantized_model_with_adapter"

# --------------------------
# Step 1: ProjQ Quantization
# --------------------------
python main.py \
    --model_id ${MODEL_ID} \
    --bits 2 \
    --group_size 128 \
    --quant_method PROJQ \
    --rank 16 \
    --iteration 5 \
    --save_dir ${QUANT_SAVE_DIR}

# --------------------------
# Step 2: Error Compensation
# --------------------------
python comp_train.py \
    --model_id ${MODEL_ID} \
    --quantized_model_dir ${QUANT_SAVE_DIR} \
    --comp_rank 64 \
    --comp_method lordq

# --------------------------
# Step 3: Commensense-Reasoning Fine-tuning
# --------------------------
python cs_ft.py \
    --model_id ${MODEL_ID} \
    --quantized_model_dir ${QUANT_MODEL_WITH_ADAPTER} \
    --adapter_rank 64 \
    --bits 2 \
    --lora_alpha 16 \
    --learning_rate 1e-4 \
    --save_strategy "epoch" \
    --evaluation_strategy "steps" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --seed 11 \
    --logging_steps 5 \
    --num_train_epochs 3 \
    --block_size 256 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --remove_unused_columns False

# --------------------------
# Step 4: GSM8K fine Tuning&Evaluation
# --------------------------
python gsm8k_ft.py \
    --model_id ${MODEL_ID} \
    --quantized_model_dir ${QUANT_MODEL_WITH_ADAPTER} \
    --rank 64 \
    --bits 2 \
    --lora_alpha 16 \
    --learning_rate 5e-5 \
    --seed 11 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --save_strategy "epoch" \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --output_dir /path/to/gsm8k_lora \
    --remove_unused_columns False


python gsm8k_eval.py \
    --model_name_or_path ${MODEL_ID} \
    --quantized_model_dir ${QUANT_MODEL_WITH_ADAPTER} \
    --batch_size 16

# --------------------------
# Step 5: WikiText Tuning & Evaluation
# --------------------------
python wiki_ft.py \
    --model_id ${MODEL_ID} \
    --quantized_model_dir ${QUANT_MODEL_WITH_ADAPTER} \
    --bits 2 \
    --rank 64 \
    --lora_alpha 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --block_size 512 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --save_strategy "epoch" \
    --evaluation_strategy "steps" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --do_train \
    --do_eval \
    --remove_unused_columns False

python wiki_eval.py \
    --model_id ${MODEL_ID} \
    --quantized_model_dir ${QUANT_MODEL_WITH_ADAPTER} \
    --batch_size 16

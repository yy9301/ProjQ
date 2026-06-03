

# ProjQ
This repository contains code for the ICML 2026 paper [ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](https://arxiv.org/abs/2606.00494).

## Introduction
 We propose ProjQ, a novel framework for constraining quantization noise to the low-rank manifold via orthogonal subspace projection. We derive an efficient alternating algorithm that shapes the quantization noise into a low-rank structure, effectively offloading dominant error components to the subsequent adapter while minimizing the residual error in the orthogonal ”uncorrectable” subspace. Our algorithm consists of two phases: (1)Subspace-aware Quantization; and (2) Error Compensation with LoRA adapter initialization.
 The current release includes the following features:

* Phase 1 iterative projection with GPTQ quantizer: `gptqmodel/quantization/projq_gptq.py`.
* Phase 2 Low-rank error compensation and lora adapter initialization: `gptqmodel/eora/lordq.py`.
* LoRA fine-tuning tasks including GSM8K, WikiText-2 and Commonsense Reasoning: `/peft`.
* Evaluating the performance of quantized models on several ZeroShot tasks: `eval_acc.py`.
* datasets for language model evaluation: `datautils.py`.
* Evaluating the perplexity of quantized models on several language generation tasks is included in the main execution script, see the details below.


## Installation

```bash
git clone https://github.com/yourname/ProjQ.git
cd ProjQ
pip install -r requirements.txt
```
## Quantization Error Compensation
The code is primarily tested and run on Llama 2, Qwen2.5-Instruct, and Qwen3 models. Since the implementation is adapted based on [GPTQModel](https://github.com/ModelCloud/GPTQModel), running it on other models can also refer to the corresponding relevant instructions and documentation.
 `--rank` represents the designed rank which governs the dimensionality of the subspace used to shape the quantization noise during the Phase 1. The number of alternating iterations `--iteration` is set to 5. The following command runs the 2-bit quantization process.
```bash
python main.py \
    --model_id /path/to/model \
    --bits 2 \
    --group_size 128 \
    --quant_method PROJQ \
    --rank 16 \
    --iteration 5 \
    --save_dir /path/to/quantized_model
```
After obtaining the quantized model, run the following code to perform error compensation, which also yields the initial adapter. Here, `--comp_rank` denotes the adapter rank in Phase 2.

```bash
python comp_train.py \
    --model_id /path/to/model \
    --quantized_model_dir /path/to/quantized_model \
    --comp_rank 64 \
    --comp_method lordq  

```
## Fine-tuning for Downstream Tasks
The code for LoRA fine-tuning tasks is located in the `peft/`, which includes three types of tasks: GSM8K, WikiText-2 and Commonsense Reasoning. `peft/gsm8k_ft.py` and `peft/wiki_ft.py` are used for LoRA fine-tuning; `peft/gsm8k_eval.py` and `peft/wiki_eval.py` are used for the corresponding evaluation. `peft/cs_ft.py` includes both training and evaluation.

You can find fine-tuning implementation in `run.sh`. Below is an example of fine-tuning and evaluation on the **GSM8K** task. Here, `--rank` must be the same as the adapter rank in phase 2.
```bash
python gsm8k_ft.py \
    --model_id /path/to/model \
    --quantized_model_dir /path/to/quantized_model_with_adapter \
    --rank 64\
    --bits 2\
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
```
```bash
python gsm8k_eval.py \
    --model_name_or_path /path/to/model \
    --quantized_model_dir /path/to/quantized_model_with_adapter \
    --batch_size 16
```

## Acknowledgements

This project is based on and modified from [GPTQModel](https://github.com/ModelCloud/GPTQModel) and [LoftQ](https://github.com/yxli2123/LoftQ/).
Sincere thanks for their efforts.


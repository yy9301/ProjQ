import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import evaluate
import datasets
import torch
from datasets import load_dataset
import torch.nn as nn

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from gptqmodel import GPTQModel, BACKEND
from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from safetensors.torch import save_file
import json
from peft import PeftModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import default_data_collator

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    model_id: str = field(metadata={"help": "Original model ID (for tokenizer and config)"})
    quantized_model_dir: str = field(metadata={"help": "Directory containing quantized GPTQ model"})
    comp_save_path: str = field(metadata={"help": "Directory where finetuned adapter is saved"})
    rank: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "Torch dtype: auto, float16, bfloat16, float32"})

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    
    
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
import torch.nn.functional as F

class GPTQCausalLMWrapper(torch.nn.Module):
    def __init__(self, gptq_model):
        super().__init__()
        self.model = gptq_model

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # GPTQModel 可能返回 tuple 或 BaseOutput
        if isinstance(outputs, tuple):
            logits = outputs[0]
        elif isinstance(outputs, CausalLMOutputWithPast):
            logits = outputs.logits
        else:
            logits = outputs  # 直接 tensor

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )


# ================== 检查 LoRA adapter 是否生效 ==================
def check_lora_active(model, tokenizer):
    model.eval()
    # 取一条随机文本
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(next(model.parameters()).device)

    # forward 正常
    with torch.no_grad():
        normal_logits = model(input_ids=input_ids).logits

    # 暂时清零 LoRA 参数
    lora_params = []
    for name, p in model.named_parameters():
        if "lora_" in name:
            lora_params.append((name, p.clone()))
            p.data.zero_()

    with torch.no_grad():
        zeroed_logits = model(input_ids=input_ids).logits

    # 恢复 LoRA 参数
    for name, p_clone in lora_params:
        model.get_parameter(name).data.copy_(p_clone.data)

    # 检查 logits 是否有差异
    if torch.allclose(normal_logits, zeroed_logits, atol=1e-6):
        raise RuntimeError("LoRA adapter does not affect model output! Check load order or keys.")
    else:
        logger.info(" LoRA adapter is active and affecting model output.")



# ==================== Main ====================
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ========== Load tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ========== Load quantized GPTQ model ==========
    torch_dtype = getattr(torch, model_args.torch_dtype)
    model = GPTQModel.load(
        model_args.quantized_model_dir,
        device_map="auto",
        torch_dtype=torch_dtype,
        backend=BACKEND.TORCH,
        use_triton=True,
    )
    model.optimize()
    model = model.to(torch_dtype)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # ========== Inject LoRA ==========
    model_name_lower = model_args.model_id.lower()  
    if any(name in model_name_lower for name in ["opt"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    elif any(name in model_name_lower for name in ["llama", "mistral", "falcon", "qwen", "baichuan"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif any(name in model_name_lower for name in ["phi"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    else:
        raise ValueError(f"Unsupported model: {model_args.model_id}")

    lora_config = LoraConfig(
        r=model_args.rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ========== Load finetuned adapter ==========
    adapter_path = os.path.join(model_args.comp_save_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    lora_state = load_file(adapter_path)
    for n, p in model.named_parameters():
        if "lora_" in n:
            # 去掉 .default 后缀
            key = n.replace(".default", "")
            if key in lora_state:
                p.data.copy_(lora_state[key])
    
    print("LoRA state keys:", list(lora_state.keys())[:10])
    print("Model trainable params:", [n for n,p in model.named_parameters() if p.requires_grad][:10])
    
    # optional key remapping if needed
    model.load_state_dict(lora_state, strict=False)
    logger.info(f"Loaded finetuned LoRA adapter from {adapter_path}")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("LoRA param:", n, p.shape)
            
    check_lora_active(model, tokenizer)


    # ========== Load evaluation dataset ==========
    dataset_files = {
        "validation": "/code/datasets/wikitext/wikitext-2-raw-v1/validation-00000-of-00001.parquet"
    }
    raw_datasets = load_dataset("parquet", data_files=dataset_files)

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = raw_datasets["validation"].map(tokenize_fn, batched=True, remove_columns=["text"])

    # Group into block_size chunks
    block_size = 1024
    def group_texts(examples):
        concat = list(chain(*examples["input_ids"]))
        total_len = (len(concat)//block_size)*block_size
        input_ids = [concat[i:i+block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": input_ids, "labels": input_ids.copy()}

    eval_dataset = tokenized.map(group_texts, batched=True, remove_columns=tokenized.column_names)
    if data_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # ========== Evaluate ==========
    model = GPTQCausalLMWrapper(model)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()
    print(metrics.keys())

    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
    
# python wiki_eval-wrap.py \
#     --model_id /data/llama-2-7b-hf \
#     --quantized_model_dir /data/llama2-7b-2bit-rank4-projq \
#     --comp_save_path /data/llama2-7b-2bit-rank4-projq/comp_adapter_train_64/llama2-7b-2bit-rank4-projq_peft_2bit_r64_alpha16 \
#     --rank 64 \
#     --lora_alpha 16


# python wiki_eval-wrap.py \
#     --model_id /data/llama-2-7b-hf \
#     --quantized_model_dir /data/llama2-7b-2bit-GPTQ \
#     --comp_save_path /data/llama2-7b-2bit-GPTQ/comp_adapter_train_64/llama2-7b-2bit-GPTQ_peft_2bit_r64_alpha16 \
#     --rank 64 \
#     --lora_alpha 16


# python wiki_eval-wrap.py \
#     --model_id /data/llama-2-7b-hf \
#     --quantized_model_dir /data/llama2-7b-3bit-rank4-projq \
#     --comp_save_path /data/llama2-7b-3bit-rank4-projq/comp_adapter_train_64/llama2-7b-3bit-rank4-projq_peft_3bit_r64_alpha16 \
#     --rank 64 \
#     --lora_alpha 16
import logging
import os
import sys
import math
import json
from dataclasses import dataclass, field
from typing import Optional
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datautils as ld
from eval_acc import *
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer, DataCollatorForLanguageModeling
from gptqmodel import GPTQModel, BACKEND
from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
import datasets
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_id: str = field(metadata={"help": "Base model path or huggingface model id"})
    quantized_model_dir: str = field(metadata={"help": "Directory of GPTQ quantized model"})
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Path to your initial LoRA adapter"})
    adapter_rank: int = field(default=8, metadata={"help": "LoRA adapter rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_alpha: int = field(default=4, metadata={"help": "LoRA alpha"})
    torch_dtype: str = field(default="float16", metadata={"help": "torch dtype for model"})
    comp_save_path: Optional[str] = field(default=None, metadata={"help": "Path to save/load LoRA adapter"})
    bits: int = field(
        default=4,
        metadata={"help": "quantization bits, support 2/3/4 bit"},
    )


@dataclass
class DataArguments:
    block_size: Optional[int] = field(default=256, metadata={"help": "Sequence length"})
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=4)


def save_curve(y_values, x_values=None, title="Loss Curve", save_dir=".", prefix="train"):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    if x_values is None:
        x_values = list(range(1, len(y_values) + 1))
        
    plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel("Step")
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f"{prefix}_curve.png"))
    plt.close()
    json_file = os.path.join(save_dir, f"{prefix}_curve.json")
    with open(json_file, "w") as f:
        json.dump({"x": x_values, "y": y_values}, f, indent=2)



def piqa_context(item):
    return [f"Q: {item['goal']}\nA: {item['choices'][0]}",
            f"Q: {item['goal']}\nA: {item['choices'][1]}"]

def storycloze_context(item):
    return [f"{item['context']} {item['choices'][0]}",
            f"{item['context']} {item['choices'][1]}"]

def arc_context(item):
    return [f"Q: {item['question']}\nA: {choice}" for choice in item['choices']]

def boolq_context(item):
    return [f"Question: {item['question']}\nPassage: {item['passage']}\nAnswer: True",
            f"Question: {item['question']}\nPassage: {item['passage']}\nAnswer: False"]

def evaluate_lm_tasks(model, tokenizer, seq_len, nsamples, model_id):
    os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    task_results = {}
    
    _, piqa_test = ld.get_piqa(nsamples=nsamples, seed=0, seqlen=seq_len, model=model_id)
    task_results["piqa"] = eval_piqa(model, tokenizer, piqa_test, model.device)
    
    _, sc_test = ld.get_SC(nsamples=nsamples, seed=0, seqlen=seq_len, model=model_id)
    task_results["storycloze"] = eval_SC(model, tokenizer, sc_test, model.device)
    
    _, arc_easy_test = ld.get_arc_easy(nsamples=nsamples, seed=0, seqlen=seq_len, model=model_id)
    task_results["arc_easy"] = eval_arc_E(model, tokenizer, arc_easy_test, model.device)
    
    _, arc_challenge_test = ld.get_arc_challenge(nsamples=nsamples, seed=0, seqlen=seq_len, model=model_id)
    task_results["arc_challenge"] = eval_arc_C(model, tokenizer, arc_challenge_test, model.device)
    
    _, boolq_test = ld.get_boolq(nsamples=nsamples, seed=0, seqlen=seq_len, model=model_id)
    task_results["boolq"] = eval_boolq(model, tokenizer, boolq_test, model.device)
    
    return task_results

def preprocess_commonsense(example):
    question = example["question"]
    choices = example["choices"]  # {'label': [...], 'text': [...]}
    answer_key = example["answerKey"]

    try:
        answer_index = choices['label'].index(answer_key)
    except ValueError:
        print("Warning: answerKey not in choices labels")
        answer_index = 0 

    answer_text = choices['text'][answer_index]

    # 构造 LM 输入
    text = f"Q: {question}\nA: {answer_text}"
    return {"text": text}

def remap_lora_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if ".lora_A.weight" in k:
            k = k.replace(".lora_A.weight", ".lora_A.default.weight")
        if ".lora_B.weight" in k:
            k = k.replace(".lora_B.weight", ".lora_B.default.weight")
        if k.startswith("base_model.model.model."):
            k = k.replace(
                "base_model.model.model.",
                "base_model.model.model.model."
            )
        new_state[k] = v
    return new_state

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Logging & seed
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    set_seed(training_args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

   # Load GPTQ quantized model
    logger.info(f"Loading GPTQ quantized model from {model_args.quantized_model_dir}")
    torch_dtype = getattr(torch, model_args.torch_dtype)
    model = GPTQModel.load(
        model_args.quantized_model_dir,
        device_map="auto",
        torch_dtype=torch_dtype,
        backend=BACKEND.TORCH,
        use_triton=False
    )
    model.optimize()
    model = prepare_model_for_kbit_training(model)

    # Determine target modules for commonsense model (LLaMA/OPT/Falcon etc.)
    model_name_lower = model_args.model_id.lower()  
    if any(name in model_name_lower for name in ["opt"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    elif any(name in model_name_lower for name in ["qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif any(name in model_name_lower for name in ["llama", "mistral", "falcon"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif any(name in model_name_lower for name in ["phi"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    else:
        raise ValueError(f"Unsupported model: {model_args.model_id}")

    # Setup LoRA
    lora_config = LoraConfig(
        r=model_args.adapter_rank,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load existing adapter if exists
    if model_args.comp_save_path is None:
        adapter_dir_name = f"comp_adapter_train_{model_args.adapter_rank}"
        comp_save_path = os.path.join(model_args.quantized_model_dir, adapter_dir_name)
    else:
        comp_save_path = model_args.comp_save_path
    adapter_state_path = os.path.join(comp_save_path, "adapter_model.safetensors") 
    if not os.path.exists(adapter_state_path):
        logger.warning(f"No adapter found at {adapter_state_path}, training LoRA from scratch.")

    
    if os.path.exists(adapter_state_path): 
        lora_state = load_file(adapter_state_path) 
        lora_state = remap_lora_keys(lora_state)
        model.load_state_dict(lora_state, strict=False)

    model.config.use_cache = False
    
    # 确认 LoRA 加载
    # model.print_trainable_parameters()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable param: {name}")
    # missing, unexpected = model.load_state_dict(lora_state, strict=False)
    # logger.info(f"Missing keys: {missing}")
    # logger.info(f"Unexpected keys: {unexpected}")
    
    # --------- Load common sense dataset ---------
    raw_datasets = load_dataset(
        "parquet",
        data_files={"train": "/code/datasets/commonsense_qa/train-00000-of-00001.parquet"},
        split="train"
    )
    raw_datasets = raw_datasets.map(preprocess_commonsense)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_args.block_size,
            padding=False
        )
    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets.column_names
    )
    
    if data_args.max_train_samples is not None:
        tokenized_datasets = tokenized_datasets.shuffle(seed=42).select(range(data_args.max_train_samples))

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --------- Collate function ---------
    # def collate_fn(batch):
    #     input_ids = torch.nn.utils.rnn.pad_sequence(
    #         [torch.tensor(x["input_ids"]) for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id
    #     )
    #     labels = torch.nn.utils.rnn.pad_sequence(
    #         [torch.tensor(x["labels"]) for x in batch], batch_first=True, padding_value=IGNORE_INDEX
    #     )
    #     attention_mask = input_ids.ne(tokenizer.pad_token_id)
    #     return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    # --------- LoRA Fine-tuning ---------
    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_datasets)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    model_name = os.path.basename(model_args.quantized_model_dir.rstrip("/"))
    save_dir_name = f"{model_name}_peftcs_{model_args.bits}bit_r{model_args.adapter_rank}_alpha{model_args.lora_alpha}_epoch{training_args.num_train_epochs}_lr{training_args.learning_rate}"
    save_dir = os.path.join(comp_save_path, save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Finetuned Adapter saved to custom path: {save_dir}")

    train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    save_curve(
        train_losses,
        title="Training Loss",
        save_dir=os.path.join(save_dir, "cs curves"),
        prefix="train"
    )
    # --------- Evaluate on LM tasks (PIQA, StoryCloze, etc.) ---------
    logger.info("Running LM task evaluation after fine-tuning...")
    model.eval()
    with torch.no_grad():
        task_results = evaluate_lm_tasks(
            model, tokenizer, seq_len=data_args.block_size, nsamples=2048, model_id=model_args.model_id
        )
    for task, acc in task_results.items():
        logger.info(f"{task} Accuracy: {acc:.4f}")




if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

# python cs_ft.py \
#     --model_id /data/qwen2.5-7B-Instruct \
#     --quantized_model_dir /data/RTN/qwen2.5-7b-2bit-rank4-LOFTQ \
#     --adapter_rank 64\
#     --bits 2\
#     --lora_alpha 16 \
#     --learning_rate 1e-4 \
#     --save_strategy "epoch" \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --seed 11 \
#     --logging_steps 5 \
#     --num_train_epochs 3 \
#     --block_size 256 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --comp_save_path /data/RTN/qwen2.5-7b-2bit-rank4-LOFTQ/loftq_initial_64 \
#     --remove_unused_columns False

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import Trainer

from datasets import load_dataset
import transformers
from transformers import (
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

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    # 若 lora_init=True：从零初始化 LoRA 适配器（适用于 QLoRA 或全精度 LoRA）
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    
    full_precision:  bool = field(
        default=False,
        metadata={"help": "False: Use bitsandbytes Linear4bit, real quantization"
                          "True: Use quantization equivalent fp16/fp32 weights."
                  },
    )
    rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    model_id: str = field(
        default=None,
        metadata={"help": "Original model ID (for tokenizer and config)"}
    )
    quantized_model_dir: str = field(
        default=None,
        metadata={"help": "Directory containing quantized model files"}
    )
    bits: int = field(
        default=4,
        metadata={"help": "quantization bits, support 2/3/4 bit"},
    )
    comp_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save/load compensation adapter"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model weights, bfloat16 is better for QLoRA"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )
@dataclass
class DataArguments:
    data_name: str = field(
        default="gsm8k",
        metadata={"help": "Dataset name."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    # max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm (for clipping)"})
    evaluation_strategy: str = field(default="no", metadata={"help": "Evaluation strategy to use"})

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(
        "parquet",  # 数据格式（json/csv/text等）
        data_files={
            "train": "/code/datasets/gsm8k/train-00000-of-00001.parquet",  # 本地训练集路径
            "test": "/code/datasets/gsm8k/test-00000-of-00001.parquet"  # 可选：测试集/验证集
        }
    )
    train_set = dataset['train']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def save_curve(y_values, x_values=None, title="Loss Curve", save_dir=".", prefix="train"):
    """
    保存训练/评估曲线及对应 JSON
    y_values: list[float]，loss 或 ppl
    x_values: list[int] 或 None，x 轴
    title: str, 图标题
    save_dir: str, 保存目录
    prefix: str, 文件名前缀 train/eval
    """
    os.makedirs(save_dir, exist_ok=True)
    if x_values is None:
        x_values = list(range(1, len(y_values) + 1))

    # 保存图像
    plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel("Step")
    plt.ylabel(title)
    plt.title(title)
    png_file = os.path.join(save_dir, f"{prefix}_curve.png")
    plt.savefig(png_file)
    plt.close()

    # 保存 JSON
    json_file = os.path.join(save_dir, f"{prefix}_curve.json")
    data = {"x": x_values, "y": y_values}
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
        
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

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.comp_save_path is None:
        adapter_dir_name = f"comp_adapter_train_{model_args.rank}"
        comp_save_path = os.path.join(model_args.quantized_model_dir, adapter_dir_name)
    else:
        comp_save_path = model_args.comp_save_path


    # Load quantized model with GPTQ
    logger.info(f"Loading quantized model from {model_args.quantized_model_dir}")
    torch_dtype = getattr(torch, model_args.torch_dtype)
    model = GPTQModel.load(
        model_args.quantized_model_dir,
        device_map="auto",
        torch_dtype=torch_dtype,
        backend=BACKEND.TORCH,
        use_triton=True
    )
    model.optimize() 
    model = prepare_model_for_kbit_training(model)
    
    ##########################
    #       Peft Model       #
    ##########################
    
    task_type = TaskType.CAUSAL_LM
    # 配置目标模块，适配主流LLM模型
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
        task_type=task_type,
        r=model_args.rank, 
        lora_alpha=model_args.lora_alpha, 
        target_modules=target_modules, 
        lora_dropout=model_args.lora_dropout, 
        bias="none", 
    )
    model = get_peft_model(model, lora_config)
    adapter_state_path = os.path.join(comp_save_path, "adapter_model.safetensors") 
    if os.path.exists(adapter_state_path): 
        lora_state = load_file(adapter_state_path) 
        lora_state = remap_lora_keys(lora_state)
        model.load_state_dict(lora_state, strict=False)

    model.config.use_cache = False
    
    # # 确认 LoRA 已经成功注入
    # print(f"======================确认 LoRA 已经成功注入=======================")
    # model.print_trainable_parameters()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable param: {name}")
    # missing, unexpected = model.load_state_dict(lora_state, strict=False)
    # logger.info(f"Missing keys: {missing}")
    # logger.info(f"Unexpected keys: {unexpected}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )
#     # ================== 跳过最后一个 step ==================
#     train_dataloader = DataLoader(
#         data_module["train_dataset"],
#         batch_size=training_args.per_device_train_batch_size,
#         shuffle=True,
#         collate_fn=data_module["data_collator"],
#     )

#     steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
#     total_steps = int(steps_per_epoch * training_args.num_train_epochs)

#     training_args.max_steps = total_steps - 1

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(trainer.train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    model_name = os.path.basename(model_args.quantized_model_dir.rstrip("/"))
    save_dir_name = f"{model_name}_peftgsm8k_{model_args.bits}bit_r{model_args.rank}_alpha{model_args.lora_alpha}_epoch{training_args.num_train_epochs}_lr{training_args.learning_rate}"
    if model_args.comp_save_path is None:
        adapter_dir_name = f"comp_adapter_train_{model_args.rank}"
        comp_save_path = os.path.join(model_args.quantized_model_dir, adapter_dir_name)
    else:
        comp_save_path = model_args.comp_save_path
    save_dir = os.path.join(comp_save_path, save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f" finetuned Adapter saved to custom path: {save_dir}")

    train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    save_curve(
        train_losses,
        title="Training Loss",
        save_dir=os.path.join(save_dir, "curves"),
        prefix="train"
    )
    print(os.listdir(training_args.output_dir))



if __name__ == "__main__":
    train()

    
# python gsm8k_ft.py \
#     --model_id /data/llama-2-7b-hf \
#     --quantized_model_dir /data/llama2-7b-2bit-rank4-projq \
#     --rank 64\
#     --bits 2\
#     --lora_alpha 16 \
#     --learning_rate 5e-5 \
#     --seed 11 \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --lr_scheduler_type "cosine" \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --logging_steps 10 \
#     --output_dir /data/experiments/gsm8k_lora \
#     --expt_name llama2-7b-2bit-rank4-64-alpha16-projq-gsm8k \
#     --remove_unused_columns False 


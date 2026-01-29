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


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.37.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:

    model_id: str = field(
        metadata={"help": "Original model ID (for tokenizer and config)"}
    )
    rank: int = field(
        default=8,
        metadata={"help": "Rank of LoRA adapters"}
    )
    comp_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save/load compensation adapter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "lora_alpha, QLORA suggest alpha = 2*rank"},
    )
    full_precision: bool = field(
        default=False,
        metadata={"help": "True: full precision no quantization, False: use bitsandbytes quantization"
                  },
    )
    bits: int = field(
        default=4,
        metadata={"help": "quantization bits, support 2/3/4 bit"},
    )

    quantized_model_dir: str = field(
        default=None,
        metadata={"help": "Directory containing quantized model files"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    
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


    
def save_curve(y_values, x_values=None, title="Loss Curve", save_dir=".", prefix="train"):
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

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed
    set_seed(training_args.seed)

    # Load local WikiText2 dataset 
    data_files = {
        "train": "/code/datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet", 
        "test": "/code/datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet",
        "validation": "/code/datasets/wikitext/wikitext-2-raw-v1/validation-00000-of-00001.parquet"
    }
    
    raw_datasets = load_dataset(
        "parquet",
        data_files=data_files
    )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_id:
        config = AutoConfig.from_pretrained(model_args.model_id, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  


    
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
    
    
    lora_config = LoraConfig( 
        r=model_args.rank, 
        lora_alpha=model_args.lora_alpha, 
        target_modules=target_modules, 
        lora_dropout=model_args.lora_dropout, 
        bias="none", 
        task_type="CAUSAL_LM", 
    ) 
    model = get_peft_model(model, lora_config) 
    adapter_state_path = os.path.join(comp_save_path, "adapter_model.safetensors") 
    if os.path.exists(adapter_state_path): 
        lora_state = load_file(adapter_state_path)
        lora_state = remap_lora_keys(lora_state)
        model.load_state_dict(lora_state, strict=False)
    
    model.config.use_cache = False
    
    # 确认 LoRA 已经成功注入
    # print(f"======================确认 LoRA 已经成功注入=======================")
    # model.print_trainable_parameters()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable param: {name}")
    # missing, unexpected = model.load_state_dict(lora_state, strict=False)
    # logger.info(f"Missing keys: {missing}")
    # logger.info(f"Unexpected keys: {unexpected}")
    # # ======================确认 LoRA 已经成功注入=======================
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
            
    # Group texts into chunks of block_size
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if "train" not in tokenized_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = lm_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))


    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = lm_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)
         

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,

    )
    
    # 显式检查
    # print(f"======================显式检查=======================")
    # logger.info(model)
    

    model_name = os.path.basename(model_args.quantized_model_dir.rstrip("/"))
    save_dir_name = f"{model_name}_peft_{model_args.bits}bit_r{model_args.rank}_alpha{model_args.lora_alpha}_epoch{training_args.num_train_epochs}_lr{training_args.learning_rate}"
    save_dir = os.path.join(comp_save_path, save_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    adapter_file = os.path.join(save_dir, "adapter_model.safetensors")  # 训练好的adapter文件

    if os.path.exists(adapter_file):
        logger.info(f"Found existing finetuned adapter at {adapter_file}, loading for evaluation...")
        lora_state = load_file(adapter_file)
        model.load_state_dict(lora_state, strict=False)
        adapter_exists = True
    else:
        adapter_exists = False
        

    if training_args.do_train and not adapter_exists:
    # if training_args.do_train:
        logger.info("No existing adapter found, starting training...")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train()
        trainer.save_model()  
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # 保存到自定义路径
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Finetuned Adapter saved to custom path: {save_dir}")

        train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        save_curve(
            train_losses,
            title="Training Loss",
            save_dir=os.path.join(save_dir, "curves"),
            prefix="train"
        )




if __name__ == "__main__":
  
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
    


# python wiki_ft.py \
#     --model_id /data/llama-2-7b-hf \
#     --quantized_model_dir /data/llama2-7b-2bit-rank4-projq \
#     --bits 2 \
#     --rank 64\
#     --lora_alpha 16 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --block_size 512 \
#     --num_train_epochs 1 \
#     --learning_rate 3e-4 \
#     --save_strategy "epoch" \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --do_train \
#     --do_eval False\
#     --remove_unused_columns=False

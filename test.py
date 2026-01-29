import logging
logging.getLogger("gptqmodel").setLevel(logging.WARNING) 
import os
import torch
import math
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig, get_best_device
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.perplexity import Perplexity
from gptqmodel.utils import BACKEND
from lm_eval import evaluate
from lm_eval.utils import make_table
import tempfile
import datautils as ld
import torch.nn.functional as F
from eval_acc import *
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora
import argparse
from datasets import Dataset, DatasetDict
from datautils import get_loaders


def prepare_calibration_data(
    tokenizer,
    rows: int,
    input_max_length: int,
    calib_dataset: str = "c4"
):
    candidate_datas = []

    if calib_dataset == "c4":
        calibration_data = load_dataset(
            'json',
            data_files={'train': '/code/datasets/c4/en/c4-train.00000-of-01024.json.gz'},
            split='train',
            streaming=True
        )

        for sample in calibration_data:
            if "text" not in sample:
                continue

            tokenized = tokenizer(
                sample["text"],
                truncation=True,
                max_length=input_max_length,
                padding=False,
                return_tensors=None
            )

            if len(tokenized["input_ids"]) >= input_max_length:
                input_ids = tokenized["input_ids"][:input_max_length]
                attention_mask = [1] * len(input_ids)
                candidate_datas.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })

            if len(candidate_datas) >= rows * 10:
                break
                
    elif calib_dataset == "wikitext2":
        dataset = load_dataset(
            "parquet",
            data_files={"train": "/code/datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"},
            split="train",
            streaming=True
        )

        for sample in dataset:
            text = sample.get("text", "").strip()
            if len(text) == 0:
                continue

            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=input_max_length,
                padding=False,
                return_tensors=None
            )

            if len(tokenized["input_ids"]) == 0:
                continue

            input_ids = tokenized["input_ids"][:input_max_length]
            attention_mask = [1] * len(input_ids)

            candidate_datas.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            if len(candidate_datas) >= rows * 10:
                break

    elif calib_dataset == "commonsense":
        dataset = load_dataset(
            "parquet",
            data_files={"train": "/code/datasets/commonsense_qa/train-00000-of-00001.parquet"},
            split="train"
        )

        for sample in dataset:
            question = sample.get("question", "").strip()
            choices = sample.get("choices", {}).get("text", [])

            if question == "" or len(choices) == 0:
                continue

            # 构造“完整推理输入”，而不是 instruction
            text = (
                "Question: " + question + "\n"
                + "Choices: " + " ".join(choices)
            )

            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=input_max_length,
                padding=False,
                return_tensors=None
            )

            if len(tokenized["input_ids"]) < 64: 
                continue

            candidate_datas.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": [1] * len(tokenized["input_ids"])
            })

            if len(candidate_datas) >= rows * 10:
                break


    else:
        raise ValueError(f"Unknown calibration dataset: {calib_dataset}")

    import random
    random.seed(42)
    random.shuffle(candidate_datas)

    return candidate_datas[:rows]



def calculate_perplexity(model, tokenizer, dataset_name, seq_len, nsamples, model_id):

    from datautils import get_loaders

    _, testenc = get_loaders(
        name=dataset_name,
        nsamples=nsamples,
        seed=0,
        seqlen=seq_len,
        model=model_id,
    )

    input_ids = testenc.input_ids.to(model.device)
    total_tokens = input_ids.size(1)
    nlls = []

    for i in range(0, total_tokens, seq_len):
        chunk = input_ids[:, i:i + seq_len]
        cur_len = chunk.size(1)

        if cur_len < seq_len:
            pad_len = seq_len - cur_len
            pad = torch.full(
                (1, pad_len),
                tokenizer.pad_token_id,
                device=model.device,
            )
            chunk = torch.cat([chunk, pad], dim=1)

            labels = torch.cat(
                [
                    chunk[:, :cur_len],
                    torch.full((1, pad_len), -100, device=model.device),
                ],
                dim=1,
            )
        else:
            labels = chunk

        with torch.no_grad():
            output = model(chunk, labels=labels)
            neg_log_likelihood = output.loss * cur_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()

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

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GPTQ Model Quantization and Evaluation")
    
    # 模型相关参数
    parser.add_argument("--model_id", type=str, default="/data/facebook-opt-125m", 
                        help="Path or Hugging Face model ID")
    
    # 量化相关参数
    parser.add_argument("--quant_method", type=str, default="PROJQ", 
                        choices=["PROJQ", "GPTQ"],  # 根据实际支持的方法调整
                        help="Quantization method")
    parser.add_argument("--bits", type=int, default=4, 
                        help="Number of quantization bits")
    parser.add_argument("--group_size", type=int, default=128, 
                        help="Group size for quantization")
    parser.add_argument("--rank", type=int, default=None, 
                        help="Rank for PROJQ quantization")
    
    # 数据和评估相关参数
    parser.add_argument("--seq_len", type=int, default=2048, 
                        help="Sequence length for evaluation")
    parser.add_argument("--nsamples", type=int, default=128, 
                        help="Number of samples for evaluation")
    parser.add_argument("--input_max_length", type=int, default=2048, 
                        help="Maximum input length for calibration")
    
    # 其他参数
    parser.add_argument("--save_dir", type=str, default=None, 
                        help="Directory to save quantized model")
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="c4",
        choices=["c4", "commonsense", "wikitext2"],
        help="Calibration dataset"
    )

    
    args = parser.parse_args()
    
    if args.save_dir is None:
        model_name = args.model_id.split("/")[-1]
        save_dir = f"./{args.quant_method.lower()}_{model_name}_{args.bits}bit_r{args.rank}_g{args.group_size}"
    else:
        save_dir = args.save_dir
    
    os.makedirs(save_dir, exist_ok=True)
    quant_model_file = os.path.join(save_dir, "model.safetensors") 
    if os.path.exists(quant_model_file):
        print(f"Found existing quantized model in {save_dir}, skipping quantization.")
        quantized_model = GPTQModel.load(
            save_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            backend=BACKEND.TORCH,
            use_triton=True
        )
    else:
    
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 准备校准数据
        calibration_dataset = prepare_calibration_data(
            tokenizer,
            rows=128,
            input_max_length=args.input_max_length,
            calib_dataset=args.calib_dataset
        )


        quantize_config = QuantizeConfig(
            fail_safe=True,
            quant_method=getattr(METHOD, args.quant_method),
            bits=args.bits,                   # 量化位数
            group_size=args.group_size,       # 分组大小
            desc_act=False,
            sym=False,
            projq_rank=args.rank,
            damp_percent=0.1,
            #dynamic={"-:fc1": {}}  # 跳过 fc1 层   
        )

        # 加载模型并量化
        model = GPTQModel.from_pretrained(
            args.model_id,
            quantize_config=quantize_config,
            torch_dtype=torch.float16,  
            device_map="auto"  
        )

        model.quantize(calibration_dataset, batch_size=16)  
        # 保存并重新加载量化模型
        model.save_quantized(save_dir)
        del model

        # 加载量化模型
        quantized_model = GPTQModel.load(
            save_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            backend=BACKEND.TORCH,
            use_triton=True  
        )

    # 评估Perplexity
    print("=== Perplexity Evaluation ===")
    PERPLEXITY_DATASETS = {
        "c4": {"name": "c4", "split": "validation"},
        "ptb": {"name": "ptb", "split": "validation"},
        "wikitext2": {"name": "wikitext2", "split": "test"}
    }

    for name, config in PERPLEXITY_DATASETS.items():
        ppl = calculate_perplexity(quantized_model, tokenizer, config["name"], args.seq_len, args.nsamples, args.model_id)
        print(f"{name} Perplexity: {ppl:.4f}")

    # 评估LM任务
    print("\n=== LM Evaluation Tasks ===")
    results = evaluate_lm_tasks(quantized_model, tokenizer, args.seq_len, args.nsamples, args.model_id)
    for task, accuracy in results.items():
        print(f"{task} Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 优化内存碎片
    ld.set_seed(42)
    main()
    
# python test.py --model_id /data/llama-2-7b-hf --quant_method GPTQ --bits 2 --group_size 128  --save_dir /data/llama2-7b-2bit-rank64-projq-cs 


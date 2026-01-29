import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm




def eval_arc_E(model, tokenizer, dataset, dev, max_length=512):
    print('Evaluating on ARC-Easy ...')

    model.eval()
    model.to(dev)

    correct = 0
    total = 0

    for example in tqdm(dataset):
        question = example['question']
        options = example['choices']
        label = example['label']

        logprobs = []

        for opt in options:
            prompt = f"Q: {question}\nA: {opt}"

            inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=max_length).to(dev)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # [1, seq_len, vocab]
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs.input_ids[:, 1:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                score = selected.sum().item()
                logprobs.append(score)

        pred = int(torch.tensor(logprobs).argmax().item())
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total
    print(f"ARC-Easy Accuracy: {acc:.4f}")
    return acc

def eval_arc_C(model, tokenizer, dataset, dev, max_length=512):
    print('Evaluating on ARC-Challenge ...')

    model.eval()
    model.to(dev)

    correct = 0
    total = 0

    for example in tqdm(dataset):
        question = example['question']
        options = example['choices']
        label = example['label']

        logprobs = []

        for opt in options:
            prompt = f"Q: {question}\nA: {opt}"

            inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=max_length).to(dev)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # [1, seq_len, vocab]
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs.input_ids[:, 1:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                score = selected.sum().item()
                logprobs.append(score)

        pred = int(torch.tensor(logprobs).argmax().item())
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total
    print(f"ARC-Challenge Accuracy: {acc:.4f}")
    return acc


# def opt_eval_lambada(model, tokenizer, dataset, dev, max_length=512):
#     import torch
#     import torch.nn.functional as F
#     from tqdm import tqdm

#     print('Evaluating on LAMBADA ...')

#     model.eval()
#     model.to(dev)

#     correct = 0
#     total = 0

#     for example in tqdm(dataset):
#         context = example['context']
#         target = example['target']

#         inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=max_length).to(dev)

#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits

#         last_token_logits = logits[0, -1, :]
#         pred_id = torch.argmax(last_token_logits).item()

#         # token-id级别比较而不是字符串
#         target_ids = tokenizer.encode(" " + target, add_special_tokens=False)
#         if len(target_ids) != 1:
#             continue
#         if len(target_ids) == 1 and pred_id == target_ids[0]:
#             correct += 1
#         total += 1

#     acc = correct / total if total > 0 else 0
#     print(f"LAMBADA Accuracy: {acc:.4f}")
#     return acc

def eval_lambada(model, tokenizer, dataset, dev, max_length=1024):
    from tqdm import tqdm
    import torch

    print("Evaluating on LAMBADA (generation-based)...")
    model.eval()
    model.to(dev)

    correct = 0
    total = 0
    skipped = 0

    for example in tqdm(dataset):
        context = example['context']
        target = example['target']

        context_ids = tokenizer.encode(context, return_tensors='pt', add_special_tokens=False).to(dev)
        target_ids = tokenizer.encode(" " + target, add_special_tokens=False)

        if context_ids.shape[1] > max_length - len(target_ids) - 1:
            skipped += 1
            continue

        try:
            outputs = model.generate(
                input_ids=context_ids,
                max_new_tokens=len(target_ids),
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        except:
            skipped += 1
            continue

        gen_ids = outputs[0][-len(target_ids):].tolist()
        if len(target_ids) != 1:
            continue
        if gen_ids == target_ids:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"LAMBADA Accuracy: {acc:.4f} | Evaluated: {total}, Skipped: {skipped}")
    return acc


def eval_piqa(model, tokenizer, dataset, dev, max_length=1024):
    """
    Evaluate multiple-choice PIQA task using log-likelihood ranking.

    Parameters:
        model: HuggingFace causal LM
        tokenizer: matching tokenizer
        dataset: testloader from get_piqa(), each item has 'goal', 'choices', 'label'
        dev: device (e.g., 'cuda')
        max_length: max token length
    """
    print('Evaluating on PIQA ...')

    model.eval()
    model.to(dev)

    correct = 0
    total = 0

    for example in tqdm(dataset):
        goal = example['goal']
        options = example['choices']
        label = example['label']

        logprobs = []

        for opt in options:
            # prompt = f"Q: {goal}\nA: {opt}"
            prompt = f"{goal} {opt}"

            inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=max_length).to(dev)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # [1, seq_len, vocab]

            shift_logits = logits[:, :-1, :]
            shift_labels = inputs.input_ids[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            score = selected.sum().item()
            logprobs.append(score)

        pred = int(torch.tensor(logprobs).argmax().item())
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total
    print(f"PIQA Accuracy: {acc:.4f}")
    return acc


def eval_SC(model, tokenizer, dataset, dev, max_length=512):
    print("Evaluating on StoryCloze ...")

    model.eval()
    model.to(dev)

    correct = 0
    total = 0

    for example in tqdm(dataset):
        context = example['context']
        options = example['choices']
        label = example['label']

        logprobs = []

        for ending in options:
            prompt = f"{context} {ending}"
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=max_length).to(dev)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = inputs.input_ids[:, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            score = selected.sum().item()
            logprobs.append(score)

        pred = int(torch.tensor(logprobs).argmax().item())
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total
    print(f"StoryCloze Accuracy: {acc:.4f}")
    return acc


def eval_boolq(model, tokenizer, dataset, dev, max_length=512):
    print('Evaluating on BoolQ ...')

    model.eval()
    model.to(dev)

    correct = 0
    total = 0

    for example in tqdm(dataset):
        question = example['question']
        passage = example['passage']
        label = example['label']  # 0=False, 1=True

        options = ["True", "False"]
        logprobs = []

        for opt in options:
            # 构造 Prompt：Question + Passage + Answer
            prompt = f"Question: {question}\nPassage: {passage}\nAnswer: {opt}"

            inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=max_length).to(dev)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # [1, seq_len, vocab]
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs.input_ids[:, 1:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                score = selected.sum().item()
                logprobs.append(score)

        pred = int(torch.tensor(logprobs).argmax().item())
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total
    print(f"BoolQ Accuracy: {acc:.4f}")
    return acc
import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dataset = load_dataset(
            "parquet",  
            data_files={
                "train": "/code/datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet",  # 本地训练集路径
                "test": "/code/datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"  # 测试集
            }
        )

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    # 加载本地 Parquet 文件
    # traindata = load_dataset('parquet', data_files={'train': '/home/vipuser/code/datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet'})['train']
    # testdata = load_dataset('parquet', data_files={'test': '/home/vipuser/code/datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet'})['test']
    traindata = load_dataset('F:/code/datasets/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('F:/code/datasets/wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # 拼接所有文本进行编码
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

# def get_ptb(nsamples, seed, seqlen, model):
#     from datasets import load_dataset
#     traindata = load_dataset('parquet', data_files={'train': '/home/vipuser/code/datasets/ptb_text_only/data/train-00000-of-00001.parquet'})['train']
#     testdata = load_dataset('parquet', data_files={'test': '/home/vipuser/code/datasets/ptb_text_only/data/validation-00000-of-00001.parquet'})['test']

#     from transformers import AutoTokenizer 
#     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
#     trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
#     testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

#     import random
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))
#     return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    import random
    import torch
    from datasets import load_dataset
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    # 使用 arrow 格式加载 train 和 val 数据
    # traindata = load_from_disk('/home/vipuser/code/datasets/ptb_text_only/ptb_train')
    # valdata = load_from_disk('/home/vipuser/code/datasets/ptb_text_only/ptb_val')
    traindata = load_dataset('F:/code/datasets/ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('F:/code/datasets/ptb_text_only', 'penn_treebank', split='validation')
    
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # 拼接所有 train 文本形成大语料串
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenс = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenс


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    # traindata = load_dataset(
    #     'json',
    #     data_files={'train': '/code/datasets/c4/en/c4-train.00000-of-01024.json.gz'},
    #     split='train'
    # )
    # valdata = load_dataset(
    #     'json',
    #     data_files={'validation': '/code/datasets/c4/en/c4-validation.00000-of-00008.json.gz'},
    #     split='validation'
    # )
    # valdata = traindata.select(range(min(10000, len(traindata))))
    traindata = load_dataset(
        'F:/code/datasets/c4', 'en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'F:/code/datasets/c4', 'en', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_arc_easy(nsamples, seed, seqlen, model):
    import random
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # 加载本地 ARC-Easy Parquet 文件
#     traindata = load_dataset('parquet', data_files={
#         'train': '/home/vipuser/code/datasets/ai2_arc/ARC-Easy/train-00000-of-00001.parquet'
#     })['train']
    
#     testdata = load_dataset('parquet', data_files={
#         'test': '/home/vipuser/code/datasets/ai2_arc/ARC-Easy/validation-00000-of-00001.parquet'
#     })['test']
    traindata = load_dataset('parquet', data_files={
        'train': 'F:/code/datasets/ai2_arc/ARC-Easy/train-00000-of-00001.parquet'
    })['train']
    
    testdata = load_dataset('parquet', data_files={
        'test': 'F:/code/datasets/ai2_arc/ARC-Easy/validation-00000-of-00001.parquet'
    })['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # 格式化训练集为 prompt 形式：Q + choices + A:
    def format_arc_prompt(example):
        q = example['question']
        choices = example['choices']
        prompt = f"Q: {q}\n"
        for label, choice in zip(choices['label'], choices['text']):
            prompt += f"{label}: {choice}\n"
        prompt += "A:"
        return prompt

    # === 训练集处理（像 WikiText2）===
    train_text = "\n\n".join(format_arc_prompt(ex) for ex in traindata)
    trainenc = tokenizer(train_text, return_tensors='pt')
    input_ids = trainenc.input_ids[0]  # shape: [total_len]

    # 构造训练样本：[1, seqlen] → (input, target) 对
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[0] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[i:j].unsqueeze(0)  # [1, seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # === 测试集处理：构造结构化多选样本 ===
    testloader = []
    for ex in testdata:
        if 'answerKey' not in ex or ex['answerKey'] is None:
            continue
        answer = ex['answerKey'].strip().upper()
        labels = ex['choices']['label']
        if answer not in labels:
            continue  # 跳过异常样本
        label_idx = labels.index(answer)
        testloader.append({
            'question': ex['question'],
            'choices': ex['choices']['text'],
            'label': label_idx
        })

    return trainloader, testloader


def get_arc_challenge(nsamples, seed, seqlen, model):
    import random
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # 加载本地 ARC-Easy Parquet 文件
#     traindata = load_dataset('parquet', data_files={
#         'train': '/home/vipuser/code/datasets/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet'
#     })['train']
    
#     testdata = load_dataset('parquet', data_files={
#         'test': '/home/vipuser/code/datasets/ai2_arc/ARC-Challenge/validation-00000-of-00001.parquet'
#     })['test']
    traindata = load_dataset('parquet', data_files={
        'train': 'F:/code/datasets/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet'
    })['train']
    
    testdata = load_dataset('parquet', data_files={
        'test': 'F:/code/datasets/ai2_arc/ARC-Challenge/validation-00000-of-00001.parquet'
    })['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # 格式化训练集为 prompt 形式：Q + choices + A:
    def format_arc_prompt(example):
        q = example['question']
        choices = example['choices']
        prompt = f"Q: {q}\n"
        for label, choice in zip(choices['label'], choices['text']):
            prompt += f"{label}: {choice}\n"
        prompt += "A:"
        return prompt

    # === 训练集处理（像 WikiText2）===
    train_text = "\n\n".join(format_arc_prompt(ex) for ex in traindata)
    trainenc = tokenizer(train_text, return_tensors='pt')
    input_ids = trainenc.input_ids[0]  # shape: [total_len]

    # 构造训练样本：[1, seqlen] → (input, target) 对
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[0] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[i:j].unsqueeze(0)  # [1, seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # === 测试集处理：构造结构化多选样本 ===
    testloader = []
    for ex in testdata:
        if 'answerKey' not in ex or ex['answerKey'] is None:
            continue
        answer = ex['answerKey'].strip().upper()
        labels = ex['choices']['label']
        if answer not in labels:
            continue  # 跳过异常样本
        label_idx = labels.index(answer)
        testloader.append({
            'question': ex['question'],
            'choices': ex['choices']['text'],
            'label': label_idx
        })

    return trainloader, testloader


# def get_lambada(nsamples, seed, seqlen, model):
#     import random
#     import torch
#     from datasets import load_dataset
#     from transformers import AutoTokenizer

#     # 加载 LAMBADA 测试集
#     traindata = load_dataset('parquet', data_files={
#         'train': '/home/vipuser/code/datasets/lambada/plain_text/train-00000-of-00002.parquet'
#     })['train']
    
#     testdata = load_dataset('parquet', data_files={
#         'test': '/home/vipuser/code/datasets/lambada/plain_text/validation-00000-of-00001.parquet'
#     })['test']

#     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

#     # 拼接所有样本文本为大段文本（仅保留 context，不含 target）
#     all_context = []
#     structured_test = []

#     for ex in testdata:
#         words = ex["text"].strip().split()
#         if len(words) < 2:
#             continue
#         context = " ".join(words[:-1])
#         target = words[-1]
#         all_context.append(context)
#         structured_test.append({"context": context, "target": target})

#     # === 构造 trainloader：拼接 context 构造长文本 ===
#     long_text = "\n\n".join(all_context)
#     enc = tokenizer(long_text, return_tensors='pt')
#     input_ids = enc.input_ids[0]  # shape: [total_len]

#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         i = random.randint(0, input_ids.shape[0] - seqlen - 1)
#         j = i + seqlen
#         inp = input_ids[i:j].unsqueeze(0)  # [1, seqlen]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))

#     # === testloader: 保留结构化样本：用于 accuracy 测试 ===
#     testloader = structured_test

#     return trainloader, testloader


def get_piqa(nsamples, seed, seqlen, model):
    import random
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # 加载本地 PIQA 数据
#     traindata = load_dataset('parquet', data_files={
#         'train': '/home/vipuser/code/datasets/piqa/data/train-00000-of-00001.parquet'
#     })['train']
    
#     testdata = load_dataset('parquet', data_files={
#         'test': '/home/vipuser/code/datasets/piqa/data/validation-00000-of-00001.parquet'
#     })['test']
    traindata = load_dataset('parquet', data_files={
        'train': 'F:/code/datasets/piqa/data/train-00000-of-00001.parquet'
    })['train']
    
    testdata = load_dataset('parquet', data_files={
        'test': 'F:/code/datasets/piqa/data/validation-00000-of-00001.parquet'
    })['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # === 训练集拼接文本 ===
    def format_train_text(example):
        return f"Q: {example['goal']}\nA: {example['sol1']}"

    train_text = "\n\n".join(format_train_text(ex) for ex in traindata)
    trainenc = tokenizer(train_text, return_tensors='pt')
    input_ids = trainenc.input_ids[0]

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[0] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[i:j].unsqueeze(0)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # === 构造结构化测试数据 ===
    testloader = []
    for ex in testdata:
        try:
            label = int(ex['label'])
            if label not in [0, 1]:
                continue  # ❗️跳过 label=-1 或异常样本
            testloader.append({
                'goal': ex['goal'],
                'choices': [ex['sol1'], ex['sol2']],
                'label': label
            })
        except:
            continue

    return trainloader, testloader




def get_SC(nsamples, seed, seqlen, model):
    import random
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # 加载本地 Parquet 文件（验证集）
    # data = load_dataset('parquet', data_files={'test': '/home/vipuser/code/datasets/story_cloze/data/validation-00000-of-00001.parquet'})['test']
    data = load_dataset('parquet', data_files={'test': 'F:/code/datasets/story_cloze/data/validation-00000-of-00001.parquet'})['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # === 训练数据：拼接前四句构造长文本 ===
    def format_context(example):
        return f"{example['input_sentence_1']} {example['input_sentence_2']} {example['input_sentence_3']} {example['input_sentence_4']}"

    full_text = "\n\n".join(format_context(ex) for ex in data)
    enc = tokenizer(full_text, return_tensors='pt')
    input_ids = enc.input_ids[0]

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[0] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[i:j].unsqueeze(0)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # === 测试数据结构化：story + 两个选项 + 正确结尾索引 ===
    testloader = []
    for ex in data:
        try:
            label = int(ex['answer_right_ending']) - 1
            assert label in [0, 1]
        except:
            continue
        context = format_context(ex)
        testloader.append({
            'context': context,
            'choices': [ex['sentence_quiz1'], ex['sentence_quiz2']],
            'label': label
        })

    return trainloader, testloader



def get_boolq(nsamples, seed, seqlen, model):
    import random
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # === 加载 BoolQ Parquet 文件 ===
    # traindata = load_dataset('parquet', data_files={
    #     'train': '/home/vipuser/code/datasets/boolq/data/train-00000-of-00001.parquet'
    # })['train']
    # testdata = load_dataset('parquet', data_files={
    #     'test': '/home/vipuser/code/datasets/boolq/data/validation-00000-of-00001.parquet'
    # })['test']
    traindata = load_dataset('parquet', data_files={
        'train': 'F:/code/datasets/boolq/data/train-00000-of-00001.parquet'
    })['train']
    testdata = load_dataset('parquet', data_files={
        'test': 'F:/code/datasets/boolq/data/validation-00000-of-00001.parquet'
    })['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    
    def format_boolq_prompt(example):
        q = example['question']
        passage = example['passage']
        prompt = f"Question: {q}\nPassage: {passage}\nAnswer:"
        return prompt

    train_text = "\n\n".join(format_boolq_prompt(ex) for ex in traindata)
    trainenc = tokenizer(train_text, return_tensors='pt')
    input_ids = trainenc.input_ids[0]  # [total_len]

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[0] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[i:j].unsqueeze(0)  # [1, seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100  # 仅预测最后一个 token
        trainloader.append((inp, tar))

    testloader = []
    for ex in testdata:
        if 'answer' not in ex:
            continue
        label = 1 if ex['answer'] else 0  # True=1, False=0
        testloader.append({
            'question': ex['question'],
            'passage': ex['passage'],
            'label': label
        })

    return trainloader, testloader



def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)

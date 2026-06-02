import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
def get_gsm8k(nsamples, seed, seqlen, model):
    # Local parquet
    traindata = load_dataset('parquet', data_files={
        'train': '/code/datasets/gsm8k/train-00000-of-00001.parquet'
    })['train']

    testdata = load_dataset('parquet', data_files={
        'test': '/code/datasets/gsm8k/test-00000-of-00001.parquet'
    })['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== Build training text =====
    def format_train(example):
        q = example["question"]
        a = example["answer"]
        return f"Question: {q}\nAnswer: {a}"

    train_text = "\n\n".join(format_train(ex) for ex in traindata)

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

    # ===== Test set =====

    testloader = []

    for ex in testdata:

        question = ex["question"]

        answer = ex["answer"]

        # GSM8K format: #### 123
        if "####" in answer:
            answer = answer.split("####")[-1].strip()

        testloader.append({
            "question": question,
            "answer": answer
        })

    return trainloader, testloader  





def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    # Load local Parquet file (wikitext2)
    traindata = load_dataset('parquet', data_files={'train': '/code/datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet'})['train']
    testdata = load_dataset('parquet', data_files={'test': '/code/datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet'})['test']
    # traindata = load_dataset('F:/code/datasets/wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('F:/code/datasets/wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # Concatenate all text for encoding
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

    # Use arrow format to load train and val data
    traindata = load_from_disk('/code/datasets/ptb_text_only/ptb_train')
    valdata = load_from_disk('/code/datasets/ptb_text_only/ptb_val')
    # traindata = load_dataset('F:/code/datasets/ptb_text_only', 'penn_treebank', split='train')
    # valdata = load_dataset('F:/code/datasets/ptb_text_only', 'penn_treebank', split='validation')
    
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # Concatenate all train text to form large corpus string
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
    #traindata = load_dataset(
    #    '/code/datasets/c4', 'en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    #)
    #valdata = load_dataset(
    #    '/code/datasets/c4', 'en', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    #)
    traindata = load_dataset(
        'json',
        data_files={'train': '/code/datasets/c4/en/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    valdata = load_dataset(
        'json',
        data_files={'validation': '/code/datasets/c4/en/c4-validation.00000-of-00008.json.gz'},
        split='validation'
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
            idx = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[idx]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if tmp.input_ids.shape[1] == seqlen:
            i = 0
        else:
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

    # Load local ARC-Easy Parquet file
    traindata = load_dataset('parquet', data_files={
        'train': '/code/datasets/ai2_arc/ARC-Easy/train-00000-of-00001.parquet'
    })['train']
    
    testdata = load_dataset('parquet', data_files={
        'test': '/code/datasets/ai2_arc/ARC-Easy/validation-00000-of-00001.parquet'
    })['test']
    #traindata = load_dataset('parquet', data_files={
    #    'train': 'F:/code/datasets/ai2_arc/ARC-Easy/train-00000-of-00001.parquet'
    #})['train']
    
    #testdata = load_dataset('parquet', data_files={
    #    'test': 'F:/code/datasets/ai2_arc/ARC-Easy/validation-00000-of-00001.parquet'
    #})['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # Format training set as prompt: Q + choices + A:
    def format_arc_prompt(example):
        q = example['question']
        choices = example['choices']
        prompt = f"Q: {q}\n"
        for label, choice in zip(choices['label'], choices['text']):
            prompt += f"{label}: {choice}\n"
        prompt += "A:"
        return prompt

    # === Training set processing (like WikiText2) ===
    train_text = "\n\n".join(format_arc_prompt(ex) for ex in traindata)
    trainenc = tokenizer(train_text, return_tensors='pt')
    input_ids = trainenc.input_ids[0]  # shape: [total_len]

    # Build training samples: [1, seqlen] → (input, target) pairs
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[0] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[i:j].unsqueeze(0)  # [1, seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # === Test set processing: build structured multi-choice samples ===
    testloader = []
    for ex in testdata:
        if 'answerKey' not in ex or ex['answerKey'] is None:
            continue
        answer = ex['answerKey'].strip().upper()
        labels = ex['choices']['label']
        if answer not in labels:
            continue  # Skip anomalous samples
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

    # Load local ARC-Challenge Parquet file
    traindata = load_dataset('parquet', data_files={
        'train': '/code/datasets/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet'
    })['train']
    
    testdata = load_dataset('parquet', data_files={
        'test': '/code/datasets/ai2_arc/ARC-Challenge/validation-00000-of-00001.parquet'
    })['test']
    #traindata = load_dataset('parquet', data_files={
    #    'train': 'F:/code/datasets/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet'
    #})['train']
    
    #testdata = load_dataset('parquet', data_files={
    #    'test': 'F:/code/datasets/ai2_arc/ARC-Challenge/validation-00000-of-00001.parquet'
    #})['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # Format training set as prompt: Q + choices + A:
    def format_arc_prompt(example):
        q = example['question']
        choices = example['choices']
        prompt = f"Q: {q}\n"
        for label, choice in zip(choices['label'], choices['text']):
            prompt += f"{label}: {choice}\n"
        prompt += "A:"
        return prompt

    # === Training set processing (like WikiText2) ===
    train_text = "\n\n".join(format_arc_prompt(ex) for ex in traindata)
    trainenc = tokenizer(train_text, return_tensors='pt')
    input_ids = trainenc.input_ids[0]  # shape: [total_len]

    # Build training samples: [1, seqlen] → (input, target) pairs
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[0] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[i:j].unsqueeze(0)  # [1, seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # === Test set processing: build structured multi-choice samples ===
    testloader = []
    for ex in testdata:
        if 'answerKey' not in ex or ex['answerKey'] is None:
            continue
        answer = ex['answerKey'].strip().upper()
        labels = ex['choices']['label']
        if answer not in labels:
            continue  # Skip anomalous samples
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

#     # Load LAMBADA test set
#     traindata = load_dataset('parquet', data_files={
#         'train': '/home/vipuser/code/datasets/lambada/plain_text/train-00000-of-00002.parquet'
#     })['train']
    
#     testdata = load_dataset('parquet', data_files={
#         'test': '/home/vipuser/code/datasets/lambada/plain_text/validation-00000-of-00001.parquet'
#     })['test']

#     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

#     # Join all sample text into large text (context only, exclude target)
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

#     # === Build trainloader: concatenate context to form long text ===
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

#     # === testloader: keep structured samples for accuracy testing ===
#     testloader = structured_test

#     return trainloader, testloader


def get_piqa(nsamples, seed, seqlen, model):
    import random
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load local PIQA data
    traindata = load_dataset('parquet', data_files={
        'train': '/code/datasets/piqa/data/train-00000-of-00001.parquet'
    })['train']
    
    testdata = load_dataset('parquet', data_files={
        'test': '/code/datasets/piqa/data/validation-00000-of-00001.parquet'
    })['test']
    #traindata = load_dataset('parquet', data_files={
    #    'train': 'F:/code/datasets/piqa/data/train-00000-of-00001.parquet'
    #})['train']
    
    #testdata = load_dataset('parquet', data_files={
    #    'test': 'F:/code/datasets/piqa/data/validation-00000-of-00001.parquet'
    #})['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # === Concatenate training set text ===
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

    # === Build structured test data ===
    testloader = []
    for ex in testdata:
        try:
            label = int(ex['label'])
            if label not in [0, 1]:
                continue  # Skip label=-1 or anomalous samples
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

    # Load local Parquet file (validation set)
    data = load_dataset('parquet', data_files={'test': '/code/datasets/story_cloze/data/validation-00000-of-00001.parquet'})['test']
    #data = load_dataset('parquet', data_files={'test': 'F:/code/datasets/story_cloze/data/validation-00000-of-00001.parquet'})['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # === Training data: concatenate first four sentences to build long text ===
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

    # === Test data structuring: story + two options + correct ending index ===
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

    # === Load BoolQ Parquet file ===
    traindata = load_dataset('parquet', data_files={
        'train': '/code/datasets/boolq/data/train-00000-of-00001.parquet'
    })['train']
    testdata = load_dataset('parquet', data_files={
        'test': '/code/datasets/boolq/data/validation-00000-of-00001.parquet'
    })['test']
    #traindata = load_dataset('parquet', data_files={
    #    'train': 'F:/code/datasets/boolq/data/train-00000-of-00001.parquet'
    #})['train']
    #testdata = load_dataset('parquet', data_files={
    #    'test': 'F:/code/datasets/boolq/data/validation-00000-of-00001.parquet'
    #})['test']

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
        tar[:, :-1] = -100  # Predict only the last token
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
    if 'gsm8k' in name:
        return get_gsm8k(nsamples, seed, seqlen, model)

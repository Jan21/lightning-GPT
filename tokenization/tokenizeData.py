from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import tiktoken
import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset # huggingface datasets
import pickle

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
trainfile = '/home/jan/projects/LMStartup/data/cc_czech/nov_dec_50M/train.txt'
testfile = "/home/jan/projects/LMStartup/data/cc_czech/nov_dec_50M/test.txt"
use_tiktoken = False
tokenizerfile = "temp/cc/nov_dec_2022/tokenizerWL.json"
if not use_tiktoken:
    tokenizer = Tokenizer.from_file(tokenizerfile)
    print(f'vocab size: {tokenizer.get_vocab_size()}')
else:
    tokenizer = tiktoken.get_encoding("gpt2")
num_proc = 8
# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("text", data_files={
    "train": trainfile,
    "test": testfile})
# owt by default only contains the 'train' split, so create a test split
print('b')
dataset['val'] = dataset.pop('test') # rename the test split to val

def process(example):
    if use_tiktoken:
        enc =tokenizer.encode(example['text']+"<|endoftext|>",allowed_special={'<|endoftext|>'}) # encode_ordinary ignores any special tokens
        out = {'ids': enc, 'len': len(enc)}
    else:
        enc = tokenizer.encode(example['text']+"<|endoftext|>") 
        out = {'ids': enc.ids, 'len': len(enc.ids)}
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    return out

# tokenize the dataset
tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

with open('temp/cc/nov_dec_2022/tokenized.pkl', 'wb') as f:
    pickle.dump(tokenized, f)
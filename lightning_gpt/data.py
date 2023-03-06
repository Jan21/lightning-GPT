import math
import random
from typing import Tuple, Union

import torch
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import Dataset
import numpy as np
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)



class CharDataset(Dataset):
    def __init__(self, data: str, block_size: int):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self) -> int:
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def to_tokens(self, message: str, device: Union[str, torch.device]) -> torch.Tensor:
        return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[None, ...].to(device)

    def from_tokens(self, tokens: torch.Tensor) -> str:
        return "".join([self.itos[int(i)] for i in tokens])

class cc_czech_Dataset(Dataset):
    def __init__(self, data: str, block_size: int, tokenizer_str: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_str)
        vocab_size = self.tokenizer.get_vocab_size()
        if vocab_size > 60000:
            dtype=np.uint32
        else:
            dtype=np.uint16
        data = np.memmap(data, dtype=dtype, mode='r')
        data_size = len(data)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self) -> int:
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i : i + self.block_size + 1].astype(np.int64)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def to_tokens(self, message: str, device: Union[str, torch.device]) -> torch.Tensor:
        return torch.tensor([self.tokenizer.token_to_id(s) for s in message], dtype=torch.long)[None, ...].to(device)

    def from_tokens(self, tokens: torch.Tensor) -> str:
        return "".join([self.tokenizer.id_to_token(int(i)) for i in tokens])

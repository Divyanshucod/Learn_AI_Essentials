from tokenizer import BPETokenizer
from tokenizer import BPETokenizer
from model import GPTSimulation

import torch
import os
dataset = open("input.txt").read()

vocab_size = 300
context_length = 128
batch_size = 32
n_embed = 256
n_head = 8
max_iter = 10000
eval_iters = 100
eval_interval = 500
learning_rate = 5e-5
dropout = 0.1
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = BPETokenizer(vocab_size)

if os.path.exists("artifacts/tokenizer.pt"):
    tokenizer.load("artifacts/tokenizer.pt")

special_tokens = {
    "<|user|>": 300,
    "<|assistant|>": 301,
    "<|end|>": 302
}
for key, val in special_tokens.items():
    tokenizer.itos[val] = key


tokens = tokenizer.encode(text)

print(tokens)
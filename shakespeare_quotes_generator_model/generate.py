import torch

from tokenizer import BPETokenizer
from model import GPTSimulation
vocab_size = 300
context_length = 128
batch_size = 32
n_embed = 256
n_head = 8
max_iter = 10000
eval_iters = 100
eval_interval = 500
learning_rate = 3e-4
dropout = 0.1
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = BPETokenizer(vocab_size)
tokenizer.load("artifacts/tokenizer.pt")

checkpoint = torch.load(
    "artifacts/model.pt",
    map_location="cpu"
)

model = GPTSimulation(
    vocab_size=checkpoint["vocab_size"],
    context_length=checkpoint["context_length"],
    n_embed=checkpoint["n_embed"],
    n_head=checkpoint["n_head"],
    dropout=checkpoint["dropout"]
)
model = model.to(device)
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

## Prompt
prompt = "HAMLET:"

tokens = tokenizer.encode(prompt)

input_tokens = torch.tensor(
    [tokens],
    dtype=torch.long
)
input_tokens = input_tokens.to(device)
output = model.generate(
    input_tokens,
    max_new_tokens=2000
)

text = tokenizer.decode(
    output[0].tolist()
)
print(text)
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
learning_rate = 3e-4
dropout = 0.1
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = BPETokenizer(vocab_size)

if os.path.exists("artifacts/tokenizer.pt"):
    tokenizer.load("artifacts/tokenizer.pt")
else:
    tokenizer.train(dataset)
    tokenizer.save("artifacts/tokenizer.pt")

tokenization = torch.tensor(
    tokenizer.encode(dataset),
    dtype=torch.long
)

## train and val split
n = int(0.9 * len(tokenization))

train_dataset = tokenization[:n]
val_dataset = tokenization[n:]

model = GPTSimulation(
    vocab_size=tokenizer.vocab_size,
    context_length=context_length,
    n_embed=n_embed,
    n_head=n_head,
    dropout=dropout
).to(device)

def get_batch(split):
    splitToProcess = train_dataset if split == 'train' else val_dataset
    startingPointers = torch.randint(0, len(splitToProcess)-context_length-1 , (batch_size,))
    x = torch.stack([splitToProcess[i:i+context_length] for i in startingPointers])
    y = torch.stack([splitToProcess[i+1:i+context_length+1] for i in startingPointers])
    return x,y

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


@torch.no_grad()
def estimate_loss():

    out = {}

    model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            X = X.to(device)
            Y = Y.to(device)

            _, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

for iteration in range(max_iter):

    if iteration % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"step {iteration}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    X, Y = get_batch("train")

    X = X.to(device)
    Y = Y.to(device)

    _, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

torch.save({
    "model_state_dict": model.state_dict(),

    "vocab_size": tokenizer.vocab_size,
    "context_length": 6,
    "n_embed": 32,
    "n_head": 4,
    "dropout": 0.2,

}, "artifacts/model.pt")
from tokenizer import BPETokenizer
from tokenizer import BPETokenizer
from model import GPTSimulation

import torch
import os
dataset = open("./post_training/new_formatted_corpus.txt").read()


base_vocab_size = 300
vocab_size = 303
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
    tokenizer.stoi[key] = val

## train and val split
n = int(0.9 * len(dataset))

train_data = dataset[:n]
val_data = dataset[n:]

tokenizationTraining = torch.tensor(
    tokenizer.encode(train_data),
    dtype=torch.long
)

tokenizationValidation = torch.tensor(
    tokenizer.encode(val_data),
    dtype=torch.long
)
train_dataset = tokenizationTraining
val_dataset = tokenizationValidation


# checkpoint = torch.load(
#     "artifacts/model.pt",
#     map_location="cpu"
# )

# old_state = checkpoint["model_state_dict"]

# model = GPTSimulation(
#     vocab_size=303,
#     context_length=checkpoint["context_length"],
#     n_embed=checkpoint["n_embed"],
#     n_head=checkpoint["n_head"],
#     dropout=checkpoint["dropout"]
# )

# new_state = model.state_dict()

# # Copy all parameters whose shapes didn't change
# for key in new_state:

#     if key not in old_state:
#         continue

#     if key in [
#         "Embedding.weight",
#         "ll_head.weight",
#         "ll_head.bias"
#     ]:
#         continue

#     new_state[key] = old_state[key]

# new_state["Embedding.weight"][:300] = old_state["Embedding.weight"]
# new_state["ll_head.weight"][:300] = old_state["ll_head.weight"]
# new_state["ll_head.bias"][:300] = old_state["ll_head.bias"]

# model.load_state_dict(new_state)

# model = model.to(device)

# def get_batch(split):
#     splitToProcess = train_dataset if split == 'train' else val_dataset
#     startingPointers = torch.randint(0, len(splitToProcess)-context_length-1 , (batch_size,))
#     x = torch.stack([splitToProcess[i:i+context_length] for i in startingPointers])
#     y = torch.stack([splitToProcess[i+1:i+context_length+1] for i in startingPointers])
#     return x,y

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=learning_rate
# )


# @torch.no_grad()
# def estimate_loss():

#     out = {}

#     model.eval()

#     for split in ["train", "val"]:

#         losses = torch.zeros(eval_iters)

#         for k in range(eval_iters):

#             X, Y = get_batch(split)

#             X = X.to(device)
#             Y = Y.to(device)

#             _, loss = model(X, Y)

#             losses[k] = loss.item()

#         out[split] = losses.mean()

#     model.train()

#     return out

# for iteration in range(max_iter):

#     if iteration % eval_interval == 0:

#         losses = estimate_loss()

#         print(
#             f"step {iteration}: "
#             f"train loss {losses['train']:.4f}, "
#             f"val loss {losses['val']:.4f}"
#         )

#     X, Y = get_batch("train")

#     X = X.to(device)
#     Y = Y.to(device)

#     _, loss = model(X, Y)

#     optimizer.zero_grad(set_to_none=True)

#     loss.backward()

#     optimizer.step()
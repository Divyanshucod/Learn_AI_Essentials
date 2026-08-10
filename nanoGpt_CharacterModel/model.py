## implementing biagram model
import torch.nn as nn
from torch.nn import functional as F
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 300
learning_rate = 1e-2
max_iters = 3000
batch_size = 32
block_size = 8

torch.manual_seed(1337)

## Reading the dataset
data = open('input.txt','r',encoding='utf-8').read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

## First thing (Think how will you create tokens from the dataset) (as this one is a character level model so we need to represent them in number)
## building encoder (Take string and give out integer) , decoder (Take integer and give out string)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
encoder = lambda s : [stoi[c] for c in s]
decoder = lambda i : ''.join([itos[c] for c in i])

encoData = torch.tensor(encoder(data), dtype=torch.long)

## splitting dataset for train and validation
n = int(0.9*len(encoData))

train_data = encoData[:n]
val_data = encoData[n:]

## Processing data in batch

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    ## --- for implementing transformer ---
    x , y = x.to(device) , y.to(device)
    return x,y

xb , yb = get_batch('train')

## implementing biagram model
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BiagramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, target=None):
        logits = self.token_embedding_table(idx)
        ## loss
        if target == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # Focus on last time steps
            # softmwax
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BiagramLanguageModel(vocab_size)
m = model.to(device)
out, loss = m.forward(xb, yb)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# created a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(max_iters):
    ## --- loss estimation 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    xb, yb = get_batch('train')

    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

print(decoder(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=300)[0].tolist()))

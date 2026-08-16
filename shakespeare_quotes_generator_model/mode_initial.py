
## Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter

## read the dataset
dataset = open('input.txt', 'r').read()
data = sorted(list(set(dataset)))

## Global Variables
vocab_size = 300
initialVocab = len(data)
context_length = 6
batch_size = 4
n_embed = 32
max_iter = 3000
eval_iters = 200
eval_interval = 500
learning_rate = 1e-3
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_head = 4

## mapping (char to int) and (int, char)
stoi = {s:i for i,s in enumerate(data)}
itos = {i:s for i,s in enumerate(data)}
encoder = lambda l : [stoi[ch] for ch in l]
def decoder(tokens):
    return ''.join(itos[token] for token in tokens)

## 20% of training data set
n3 = int(0.2 * len(dataset))
text = dataset[:n3]
## creating tokens
tokens = encoder(text)
extravocabs = vocab_size - initialVocab 

## Pair creation
def get_pair(tokens):

    def create_pairs(tokens):
        counter = Counter()
        for pair in zip(tokens[:], tokens[1:]):
            counter[pair]+=1
        return counter
    counter = create_pairs(tokens)
    max_pair = max(counter, key=counter.get)
    return max_pair

## creating vocabulary using BPE
merges = []
for j in range(extravocabs):
    pair = get_pair(tokens)
    merges.append(pair)
    currentTokenId = j + initialVocab
    i = 0
    new_tokens = []
    while i < len(tokens):
        if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
            st = itos[tokens[i]]+itos[tokens[i+1]]
            stoi[st] = currentTokenId
            itos[currentTokenId] = st
            new_tokens.append(currentTokenId)
            i+=2
        else:
            new_tokens.append(tokens[i])
            i+=1
    tokens = new_tokens

## tokenize the whole dataset and split in training and validation
def encode(text):
    tokens = [stoi[ch] for ch in text]

    for pair in merges:
        new_tokens = []
        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                new_tokens.append(
                    stoi[itos[tokens[i]] + itos[tokens[i + 1]]]
                )
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return tokens

tokenization = torch.tensor(encode(dataset), dtype=torch.long)
n1 = int(0.9 * len(tokenization))
train_dataset = tokenization[:n1]
val_dataset = tokenization[n1:]

## Next Input and output data split with the batch
torch.manual_seed(1337)
def get_batch(split):
    splitToProcess = train_dataset if split == 'train' else val_dataset
    startingPointers = torch.randint(0, len(splitToProcess)-context_length , (batch_size,))
    x = torch.stack([splitToProcess[i:i+context_length] for i in startingPointers])
    y = torch.stack([splitToProcess[i+1:i+context_length+1] for i in startingPointers])
    return x,y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = (q @ k.transpose(-2,-1)) * C**-0.5
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ self.value(x)
        return out

class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4* n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

## Creating Transformer block for communication followed by computation
class Block(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()
        self.multiHead = MultiHead(n_head, n_embed//n_head)
        self.feedForward = FeedForward()
        self.layerNorm1 = nn.LayerNorm(n_embed)
        self.layerNorm2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        ## Residual connection
        x = x + self.multiHead(self.layerNorm1(x))
        x = x + self.feedForward(self.layerNorm2(x))
        return x

class BiagramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, n_embed)
        self.ll_head = nn.Linear(n_embed, vocab_size)
        self.positionalEmbedding = nn.Embedding(context_length, n_embed)
        self.transBlock = nn.Sequential(Block(n_head, n_embed),Block(n_head, n_embed), Block(n_head, n_embed), nn.LayerNorm(n_embed))
    def forward(self, x, out=None):
        B, T = x.shape
        position = self.positionalEmbedding(torch.arange(T))
        token_emb = self.Embedding(x)
        x = token_emb + position
        x = self.transBlock(x)
        logits = self.ll_head(x) ## (B,T,vocab_size)
        if out == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            out = out.view(B*T)
            loss = F.cross_entropy(logits,out)
        return logits , loss
    def generate(self, inputToken, max_n_tokens):
        for _ in range(max_n_tokens):
            cropped_inputToken = inputToken[:, -context_length:]
            logits , loss = self(cropped_inputToken)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            pred = torch.multinomial(probs, num_samples = 1)
            inputToken = torch.cat((inputToken, pred), dim=1)
        return inputToken

C = BiagramModel()

## Optimizer
optimizer = torch.optim.AdamW(C.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    C.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = C(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    C.train()
    return out

for iter in range(max_iter):
    ## --- loss estimation 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    xb, yb = get_batch('train')

    logits, loss = C(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

print(decoder(C.generate(torch.zeros((1,1), dtype=torch.long), 200)[0].tolist()))
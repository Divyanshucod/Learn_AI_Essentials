import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, n_embed, head_size, dropout):
        super().__init__()

        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = (q @ k.transpose(-2, -1)) * C**-0.5

        tril = torch.tril(
            torch.ones(T, T, device=x.device)
        )

        wei = wei.masked_fill(
            tril == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ self.value(x)


class MultiHead(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()

        head_size = n_embed // n_head

        self.heads = nn.ModuleList([
            Head(n_embed, head_size, dropout)
            for _ in range(n_head)
        ])

        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()

        self.multiHead = MultiHead(
            n_embed,
            n_head,
            dropout
        )

        self.feedForward = FeedForward(
            n_embed,
            dropout
        )

        self.layerNorm1 = nn.LayerNorm(n_embed)
        self.layerNorm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.multiHead(
            self.layerNorm1(x)
        )

        x = x + self.feedForward(
            self.layerNorm2(x)
        )

        return x
class GPTSimulation(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        n_embed,
        n_head,
        dropout
    ):
        super().__init__()

        self.context_length = context_length

        self.Embedding = nn.Embedding(
            vocab_size,
            n_embed
        )

        self.positionalEmbedding = nn.Embedding(
            context_length,
            n_embed
        )

        self.transBlock = nn.Sequential(
            Block(n_embed, n_head, dropout),
            Block(n_embed, n_head, dropout),
            Block(n_embed, n_head, dropout),
            nn.LayerNorm(n_embed)
        )

        self.ll_head = nn.Linear(
            n_embed,
            vocab_size
        )

    def forward(self, x, targets=None):

        B, T = x.shape

        token_emb = self.Embedding(x)

        position = self.positionalEmbedding(
            torch.arange(
                T,
                device=x.device
            )
        )

        x = token_emb + position

        x = self.transBlock(x)

        logits = self.ll_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_tokens, max_new_tokens):

        for _ in range(max_new_tokens):

            cropped_input = input_tokens[
                :, -self.context_length:
            ]

            logits, _ = self(cropped_input)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            input_tokens = torch.cat(
                (input_tokens, next_token),
                dim=1
            )

        return input_tokens
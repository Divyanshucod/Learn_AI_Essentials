import torch
from collections import Counter


class BPETokenizer:

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.stoi = {}
        self.itos = {}
        self.merges = {}

    def get_pair(self, tokens):
        counter = Counter(zip(tokens, tokens[1:]))
        return max(counter, key=counter.get)

    def train(self, text):

        data = sorted(list(set(text)))

        self.stoi = {s: i for i, s in enumerate(data)}
        self.itos = {i: s for i, s in enumerate(data)}

        tokens = [self.stoi[ch] for ch in text]

        initial_vocab = len(data)
        extra_vocab = self.vocab_size - initial_vocab

        for j in range(extra_vocab):

            pair = self.get_pair(tokens)

            new_token_id = initial_vocab + j

            # Store the merges
            self.merges[pair] = new_token_id

            # Create new token
            new_token = (
                self.itos[pair[0]]
                + self.itos[pair[1]]
            )

            self.stoi[new_token] = new_token_id
            self.itos[new_token_id] = new_token

            # Apply merge to training tokens
            new_tokens = []

            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == pair
                ):
                    new_tokens.append(new_token_id)
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

    def encode(self, text):

        tokens = [self.stoi[ch] for ch in text]

        for pair, new_token_id in self.merges.items():

            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == pair
                ):
                    new_tokens.append(new_token_id)
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def decode(self, tokens):
        return ''.join(self.itos[token] for token in tokens)

    def save(self, path):
        torch.save({
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "merges": self.merges,
        }, path)

    def load(self, path):
        data = torch.load(path)

        self.vocab_size = data["vocab_size"]
        self.stoi = data["stoi"]
        self.itos = data["itos"]
        self.merges = data["merges"]
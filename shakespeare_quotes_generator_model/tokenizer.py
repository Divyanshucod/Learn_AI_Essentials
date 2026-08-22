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
    def apply_merges(self, tokens):
        new_tokens = []

        for pair, new_token_id in self.merges.items():

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
            new_tokens = []

        return tokens

    def encode(self, text):

        special_tokens = {
            "<|user|>": 300,
            "<|assistant|>": 301,
            "<|end|>": 302
        }

        final_tokens = []

        i = 0
        normal_text = ""

        while i < len(text):

            matched = False

            for token, token_id in special_tokens.items():

                if text.startswith(token, i):

                    # Encode everything accumulated before special token
                    if normal_text:
                        tokens = [self.stoi[ch] for ch in normal_text]

                        # YOUR EXISTING BPE MERGE CODE
                        tokens = self.apply_merges(tokens)

                        final_tokens.extend(tokens)
                        normal_text = ""

                    # Add special token directly
                    final_tokens.append(token_id)

                    i += len(token)
                    matched = True
                    break

            if matched:
                continue

            normal_text += text[i]
            i += 1

        # Encode remaining normal text
        if normal_text:
            tokens = [self.stoi[ch] for ch in normal_text]

            # YOUR EXISTING BPE MERGE CODE
            tokens = self.apply_merges(tokens)

            final_tokens.extend(tokens)

        return final_tokens

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
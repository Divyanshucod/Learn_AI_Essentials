import torch

checkpoint = torch.load(
    "artifacts/model.pt",
    map_location="cpu"
)

fixed_checkpoint = {
    "model_state_dict": checkpoint["model_state_dict"],

    "vocab_size": 300,
    "context_length": 128,
    "n_embed": 256,
    "n_head": 8,
    "dropout": 0.1,
}

torch.save(
    fixed_checkpoint,
    "artifacts/model.pt"
)

print("Checkpoint metadata fixed.")
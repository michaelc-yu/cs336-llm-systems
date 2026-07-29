import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"initial dtype: {x.dtype}")

        x = self.fc1(x)
        print(f"dtype after first FC layer: {x.dtype}")

        x = self.relu(x)
        print(f"dtype after relu: {x.dtype}")

        x = self.ln(x)
        print(f"dtype after layer norm: {x.dtype}")

        x = self.fc2(x)
        print(f"dtype of logits: {x.dtype}")

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ToyModel(64, 64).to(device)
input = torch.rand((8, 64), dtype=torch.float32).to(device)
y = torch.rand((8, 64), dtype=torch.float32).to(device)


with torch.autocast(device_type=device, dtype=torch.float16):
    out = model(input)
    print(f"output dtype: {out.dtype}")

    loss = (out - y).mean()
    print(f"loss dtype: {loss.dtype}")

    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name}, param dtype: {p.dtype}, grad dtype: {p.grad.dtype}")
        else:
            print(f"{name}: param dtype: {p.dtype}")


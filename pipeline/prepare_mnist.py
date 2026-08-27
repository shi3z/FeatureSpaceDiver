"""MNIST pipeline for Feature Space Diver.

State A: 3D-bottleneck autoencoder (unsupervised).
State B: the same network fine-tuned semi-supervised (reconstruction + classification
         loss on the 3D bottleneck), which pulls the classes apart — the same idea as
         the original Feature Space Diver demo.

Outputs (into ../web/data/mnist/):
  posA.bin / posB.bin   float32 xyz per test sample (normalized, RMS radius 1)
  labels.bin            uint8 label per sample
  atlas.png             sprite atlas of the digits
  encoderA.json / encoderB.json  MLP weights so the browser can embed hand-drawn digits
  meta.json
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from common import (build_atlas, export_mlp_json, normalize_coords,
                    save_labels, save_positions, write_meta)

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(__file__), "..", "web", "data", "mnist")
os.makedirs(OUT, exist_ok=True)

AE_EPOCHS = 10
FT_EPOCHS = 10
BATCH = 256
N_VIS = 8000  # points shown in the browser


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 512)
        self.enc2 = nn.Linear(512, 128)
        self.enc3 = nn.Linear(128, 3)
        self.dec1 = nn.Linear(3, 128)
        self.dec2 = nn.Linear(128, 512)
        self.dec3 = nn.Linear(512, 784)
        self.cls = nn.Linear(3, 10)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.enc3(h)

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.dec3(h))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), self.cls(z), z


def get_loaders():
    tf = transforms.ToTensor()
    root = os.path.join(os.path.dirname(__file__), "torch_data")
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    return (
        torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0),
        test,
    )


def train(model, loader, epochs, supervised, tag):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, y in loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)
            recon, logits, _ = model(x)
            loss = F.mse_loss(recon, x)
            if supervised:
                loss = loss + 1.0 * F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"[{tag}] epoch {ep+1}/{epochs} loss {tot/n:.4f}")


@torch.no_grad()
def embed(model, images):
    model.eval()
    zs = []
    for i in range(0, len(images), 1024):
        x = images[i : i + 1024].to(DEVICE)
        zs.append(model.encode(x).cpu().numpy())
    return np.concatenate(zs)


def export_encoder(model, path, center, scale):
    layers = [
        (model.enc1.weight.detach().cpu().numpy(), model.enc1.bias.detach().cpu().numpy(), "relu"),
        (model.enc2.weight.detach().cpu().numpy(), model.enc2.bias.detach().cpu().numpy(), "relu"),
        (model.enc3.weight.detach().cpu().numpy(), model.enc3.bias.detach().cpu().numpy(), "linear"),
    ]
    export_mlp_json(path, layers, center, scale, input_norm={"scale": 1.0})


def main():
    loader, test = get_loaders()

    print(f"device: {DEVICE}")
    model = Net().to(DEVICE)
    train(model, loader, AE_EPOCHS, supervised=False, tag="autoencoder")
    state_a = {k: v.clone() for k, v in model.state_dict().items()}

    # fine-tune from the autoencoder weights (semi-supervised)
    train(model, loader, FT_EPOCHS, supervised=True, tag="fine-tune")

    # visualize a subset of the test set
    idx = np.random.permutation(len(test))[:N_VIS]
    images = torch.stack([test[i][0] for i in idx]).view(len(idx), -1)
    labels = np.array([test[i][1] for i in idx], dtype=np.uint8)

    model_b = model
    model_a = Net().to(DEVICE)
    model_a.load_state_dict(state_a)

    za = embed(model_a, images)
    zb = embed(model_b, images)
    za_n, ca, sa = normalize_coords(za)
    zb_n, cb, sb = normalize_coords(zb)

    save_positions(os.path.join(OUT, "posA.bin"), za_n)
    save_positions(os.path.join(OUT, "posB.bin"), zb_n)
    save_labels(os.path.join(OUT, "labels.bin"), labels)

    sprites = (images.numpy().reshape(-1, 28, 28) * 255).astype(np.uint8)
    cols, atlas_px = build_atlas(list(sprites), 28, os.path.join(OUT, "atlas.png"), mode="L", bg=0)

    export_encoder(model_a, os.path.join(OUT, "encoderA.json"), ca, sa)
    export_encoder(model_b, os.path.join(OUT, "encoderB.json"), cb, sb)

    write_meta(
        os.path.join(OUT, "meta.json"),
        name="MNIST",
        count=int(len(idx)),
        spritePx=28,
        atlasCols=int(cols),
        atlasPx=int(atlas_px),
        atlasMode="luminance",
        classes=[str(d) for d in range(10)],
        stateA="Autoencoder (unsupervised)",
        stateB="Semi-supervised fine-tune",
        userInput="draw",
    )
    print("done.")


if __name__ == "__main__":
    main()

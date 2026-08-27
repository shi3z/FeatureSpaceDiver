"""General-image pipeline for Feature Space Diver (CIFAR-10 by default, Caltech-101 optional).

State A: 3D UMAP of pretrained MobileNetV2 embeddings (no labels) — how the
         foundation model naturally organizes the images.
State B: supervised 3D UMAP of the same embeddings (labels as target) — the
         class-separated space, playing the role of the fine-tuned model.

Also exports:
  - mobilenet.onnx        the exact feature extractor, so the browser can embed
                          user-uploaded images with onnxruntime-web
  - projectorA/B.json     small MLPs (feature -> 3D) trained to mimic the two UMAP
                          mappings, so new points can be placed in either space

Usage: python prepare_general.py [cifar10|caltech101]
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms

from common import (build_atlas, export_mlp_json, normalize_coords,
                    save_labels, save_positions, write_meta)

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
OUT = os.path.join(os.path.dirname(__file__), "..", "web", "data", DATASET)
DATA_ROOT = os.path.join(os.path.dirname(__file__), "torch_data")
os.makedirs(OUT, exist_ok=True)

N_VIS = 8000
SPRITE_PX = 32 if DATASET == "cifar10" else 48


class FeatureExtractor(nn.Module):
    """MobileNetV2 features with ImageNet normalization baked in.

    Input: [N,3,224,224] RGB in [0,1]  ->  Output: [N,1280]
    Keeping the normalization inside the graph means the browser (ONNX) and this
    script preprocess identically: just resize and scale to [0,1].
    """

    def __init__(self):
        super().__init__()
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        h = self.features(x)
        h = F.adaptive_avg_pool2d(h, 1)
        return torch.flatten(h, 1)


def load_dataset():
    if DATASET == "cifar10":
        # HF parquet mirror (the cs.toronto.edu download is very slow):
        # https://huggingface.co/datasets/uoft-cs/cifar10 test split
        import io
        import pyarrow.parquet as pq

        pq_path = os.path.join(DATA_ROOT, "cifar10-test.parquet")
        if not os.path.exists(pq_path):
            import urllib.request
            url = ("https://huggingface.co/datasets/uoft-cs/cifar10/"
                   "resolve/main/plain_text/test-00000-of-00001.parquet")
            print("downloading", url)
            urllib.request.urlretrieve(url, pq_path)
        table = pq.read_table(pq_path)
        img_bytes = table.column("img").to_pylist()
        images = [Image.open(io.BytesIO(r["bytes"])).convert("RGB") for r in img_bytes]
        labels = np.array(table.column("label").to_pylist(), dtype=np.int64)
        classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
    elif DATASET == "caltech101":
        ds = datasets.Caltech101(DATA_ROOT, download=True)
        images, labels = [], []
        for img, y in ds:
            images.append(img.convert("RGB"))
            labels.append(y)
        labels = np.array(labels, dtype=np.int64)
        classes = ds.categories
    else:
        raise SystemExit(f"unknown dataset {DATASET}")

    if len(images) > N_VIS:
        idx = np.random.permutation(len(images))[:N_VIS]
        images = [images[i] for i in idx]
        labels = labels[idx]
    return images, labels, classes


@torch.no_grad()
def extract_features(model, images):
    model.eval()
    feats = []
    batch = []
    for i, img in enumerate(images):
        t = transforms.functional.to_tensor(
            img.resize((224, 224), Image.BILINEAR).convert("RGB"))
        batch.append(t)
        if len(batch) == 128 or i == len(images) - 1:
            x = torch.stack(batch).to(DEVICE)
            feats.append(model(x).cpu().numpy())
            batch = []
            if (i + 1) % 1024 < 128:
                print(f"  features {i+1}/{len(images)}")
    return np.concatenate(feats).astype(np.float32)


def run_umap(feats, labels):
    import umap

    print("UMAP (unsupervised)...")
    a = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                  metric="cosine", random_state=42).fit_transform(feats)
    print("UMAP (supervised)...")
    b = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine",
                  target_weight=0.6, random_state=42).fit_transform(feats, y=labels)
    return a.astype(np.float32), b.astype(np.float32)


class Projector(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.l1 = nn.Linear(in_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 3)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


def train_projector(feats, coords, tag, epochs=60):
    x = torch.from_numpy(feats).to(DEVICE)
    y = torch.from_numpy(coords).to(DEVICE)
    model = Projector(feats.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for i in range(0, n, 512):
            j = perm[i : i + 512]
            loss = F.mse_loss(model(x[j]), y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        if (ep + 1) % 20 == 0:
            print(f"[projector {tag}] epoch {ep+1}/{epochs} mse {tot/n:.4f}")
    return model


def export_projector(model, path):
    layers = [
        (model.l1.weight.detach().cpu().numpy(), model.l1.bias.detach().cpu().numpy(), "relu"),
        (model.l2.weight.detach().cpu().numpy(), model.l2.bias.detach().cpu().numpy(), "relu"),
        (model.l3.weight.detach().cpu().numpy(), model.l3.bias.detach().cpu().numpy(), "linear"),
    ]
    # projector is trained directly on normalized coords -> identity post-transform
    export_mlp_json(path, layers, [0, 0, 0], 1.0)


def export_onnx(model, path):
    model = model.cpu().eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=13, dynamo=False,
    )
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def main():
    print(f"dataset: {DATASET}  device: {DEVICE}")
    images, labels, classes = load_dataset()
    print(f"{len(images)} images, {len(classes)} classes")

    model = FeatureExtractor().to(DEVICE)
    feats = extract_features(model, images)

    coords_a, coords_b = run_umap(feats, labels)
    a_n, _, _ = normalize_coords(coords_a)
    b_n, _, _ = normalize_coords(coords_b)

    save_positions(os.path.join(OUT, "posA.bin"), a_n)
    save_positions(os.path.join(OUT, "posB.bin"), b_n)
    save_labels(os.path.join(OUT, "labels.bin"), labels.astype(np.uint8))

    cols, atlas_px = build_atlas(images, SPRITE_PX, os.path.join(OUT, "atlas.png"), mode="RGB")

    proj_a = train_projector(feats, a_n, "A")
    proj_b = train_projector(feats, b_n, "B")
    export_projector(proj_a, os.path.join(OUT, "projectorA.json"))
    export_projector(proj_b, os.path.join(OUT, "projectorB.json"))

    export_onnx(model, os.path.join(OUT, "mobilenet.onnx"))

    write_meta(
        os.path.join(OUT, "meta.json"),
        name="CIFAR-10" if DATASET == "cifar10" else "Caltech-101",
        count=int(len(images)),
        spritePx=SPRITE_PX,
        atlasCols=int(cols),
        atlasPx=int(atlas_px),
        atlasMode="rgb",
        classes=list(classes),
        stateA="MobileNetV2 embedding (unsupervised UMAP)",
        stateB="Supervised UMAP (label-separated)",
        userInput="upload",
    )
    print("done.")


if __name__ == "__main__":
    main()

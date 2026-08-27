"""Shared helpers for the Feature Space Diver data pipeline."""
import base64
import json
import os

import numpy as np
from PIL import Image


def normalize_coords(coords: np.ndarray):
    """Center and scale coords so RMS radius == 1. Returns (normed, center, scale)."""
    center = coords.mean(axis=0)
    c = coords - center
    rms = np.sqrt((c ** 2).sum(axis=1).mean())
    scale = 1.0 / max(rms, 1e-8)
    return (c * scale).astype(np.float32), center.astype(np.float64), float(scale)


def save_positions(path: str, coords: np.ndarray):
    coords.astype("<f4").tofile(path)


def save_labels(path: str, labels: np.ndarray):
    labels.astype(np.uint8).tofile(path)


def build_atlas(images, sprite_px: int, out_path: str, mode="RGB", bg=(0, 0, 0)):
    """Pack a list/array of PIL Images or HxW(xC) uint8 arrays into a square grid atlas.

    Returns (cols, atlas_size_px).
    """
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    atlas = Image.new(mode, (cols * sprite_px, cols * sprite_px), bg)
    for i, img in enumerate(images):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if img.mode != mode:
            img = img.convert(mode)
        if img.size != (sprite_px, sprite_px):
            img = img.resize((sprite_px, sprite_px), Image.LANCZOS)
        x = (i % cols) * sprite_px
        y = (i // cols) * sprite_px
        atlas.paste(img, (x, y))
    atlas.save(out_path)
    return cols, cols * sprite_px


def b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def export_mlp_json(path: str, layers, post_center, post_scale, input_norm=None):
    """Export an MLP as JSON for in-browser inference.

    layers: list of (weight [out,in] np.ndarray, bias [out] np.ndarray, activation str)
    post_center/post_scale: normalization applied to the 3D output so it matches
    the normalized coordinate bins (out = (raw - center) * scale).
    input_norm: optional dict describing input preprocessing, e.g. {"scale": 1/255}.
    """
    data = {
        "layers": [
            {
                "outDim": int(w.shape[0]),
                "inDim": int(w.shape[1]),
                "weight": b64_f32(w.flatten()),  # row-major [out][in]
                "bias": b64_f32(b),
                "act": act,
            }
            for (w, b, act) in layers
        ],
        "postCenter": [float(v) for v in post_center],
        "postScale": float(post_scale),
    }
    if input_norm:
        data["inputNorm"] = input_norm
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def write_meta(path: str, **kw):
    with open(path, "w") as f:
        json.dump(kw, f, indent=1)
    print(f"wrote {path}")

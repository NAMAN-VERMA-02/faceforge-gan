"""FaceForge demo: Gradio app served on Hugging Face Spaces.

Run locally:  python app.py   (then open http://localhost:7860)
Model source: env var FACEFORGE_MODEL (local generator.pt or HF repo id).
"""

from __future__ import annotations

import os
import random

import gradio as gr
import numpy as np
import torch

from faceforge.config import ATTRIBUTES
from faceforge.inference import DEFAULT_HUB_REPO, generate, interpolate, load_generator

_HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the full training run (model/), then the quick demo run (model_quick/), then the Hub.
_CANDIDATES = [os.path.join(_HERE, d, "generator.pt") for d in ("model", "model_quick")]
_BUNDLED = next((p for p in _CANDIDATES if os.path.exists(p)), None)
MODEL = os.environ.get("FACEFORGE_MODEL") or _BUNDLED or DEFAULT_HUB_REPO
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G, CFG = load_generator(MODEL, DEVICE)
LABELS = [a.replace("_", " ") for a in CFG.attributes]


def _attrs(selected: list[str]) -> torch.Tensor:
    return torch.tensor([1.0 if lbl in selected else 0.0 for lbl in LABELS])


def _tile(imgs: torch.Tensor, cols: int, scale: int = 3) -> np.ndarray:
    """Arrange N x H x W x 3 images in a grid and upscale with nearest-neighbour."""
    n, h, w, _ = imgs.shape
    rows = (n + cols - 1) // cols
    grid = np.full((rows * h + (rows - 1) * 2, cols * w + (cols - 1) * 2, 3), 255, np.uint8)
    for i, img in enumerate(imgs.numpy()):
        r, c = divmod(i, cols)
        grid[r * (h + 2): r * (h + 2) + h, c * (w + 2): c * (w + 2) + w] = img
    return grid.repeat(scale, 0).repeat(scale, 1)


def gen_faces(selected, seed, count, truncation):
    imgs = generate(G, _attrs(selected), seed=int(seed), n=int(count), truncation=float(truncation))
    return _tile(imgs, cols=min(int(count), 4))


def attribute_sweep(seed):
    """Same face (same z), each attribute switched on one at a time."""
    rows = [torch.zeros(len(LABELS))] + [torch.eye(len(LABELS))[i] for i in range(len(LABELS))]
    imgs = torch.cat([generate(G, y, seed=int(seed), n=1) for y in rows])
    return _tile(imgs, cols=len(rows), scale=2)


def morph(sel_a, sel_b, seed_a, seed_b, steps):
    imgs = interpolate(G, _attrs(sel_a), _attrs(sel_b), int(seed_a), int(seed_b), int(steps))
    return _tile(imgs, cols=int(steps), scale=2)


def random_seed():
    return random.randint(0, 99_999)


with gr.Blocks(title="FaceForge: Conditional GAN") as demo:
    gr.Markdown(
        "# FaceForge\nA conditional DCGAN trained on CelebA that generates **64×64 faces** you can "
        "steer with attributes. Same seed = same underlying face."
    )
    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                attrs = gr.CheckboxGroup(LABELS, value=["Smiling", "Young"], label="Attributes")
                seed = gr.Number(value=7, label="Seed", precision=0)
                dice = gr.Button("🎲 Random seed", size="sm")
                count = gr.Slider(1, 16, value=8, step=1, label="Number of faces")
                trunc = gr.Slider(0.3, 1.5, value=1.0, step=0.05, label="Truncation (lower = cleaner, less varied)")
                go = gr.Button("Generate", variant="primary")
            out = gr.Image(label="Generated faces", type="numpy")
        dice.click(random_seed, outputs=seed)
        go.click(gen_faces, [attrs, seed, count, trunc], out)
        demo.load(gen_faces, [attrs, seed, count, trunc], out)

    with gr.Tab("Attribute sweep"):
        gr.Markdown("Left: no attributes. Then " + ", ".join(LABELS) + ", one at a time, on the same face.")
        s_seed = gr.Number(value=7, label="Seed", precision=0)
        s_btn = gr.Button("Sweep", variant="primary")
        s_out = gr.Image(type="numpy")
        s_btn.click(attribute_sweep, s_seed, s_out)

    with gr.Tab("Morph"):
        with gr.Row():
            a_attrs = gr.CheckboxGroup(LABELS, value=["Male"], label="Start attributes")
            b_attrs = gr.CheckboxGroup(LABELS, value=["Smiling", "Blond Hair"], label="End attributes")
        with gr.Row():
            a_seed = gr.Number(value=1, label="Start seed", precision=0)
            b_seed = gr.Number(value=2, label="End seed", precision=0)
            steps = gr.Slider(3, 12, value=8, step=1, label="Steps")
        m_btn = gr.Button("Morph", variant="primary")
        m_out = gr.Image(type="numpy")
        m_btn.click(morph, [a_attrs, b_attrs, a_seed, b_seed, steps], m_out)

    gr.Markdown("Code: [github.com/Naman-verma-02/faceforge-gan](https://github.com/NAMAN-VERMA-02/faceforge-gan)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

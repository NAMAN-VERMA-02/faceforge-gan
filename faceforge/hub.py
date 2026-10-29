"""Publish a trained generator to the Hugging Face Hub.

    huggingface-cli login            # once, with a *write* token
    python -m faceforge.hub --repo <your-hf-username>/faceforge-gan --run runs/faceforge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_CARD = """---
license: mit
library_name: pytorch
tags: [gan, dcgan, conditional-gan, image-generation, celeba, faces]
datasets: [flwrlabs/celeba]
---

# FaceForge: conditional DCGAN (64x64 faces)

Generates 64x64 faces conditioned on six CelebA attributes:
{attrs}.

* Generator: transposed-conv DCGAN with a learned attribute embedding concatenated to z (dim {z_dim}).
* Discriminator: spectral-normalised conv net with projection conditioning.
* Weights are an exponential moving average (EMA) of the generator.

{metrics}

Code, training notebook and demo: https://github.com/{github}/faceforge-gan

```python
from faceforge.inference import load_generator, generate
import torch
G, cfg = load_generator("{repo}")
faces = generate(G, torch.tensor([1, 1, 0, 0, 1, 1]), seed=7, n=4)  # male, smiling, black hair, young
```
"""


def push(repo: str, run_dir: str, github: str = "Naman-verma-02", private: bool = False) -> str:
    from huggingface_hub import HfApi

    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text())
    history = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
    fids = [h["fid"] for h in history if "fid" in h]
    metrics = f"**Best FID:** {min(fids):.2f} after {len(history)} epochs." if fids else ""

    card = MODEL_CARD.format(attrs=", ".join(cfg["attributes"]), z_dim=cfg["z_dim"], metrics=metrics,
                             github=github, repo=repo)
    (run / "README.md").write_text(card)

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
    for name in ["generator.pt", "config.json", "history.json", "README.md"]:
        if (run / name).exists():
            api.upload_file(path_or_fileobj=run / name, path_in_repo=name, repo_id=repo)
    samples = sorted((run / "samples").glob("*.png"))
    if samples:
        api.upload_file(path_or_fileobj=samples[-1], path_in_repo="samples.png", repo_id=repo)
    url = f"https://huggingface.co/{repo}"
    print(f"pushed to {url}")
    return url


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="e.g. your-name/faceforge-gan")
    p.add_argument("--run", default="runs/faceforge")
    p.add_argument("--github", default="Naman-verma-02")
    p.add_argument("--private", action="store_true")
    a = p.parse_args()
    push(a.repo, a.run, a.github, a.private)

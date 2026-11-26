import csv
from pathlib import Path

import numpy as np
import torch

from tig.ga import GAParams, run_ga

FIELDS = ["seed_id", "label", "original_pred", "status", "predicted", "fitness", "generations"]


def run(generator, classifier, seeds, params: GAParams, seed: int, out_dir: Path):
    """Run the GA from every ``(key, z0, label)`` in ``seeds`` whose decoded
    image the classifier gets right. Returns the rows written to ``summary.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for n, (key, z0, label) in enumerate(seeds):
        # Stable Diffusion renders the seed image with its own (stronger) guidance, as the legacy did.
        decode_seed = getattr(generator, "decode_seed", generator.decode)
        with torch.no_grad():
            original = decode_seed(z0, torch.tensor([label], device=z0.device))
            original_pred = int(classifier(original).argmax(dim=1).item())
        row = {"seed_id": key, "label": label, "original_pred": original_pred}

        if original_pred != label:
            # The starting image is already misclassified: not a valid seed.
            rows.append({**row, "status": "skipped"})
            continue

        result = run_ga(generator, classifier, z0, label, params, seed=seed + n)
        name = f"image_{n}_X{label}_Y{result.predicted_label}"
        np.save(out_dir / f"{name}_original.npy", original.squeeze(0).cpu().numpy())
        np.save(out_dir / f"{name}.npy", result.image.numpy())
        rows.append({
            **row,
            "status": "misbehaviour" if result.misbehaviour else "not_found",
            "predicted": result.predicted_label,
            "fitness": result.fitness,
            "generations": result.generations,
        })

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows

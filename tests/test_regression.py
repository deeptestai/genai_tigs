"""Regression tests on the real checkpoints shipped in the repo.

Each golden file pins today's output; a refactor that changes it fails here.

The runs use float64 on a single CPU thread. With a small mutation the population members
are nearly identical, so their fitness values differ by about float32 rounding: any change in
summation order (thread count, CPU kernels, platform) reorders them and the GA takes another
path. In float32 the GAN MNIST run ended 0.17 apart on Linux and Windows, and VAE MNIST
changed with the thread count; in float64 on one thread the results match.

After an intentional behaviour change, regenerate them with:

    TIG_UPDATE_GOLDEN=1 python -m pytest tests/test_regression.py
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tig.config import get_config
from tig.ga import GAParams, run_ga
from tig.generators import build, can_encode

GOLDEN_DIR = Path(__file__).parent / "golden"
# (model, dataset, mutation level with a documented perturbation)
CASES = [
    ("vae", "mnist", "high"), ("gan", "mnist", "low"),
    # These need the Google Drive weights in weights/ and are skipped without them.
    ("vae", "svhn", "high"), ("gan", "svhn", "high"), ("vae", "cifar10", "high"), ("gan", "cifar10", "low"),
]


def _start(generator, classifier):
    """A fixed, correctly classified starting point; no dataset download needed."""
    z = torch.randn(1, generator.latent_dim, generator=torch.Generator().manual_seed(0))
    if can_encode(generator):
        z0 = generator.encode(generator.decode(z))
        return z0, int(classifier(generator.decode(z0)).argmax())
    for label in range(10):  # conditional: first class the classifier recognises
        if int(classifier(generator.decode(z, torch.tensor([label]))).argmax()) == label:
            return z, label
    raise AssertionError("no class is recognised from this latent")


@pytest.fixture
def reproducible_cpu():
    """float64 and one thread for the duration of a test, restored afterwards."""
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("model,dataset,mutation", CASES, ids=[f"{m}-{d}" for m, d, _ in CASES])
def test_matches_golden(model, dataset, mutation, reproducible_cpu):
    config = get_config(model, dataset)
    if not all(p.exists() for p in config.weights.values()):
        pytest.skip("weights not available")

    generator, classifier = build(config, torch.device("cpu"))
    (generator.vae if can_encode(generator) else generator.generator).double()
    classifier.double()
    z0, label = _start(generator, classifier)
    params = GAParams(
        generations=30, perturbation=config.perturbation(mutation), min_val=config.min_val, max_val=config.max_val
    )
    result = run_ga(generator, classifier, z0, label, params, seed=0)

    golden_path = GOLDEN_DIR / f"{dataset}_{model}_seed0.npz"
    if os.environ.get("TIG_UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.savez_compressed(
            golden_path, label=label, image=result.image.numpy(), fitness=result.fitness,
            predicted=result.predicted_label, generations=result.generations,
        )
        pytest.skip("golden file updated")

    golden = np.load(golden_path)
    assert label == golden["label"]
    assert result.predicted_label == golden["predicted"]
    assert result.generations == golden["generations"]
    assert result.fitness == pytest.approx(float(golden["fitness"]), abs=1e-6)
    np.testing.assert_allclose(result.image.numpy(), golden["image"], atol=1e-8)

"""Latent-space genetic algorithm shared by every generator (SINVAD-style)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from tig.fitness import calculate_fitness


@dataclass(frozen=True)
class GAParams:
    generations: int = 250
    pop_size: int = 25
    best_left: int = 10
    perturbation: float = 0.0146425576210022
    min_val: float = -np.inf
    max_val: float = np.inf
    # Population members decoded and classified at once (None = the whole population).
    # Smaller batches only lower peak GPU memory; the search itself is unchanged.
    batch_size: Optional[int] = None


@dataclass
class GAResult:
    image: torch.Tensor
    fitness: float
    predicted_label: int
    generations: int

    @property
    def misbehaviour(self) -> bool:
        return self.fitness < 0


def crossover(mom, dad, np_rng, unit=1):
    """One-point crossover of two ``[1, latent_dim]`` latents. The cut falls on a multiple of
    ``unit``: 1 for flat latents, one channel (64x64 values) for Stable Diffusion, whose legacy
    scripts cut its 4x64x64 latent between channels."""
    cut = unit * int(np_rng.integers(mom.shape[1] // unit))
    return torch.cat([mom[:, :cut], dad[:, cut:]], dim=1)


@torch.no_grad()
def run_ga(generator, classifier, z0, label, params, seed):
    """Search around latent ``z0`` (shape ``[1, latent_dim]``) for an input
    the classifier does not label as ``label``.

    All randomness comes from ``seed``, so equal inputs give equal outputs.
    """
    if not 2 <= params.best_left < params.pop_size:
        raise ValueError("best_left must be in [2, pop_size)")
    if params.generations < 1:
        raise ValueError("generations must be >= 1")
    if params.batch_size is not None and params.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    device = z0.device
    t_rng = torch.Generator(device="cpu").manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    latent_dim = z0.shape[1]

    def noise(shape):
        return torch.randn(shape, generator=t_rng).to(device)

    labels = torch.full((params.pop_size,), label, dtype=torch.long, device=device)
    population = [
        torch.clamp(z0 + params.perturbation * noise((1, latent_dim)), params.min_val, params.max_val)
        for _ in range(params.pop_size)
    ]
    perturbation = params.perturbation
    prev_best = np.inf
    best = GAResult(image=None, fitness=np.inf, predicted_label=label, generations=0)

    step = params.batch_size or params.pop_size
    unit = getattr(generator, "crossover_unit", 1)

    def evaluate(zs):
        """Decode and classify the population ``step`` members at a time."""
        images, logits = [], []
        for i in range(0, len(zs), step):
            batch = generator.decode(torch.cat(zs[i:i + step], dim=0), labels[i:i + step])
            logits.append(classifier(batch).detach().cpu().numpy())
            images.append(batch)
        return torch.cat(images, dim=0), np.concatenate(logits, axis=0)

    for gen in range(params.generations):
        images, logits = evaluate(population)
        scores = np.array([calculate_fitness(l, label) for l in logits])

        idx = int(np.argmin(scores))
        if scores[idx] < best.fitness:
            best.fitness = float(scores[idx])
            best.image = images[idx].detach().cpu()
            best.predicted_label = int(logits[idx].argmax())
        best.generations = gen + 1

        now_best = scores[idx]
        if now_best < 0:
            break
        # Stagnation widens the search; any progress resets it.
        perturbation = perturbation * 2 if now_best == prev_best else params.perturbation
        prev_best = now_best

        parents = [population[i] for i in np.argsort(scores)[: params.best_left]]
        children = []
        for _ in range(params.pop_size - params.best_left):
            mom, dad = np_rng.choice(params.best_left, size=2, replace=False)
            child = crossover(parents[mom], parents[dad], np_rng, unit)
            changed = (child != z0).float()  # mutate only genes that already moved
            children.append(child + perturbation * noise(child.shape) * changed)
        population = [torch.clamp(z, params.min_val, params.max_val) for z in parents + children]

    return best

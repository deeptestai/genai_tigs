import random

import numpy as np
import pytest
import torch

from tig.fitness import calculate_fitness
from tig.ga import GAParams, run_ga
from tig.seed import set_seed


def test_fitness_positive_when_label_wins():
    assert calculate_fitness(np.array([0.1, 3.0, 1.0]), 1) == pytest.approx(2.0)


def test_fitness_negative_when_label_loses():
    assert calculate_fitness(np.array([0.1, 3.0, 1.0]), 2) == pytest.approx(-2.0)


def test_set_seed_makes_all_rngs_repeatable():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()

    set_seed(42)
    first = draw()
    set_seed(42)
    assert draw() == first


def _start(gen, clf):
    z0 = torch.ones(1, gen.latent_dim)  # away from 0, where all logits tie
    label = int(clf(gen.decode(z0)).argmax())
    return z0, label


def test_ga_is_deterministic_for_same_seed(toy):
    gen, clf = toy
    z0, label = _start(gen, clf)
    params = GAParams(generations=15, pop_size=10, best_left=4, perturbation=0.05)
    a = run_ga(gen, clf, z0, label, params, seed=3)
    b = run_ga(gen, clf, z0, label, params, seed=3)
    assert torch.equal(a.image, b.image)
    assert (a.fitness, a.generations) == (b.fitness, b.generations)


def test_ga_batch_size_does_not_change_the_search(toy):
    gen, clf = toy
    z0, label = _start(gen, clf)
    base = dict(generations=15, pop_size=10, best_left=4, perturbation=0.05)
    whole = run_ga(gen, clf, z0, label, GAParams(**base), seed=3)
    batched = run_ga(gen, clf, z0, label, GAParams(**base, batch_size=3), seed=3)  # 3 + 3 + 3 + 1
    assert (batched.generations, batched.predicted_label) == (whole.generations, whole.predicted_label)
    assert batched.fitness == pytest.approx(whole.fitness, abs=1e-5)
    torch.testing.assert_close(batched.image, whole.image)


def test_ga_finds_misbehaviour_on_easy_problem(toy):
    gen, clf = toy
    z0, label = _start(gen, clf)
    result = run_ga(gen, clf, z0, label, GAParams(generations=200, pop_size=20, best_left=5, perturbation=0.5), seed=0)
    assert result.misbehaviour
    assert result.predicted_label != label
    assert result.generations < 200  # stops as soon as the label flips


def test_ga_reports_label_when_nothing_found(toy):
    gen, clf = toy
    z0, label = _start(gen, clf)
    result = run_ga(gen, clf, z0, label, GAParams(generations=1, pop_size=10, best_left=4, perturbation=1e-6), seed=0)
    assert not result.misbehaviour
    assert result.predicted_label == label


def test_ga_rejects_bad_selection_size(toy):
    gen, clf = toy
    z0, label = _start(gen, clf)
    with pytest.raises(ValueError):
        run_ga(gen, clf, z0, label, GAParams(pop_size=5, best_left=5), seed=0)
    with pytest.raises(ValueError):
        run_ga(gen, clf, z0, label, GAParams(generations=0), seed=0)
    with pytest.raises(ValueError):
        run_ga(gen, clf, z0, label, GAParams(batch_size=0), seed=0)

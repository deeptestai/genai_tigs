import csv

import pytest
import torch

from tig import cli
from tig.ga import GAParams
from tig import seeds
from tig.runner import run


def test_every_combination_is_ported():
    from tig.config import CONFIGS, DATASETS, MODELS

    assert {(m, d) for m in MODELS for d in DATASETS} == set(CONFIGS)
    assert all(c.ported for c in CONFIGS.values())


def test_unported_combination_exits_with_legacy_hint(capsys, monkeypatch):
    # Every combination is ported now; keep the hint working for a future one that isn't.
    from tig.config import CONFIGS, LEGACY, TigConfig

    unported = TigConfig(model="sd", dataset="svhn", legacy_script=LEGACY / "svhn/sd/tig_sd.py")
    monkeypatch.setitem(CONFIGS, ("sd", "svhn"), unported)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--model", "sd", "--dataset", "svhn"])
    assert exc.value.code == 2
    assert "legacy/svhn/sd/tig_sd.py" in capsys.readouterr().err


# (model, dataset, level, initial perturbation hard-coded in the legacy script)
LEGACY_PERTURBATIONS = [
    ("vae", "mnist", "high", 0.0146425576210022),
    ("vae", "svhn", "high", 0.0199879989624023),
    ("vae", "cifar10", "high", 0.0160218644142151),
    ("vae", "imagenet_pizza", "low", 0.00216982479095459),
    ("gan", "mnist", "low", 0.00177406988143921),
    ("gan", "svhn", "high", 0.0172531118392944),
    ("gan", "cifar10", "low", 0.00182106418609619),
    ("gan", "imagenet_pizza", "low", 0.000799975180625916),
    ("sd", "mnist", "low", 0.00216688718795776),
    ("sd", "svhn", "low", 0.00217826347351074),
    ("sd", "cifar10", "low", 0.00218342552185059),
    ("sd", "imagenet_pizza", "high", 0.0217526149749756),
]


@pytest.mark.parametrize("model,dataset,level,legacy_value", LEGACY_PERTURBATIONS)
def test_perturbation_rule_reproduces_legacy_values(model, dataset, level, legacy_value):
    from tig.config import get_config

    assert get_config(model, dataset).perturbation(level) == pytest.approx(legacy_value, rel=1e-9)


def test_unknown_dataset_rejected():
    with pytest.raises(SystemExit):
        cli.main(["--model", "vae", "--dataset", "fashion"])


def test_runner_writes_summary_and_skips_misclassified(toy, tmp_path):
    gen, clf = toy
    images = [gen.decode(torch.full((1, gen.latent_dim), v)).squeeze(0) for v in (-1.0, 0.0, 1.0)]
    labels = [int(clf(img.unsqueeze(0)).argmax()) for img in images]
    labels[0] = (labels[0] + 1) % 3  # force one wrong label so it is skipped
    test_set = list(zip(images, labels))

    params = GAParams(generations=50, pop_size=10, best_left=4, perturbation=0.5)
    seed_inputs = seeds.from_test_set(gen, test_set, num=3, seed=0, device="cpu")
    rows = run(gen, clf, seed_inputs, params, seed=0, out_dir=tmp_path)

    assert len(rows) == 3
    assert sum(r["status"] == "skipped" for r in rows) == 1
    with open(tmp_path / "summary.csv") as f:
        assert len(list(csv.DictReader(f))) == 3
    assert len(list(tmp_path.glob("*_original.npy"))) == 2


def test_missing_weights_exit_cleanly(capsys):
    from tig.config import get_config

    if get_config("vae", "svhn").weights["generator"].exists():
        pytest.skip("SVHN weights are present")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--model", "vae", "--dataset", "svhn"])
    assert exc.value.code == 1
    assert "missing generator weights" in capsys.readouterr().err


@pytest.mark.parametrize("extra,expected", [([], "high"), (["--mutation", "low"], "low"), (["--perturbation", "0.0177"], "p0.0177")])
def test_output_dir_names_the_perturbation_used(extra, expected):
    args = cli.build_parser().parse_args(["--model", "gan", "--dataset", "mnist", *extra])
    assert cli.run_label(args) == expected

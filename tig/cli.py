"""Command line entry point: ``python -m tig --model vae --dataset mnist``."""

import argparse
from pathlib import Path

import torch

from tig.config import DATASETS, MODELS, MUTATIONS, REPO_ROOT, get_config
from tig.ga import GAParams
from tig.seed import make_deterministic, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tig", description="Generate misbehaviour-inducing test inputs for an image classifier.")
    p.add_argument("--model", choices=MODELS, required=True, help="generative model used to explore inputs")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--mutation", choices=MUTATIONS, default="high", help="mutation extent (default: high)")
    p.add_argument("--perturbation", type=float, help="override the initial perturbation of --mutation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-images", type=int, default=100, help="seed inputs to sample (default: 100)")
    p.add_argument("--generations", type=int, default=GAParams.generations)
    p.add_argument("--pop-size", type=int, default=GAParams.pop_size)
    p.add_argument("--best-left", type=int, default=GAParams.best_left)
    p.add_argument("--batch-size", type=int, help="population members decoded at once; lower it if the GPU runs out of memory")
    p.add_argument("--sd-precision", choices=("fp16", "fp32"), default="fp16",
                   help="Stable Diffusion weights (default fp16 as in the paper; fp32 for GPUs that return NaN in fp16, e.g. GTX 16xx)")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--out", type=Path, help="output dir (default: results/<model>_<dataset>_<mutation or pVALUE>_s<seed>)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def run_label(args) -> str:
    """Mutation part of the output dir name: the level, or the explicit value when overridden."""
    return f"p{args.perturbation:g}" if args.perturbation is not None else args.mutation


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = get_config(args.model, args.dataset)
    if not config.ported:
        script = config.legacy_script.relative_to(REPO_ROOT).as_posix()
        parser.error(f"{args.model} x {args.dataset} is not ported yet; legacy script: {script}")

    perturbation = args.perturbation if args.perturbation is not None else config.perturbation(args.mutation)

    # Imported here so --help and argument errors don't pay for loading torchvision.
    from tig import seeds
    from tig.datasets import load_test_set
    from tig.generators import build, can_encode
    from tig.runner import run

    set_seed(args.seed)
    make_deterministic()
    device = torch.device(args.device)
    try:
        generator, classifier = build(config, device, sd_precision=args.sd_precision)
    except FileNotFoundError as e:
        parser.exit(1, f"tig: error: {e}\n")
    except ImportError as e:
        parser.exit(1, f"tig: error: {e}; ImageNet needs: pip install pytorch-pretrained-biggan scipy\n")
    params = GAParams(
        generations=args.generations,
        pop_size=args.pop_size,
        best_left=args.best_left,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        perturbation=perturbation,
        min_val=config.min_val,
        max_val=config.max_val,
    )
    out = args.out or REPO_ROOT / "results" / f"{args.model}_{args.dataset}_{run_label(args)}_s{args.seed}"
    if can_encode(generator):
        try:
            test_set = load_test_set(args.dataset, args.data_root)
        except FileNotFoundError as e:
            parser.exit(1, f"tig: error: {e}\n")
        seed_inputs = seeds.from_test_set(generator, test_set, args.num_images, args.seed, device)
    else:
        seed_inputs = seeds.from_prior(
            generator, args.num_images, args.seed, device, classes=config.classes, bound=config.prior_bound
        )
    rows = run(generator, classifier, seed_inputs, params, args.seed, out)

    found = sum(r["status"] == "misbehaviour" for r in rows)
    tried = sum(r["status"] != "skipped" for r in rows)
    print(f"{found}/{tried} misbehaviours found ({len(rows) - tried} skipped); results in {out}")
    return 0

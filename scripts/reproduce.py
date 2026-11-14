"""Run the fixed paper experiment matrix from this standalone folder."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 0, 1)
DATASETS = ("movielens", "mal")


def config_stems(dataset):
    return {
        "real": f"dt_{dataset}_current_state",
        "no_rtg": f"dt_no_rtg_{dataset}_current_state",
        "shuffled": f"dt_shuffled_rtg_{dataset}_current_state",
        "sasrec": f"sasrec_{dataset}",
    }


def commands(stage, split, checkpoints, results):
    """Yield argv lists in dependency order, with no shell interpolation."""
    output = results / split
    stages = ("train", "evaluate", "diagnostics", "probe", "report")
    for current in stages if stage == "all" else (stage,):
        if current == "report":
            yield ["scripts/summarize.py", "--results", str(output),
                   "--split", split, "--out", str(output / "summary")]
            continue
        for dataset in DATASETS:
            stems = config_stems(dataset)
            if current == "probe":
                yield ["scripts/rtg_state_probe.py", "--dataset", dataset,
                       "--split", split, "--mlp",
                       "--out", str(output / f"probe_{dataset}.csv")]
                continue
            for seed in SEEDS:
                ckpts = {key: str(checkpoints / f"{stem}_seed{seed}_best.pt")
                         for key, stem in stems.items()}
                if current == "train":
                    for stem in stems.values():
                        yield ["scripts/train.py", "--config", f"configs/{stem}.yaml",
                               "--seed", str(seed), "--out", str(checkpoints)]
                elif current == "evaluate":
                    for kind, stem in stems.items():
                        yield ["scripts/evaluate.py", "--checkpoint", ckpts[kind],
                               "--split", split, "--batch-size", "1024",
                               "--k", "1", "5", "10", "--n-boot", "1000",
                               "--bootstrap-seed", "0", "--out",
                               str(output / f"eval_{stem}_seed{seed}.csv")]
                elif current == "diagnostics":
                    for kind in ("real", "shuffled"):
                        yield ["scripts/prefix_k_sweep.py", "--checkpoint", ckpts[kind],
                               "--no-rtg-checkpoint", ckpts["no_rtg"],
                               "--split", split, "--traj-limit", "1000",
                               "--sample-seed", "42", "--n-boot", "1000",
                               "--n-levels", "11", "--ks", "1,5,10,15,20",
                               "--focus-genre", "Crime" if dataset == "movielens" else "Drama",
                               "--out", str(output / f"{dataset}_seed{seed}_{kind}")]
                    yield ["scripts/match_and_score.py", "--checkpoint", ckpts["real"],
                           "--split", split, "--mode", "last", "--top-k", "5",
                           "--traj-limit", "2000", "--sample-seed", "42",
                           "--n-boot", "1000", "--bootstrap-seed", "0",
                           "--n-levels", "11", "--out",
                           str(output / f"match_{dataset}_seed{seed}.csv")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "train", "evaluate", "diagnostics", "probe", "report"], default="all")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="The frozen paper uses test; val reproduces supporting validation results.")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading data or running experiments.")
    args = parser.parse_args()
    planned = list(commands(args.stage, args.split, args.checkpoints, args.results))
    print(f"Working directory: {ROOT}\nCommands: {len(planned)}", flush=True)
    for command in planned:
        argv = [sys.executable, *command]
        print(shlex.join(argv), flush=True)
        if not args.dry_run:
            subprocess.run(argv, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

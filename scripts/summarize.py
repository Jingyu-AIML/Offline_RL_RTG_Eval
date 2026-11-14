"""Rebuild paper tables and figures from the 44 final per-run result CSVs."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reproduce import DATASETS, SEEDS, config_stems
from scripts.render_figures import render


def read_result(path, split, sweep=False):
    frame = pd.read_csv(path)
    if frame.empty or set(frame["evaluation_split"]) != {split}:
        raise ValueError(f"{path}: expected nonempty {split} results")
    if sweep and set(frame["rtg_grid_split"]) != {"val"}:
        raise ValueError(f"{path}: RTG targets must come from validation support")
    return frame


def aggregate(frame, keys, values):
    grouped = frame.groupby(keys, sort=False)[values].agg(["mean", "std", "count"])
    grouped.columns = [f"{value}_{stat}" for value, stat in grouped.columns]
    return grouped.reset_index()


def collect(results, split):
    locality, accuracy, matches, probes, scope = [], [], [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            for kind in ("real", "shuffled"):
                path = results / f"{dataset}_seed{seed}_{kind}" / "prefix_k_sweep.csv"
                frame = read_result(path, split, sweep=True)
                if set(frame["K"]) != {1, 5, 10, 15, 20}:
                    raise ValueError(f"{path}: incomplete locality ladder")
                genres = [c[4:] for c in frame if c.startswith("pct_")
                          and not c.endswith(("_lo", "_hi"))]
                for k, group in frame.groupby("K", sort=True):
                    group = group.sort_values("rtg_target")
                    if len(group) != 11 or group["rtg_target"].nunique() != 11:
                        raise ValueError(f"{path}: expected 11 distinct targets at K={k}")
                    low, high = group.iloc[0], group.iloc[-1]
                    for genre in genres:
                        locality.append({
                            "dataset": dataset, "split": split, "seed": seed,
                            "kind": kind, "K": k, "genre": genre,
                            "rtg_low": low["rtg_target"], "rtg_high": high["rtg_target"],
                            "pct_low": low[f"pct_{genre}"], "pct_high": high[f"pct_{genre}"],
                            "delta_pp": 100 * (high[f"pct_{genre}"] - low[f"pct_{genre}"]),
                            "delta_ci_low_pp": 100 * low[f"delta_pct_{genre}_lo"],
                            "delta_ci_high_pp": 100 * low[f"delta_pct_{genre}_hi"],
                        })
            for kind, stem in config_stems(dataset).items():
                path = results / f"eval_{stem}_seed{seed}.csv"
                frame = read_result(path, split)
                frame = frame[frame["metric"].eq("hit_rate")].copy()
                if len(frame) != 3 or set(frame["k"]) != {1, 5, 10} or set(frame["seed"]) != {seed}:
                    raise ValueError(f"{path}: expected HR@1/5/10 for seed {seed}")
                if set(frame["checkpoint"]) != {f"{stem}_seed{seed}_best.pt"}:
                    raise ValueError(f"{path}: expected a validation-selected best checkpoint")
                frame.insert(0, "dataset", dataset)
                frame.insert(1, "kind", kind)
                accuracy.append(frame)
                if kind == "real":
                    scope.append({"dataset": dataset, "seed": seed,
                                  "n_users": frame.iloc[0]["n_users"],
                                  "n_contexts": frame.iloc[0]["n_predictions"]})
            path = results / f"match_{dataset}_seed{seed}.csv"
            frame = read_result(path, split, sweep=True).sort_values("target_rtg")
            if len(frame) != 11 or frame["target_rtg"].nunique() != 11:
                raise ValueError(f"{path}: expected 11 distinct RTG targets")
            low, high = frame.iloc[0], frame.iloc[-1]
            row = {"dataset": dataset, "seed": seed,
                   "rtg_low": low["target_rtg"], "rtg_high": high["target_rtg"]}
            for metric in ("match_rate_top1", "avg_rating_top1_match"):
                row[f"{metric}_q05"] = low[metric]
                row[f"{metric}_q95"] = high[metric]
                row[f"{metric}_delta"] = high[metric] - low[metric]
            matches.append(row)
        probe = read_result(results / f"probe_{dataset}.csv", split)
        if len(probe) != 1 or set(probe["dataset"]) != {dataset} or probe["mlp_r2_eval"].isna().any():
            raise ValueError(f"{dataset}: expected both Ridge and MLP probe results")
        probes.append(probe)
    return (pd.DataFrame(locality), pd.concat(accuracy, ignore_index=True),
            pd.DataFrame(matches), pd.concat(probes, ignore_index=True), pd.DataFrame(scope))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/test"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    locality, accuracy, matches, probes, scope = collect(args.results, args.split)
    out = args.out or args.results / "summary"
    focus = locality[((locality.dataset == "movielens") & (locality.genre == "Crime")) |
                     ((locality.dataset == "mal") & (locality.genre == "Drama"))]
    match_metrics = [c for c in matches if c.endswith(("_q05", "_q95", "_delta"))]
    tables = {
        "scope_by_seed": scope,
        "locality_all_genres_by_seed": locality,
        "locality_all_genres_summary": aggregate(locality, ["dataset", "kind", "K", "genre"], ["delta_pp"]),
        "locality_focus_by_seed": focus,
        "locality_focus_summary": aggregate(focus, ["dataset", "kind", "K", "genre"], ["delta_pp"]),
        "accuracy_by_seed": accuracy,
        "accuracy_summary": aggregate(accuracy, ["dataset", "kind", "k"], ["value"]),
        "match_by_seed": matches,
        "match_summary": aggregate(matches, ["dataset"], match_metrics),
        "probe_summary": probes,
    }
    for name, table in tables.items():
        numeric = table.select_dtypes(include="number")
        # Bootstrap intervals and the MLP probe are required for the paper report.
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{name}: missing or non-finite metrics; run all paper diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    render(locality, args.results, out / "figures", args.split)
    print(f"Paper tables: {out}\nPaper figures: {out / 'figures'}")
    print(tables["locality_focus_summary"].to_string(index=False))
    print(tables["accuracy_summary"].to_string(index=False))


if __name__ == "__main__":
    main()

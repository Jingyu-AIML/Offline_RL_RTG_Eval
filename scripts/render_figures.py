"""Regenerate the two manuscript figures at their final one-column size."""

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RED = "#b2182b"
BLUE = "#2166ac"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.25,
    }
)


def save(fig: plt.Figure, stem: str, figures) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.pdf")
    fig.savefig(figures / f"{stem}.png", dpi=300)
    plt.close(fig)


def locality_ladder(frame, figures, split) -> None:
    frame = frame[(frame["dataset"] == "movielens") & (frame["genre"] == "Crime") & (frame["split"] == split)].copy()

    fig, ax = plt.subplots(figsize=(3.35, 2.23), layout="constrained")
    styles = {
        "real": (RED, "o", "DT+RTG mean"),
        "shuffled": (BLUE, "s", "shuffled RTG mean"),
    }
    for kind, (color, marker, label) in styles.items():
        subset = frame[frame["kind"].eq(kind)]
        for _, seed_frame in subset.groupby("seed"):
            seed_frame = seed_frame.sort_values("K")
            ax.plot(
                seed_frame["K"],
                seed_frame["delta_pp"],
                color=color,
                alpha=0.20,
                linewidth=0.8,
                marker="o",
                markersize=2.0,
                markeredgewidth=0,
                zorder=1,
            )
        mean = subset.groupby("K", as_index=False)["delta_pp"].mean()
        ax.plot(
            mean["K"],
            mean["delta_pp"],
            color=color,
            marker=marker,
            markersize=3.4,
            markeredgewidth=0,
            label=label,
            zorder=3,
        )

    ax.axhline(0, color="#777777", linewidth=0.55, zorder=0)
    ax.set_title("MovieLens RTG locality ladder", pad=3)
    ax.set_xlabel("RTG positions overwritten (K)")
    ax.set_ylabel("Crime share shift, q95 − q05 (pp)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(-0.8, 28.5)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.45)
    ax.legend(loc="upper left", frameon=False, handlelength=1.8)
    save(fig, "ml_current_state_ladder_3seed", figures)


def match_and_score(results, figures) -> None:
    ml = pd.read_csv(results / "match_movielens_seed42.csv")
    mal = pd.read_csv(results / "match_mal_seed42.csv")

    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.42), layout="constrained")
    for frame, color, label in [(ml, RED, "MovieLens"), (mal, BLUE, "MAL")]:
        axes[0].plot(
            frame["target_rtg"],
            100 * frame["match_rate_top1"],
            color=color,
            marker="o",
            markersize=2.5,
            markeredgewidth=0,
            label=label,
        )
        axes[1].plot(
            frame["target_rtg"],
            frame["avg_rating_top1_match"],
            color=color,
            marker="o",
            markersize=2.5,
            markeredgewidth=0,
            label=label,
        )

    axes[0].set_title("Top-1 action match", pad=3)
    axes[0].set_ylabel("Match rate (%)")
    axes[0].set_ylim(21.8, 39.2)
    axes[0].set_yticks([22.5, 27.5, 32.5, 37.5])

    axes[1].set_title("Matched rating", pad=3)
    axes[1].set_ylabel("Mean rating")
    axes[1].set_ylim(3.35, 7.85)
    axes[1].set_yticks([4, 5, 6, 7])

    for ax in axes:
        ax.set_xlabel("Target RTG")
        ax.set_xlim(9.3, 18.6)
        ax.set_xticks([10, 12, 14, 16, 18])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45)
        ax.legend(loc="center right", frameon=False, handlelength=1.5)

    save(fig, "match_and_score_current_state", figures)


def render(locality, results, figures, split="test") -> None:
    """Figure 1 aggregates three seeds; Figure 2 uses seed 42, as in the paper."""
    locality_ladder(locality, figures, split)
    match_and_score(results, figures)

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def _save_line(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    if x not in df or y not in df or df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df[x], df[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset")
    parser.add_argument("--strategy")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--partition-type")
    args = parser.parse_args(argv)
    all_rounds = []
    summaries = []
    for metrics_path in args.results_dir.glob("*/round_metrics.csv"):
        df = pd.read_csv(metrics_path)
        df["experiment_id"] = metrics_path.parent.name
        all_rounds.append(df)
    for summary_path in args.results_dir.glob("*/summary.json"):
        try:
            summaries.append(pd.read_json(summary_path, typ="series").to_dict())
        except ValueError:
            continue
    rounds = pd.concat(all_rounds, ignore_index=True) if all_rounds else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    _save_line(rounds, "round", "global_accuracy", args.results_dir / "accuracy_by_round.png", "Accuracy by round")
    _save_line(rounds, "round", "global_loss", args.results_dir / "loss_by_round.png", "Loss by round")
    _save_line(summary, "mean_partition_js_divergence", "final_accuracy", args.results_dir / "client_fairness.png", "Fairness proxy")
    _save_line(summary, "total_communication_bytes", "final_accuracy", args.results_dir / "communication_vs_accuracy.png", "Communication vs accuracy")
    _save_line(summary, "convergence_round", "final_accuracy", args.results_dir / "convergence_rounds.png", "Convergence")
    _save_line(summary, "mean_partition_js_divergence", "final_accuracy", args.results_dir / "accuracy_vs_alpha.png", "Accuracy vs alpha/skew")
    print(f"Plots saved to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

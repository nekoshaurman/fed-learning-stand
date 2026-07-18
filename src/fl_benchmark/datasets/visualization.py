from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def save_partition_plots(statistics: pd.DataFrame, matrix: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_matrix = matrix.drop(columns=["client_id"], errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(class_matrix.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_title("Partition class counts")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "partition_heatmap.png", dpi=150)
    plt.close(fig)
    for column, filename, ylabel in [
        ("num_examples", "partition_sizes.png", "Examples"),
        ("entropy", "client_entropy.png", "Entropy"),
        ("js_divergence", "client_js_divergence.png", "JS divergence"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(statistics["client_id"], statistics[column])
        ax.set_xlabel("Client")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

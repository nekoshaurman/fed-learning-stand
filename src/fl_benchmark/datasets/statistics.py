from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PartitionStatistics:
    statistics: pd.DataFrame
    matrix: pd.DataFrame


def entropy(distribution: np.ndarray) -> float:
    probs = distribution[distribution > 0]
    return float(-(probs * np.log2(probs)).sum())


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(float(p.sum()), 1.0)
    q = q / max(float(q.sum()), 1.0)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_partition_statistics(
    targets: np.ndarray,
    partitions: list[list[int]],
    num_classes: int | None = None,
) -> PartitionStatistics:
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.arange(num_classes or int(targets.max()) + 1)
    global_counts = np.bincount(targets, minlength=len(classes)).astype(float)
    global_dist = global_counts / max(global_counts.sum(), 1.0)
    matrix_rows: list[np.ndarray] = []
    stat_rows: list[dict[str, float | int | str]] = []
    for cid, indices in enumerate(partitions):
        local_targets = targets[np.asarray(indices, dtype=int)]
        counts = np.bincount(local_targets, minlength=len(classes)).astype(float)
        matrix_rows.append(counts)
        total = float(counts.sum())
        dist = counts / total if total else np.zeros_like(counts)
        nonzero = counts[counts > 0]
        stat_rows.append(
            {
                "client_id": cid,
                "num_examples": int(total),
                "num_classes": int((counts > 0).sum()),
                "class_distribution": ";".join(str(int(v)) for v in counts),
                "entropy": entropy(dist),
                "js_divergence": jensen_shannon_divergence(dist, global_dist),
                "max_class_fraction": float(dist.max()) if total else 0.0,
                "imbalance_ratio": float(nonzero.max() / nonzero.min()) if len(nonzero) else 0.0,
            }
        )
    matrix = pd.DataFrame(matrix_rows, columns=[f"class_{int(cls)}" for cls in classes])
    matrix.insert(0, "client_id", range(len(partitions)))
    return PartitionStatistics(pd.DataFrame(stat_rows), matrix)

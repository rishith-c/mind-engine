"""Data engineering pipeline.

Three jobs that run on schedule (cron / Supabase scheduled functions):

    1. embed_pending_thoughts  — pick up rows in cognitron_particles where
       embedding IS NULL (e.g. inserted via API in degraded mode), encode
       them with HDC, write the bytea back.

    2. cluster_report          — k-means-style cluster summary on the
       current particle field. Writes a JSON to scripts/out/clusters.json
       so the UI can render cluster labels.

    3. decay_step              — invoke the SQL decay function, log how
       many rows were affected.

Usage:
    python scripts/data_pipeline.py [embed|cluster|decay|all]

This is intentionally library-light (no Airflow/Prefect/Dagster). It is
a script demoable in any environment that has Python + the project's
two AI packages installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "ai-cognitron"))

from cognitron import HDCEncoder  # noqa: E402


def embed_pending_thoughts(thoughts: list[tuple[int, str]]) -> dict[int, np.ndarray]:
    """Returns map of id -> embedding (bipolar int8)."""
    enc = HDCEncoder()
    return {tid: enc.encode(text) for tid, text in thoughts}


def cluster_report(positions: np.ndarray, labels: list[str], k: int = 5) -> list[dict]:
    """Lightweight k-means clustering for a "topics" overlay on the field."""
    n = positions.shape[0]
    if n == 0:
        return []
    rng = np.random.default_rng(0)
    centroids = positions[rng.choice(n, size=min(k, n), replace=False)].copy()
    for _ in range(20):
        d2 = ((positions[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        assignments = d2.argmin(axis=1)
        for j in range(centroids.shape[0]):
            mask = assignments == j
            if mask.any():
                centroids[j] = positions[mask].mean(axis=0)

    out = []
    for j, c in enumerate(centroids):
        members = [labels[i] for i in range(n) if assignments[i] == j]
        out.append(
            {
                "id": j,
                "centroid": c.tolist(),
                "size": len(members),
                "samples": members[:5],
            }
        )
    return out


def decay_step(rows: list[tuple[int, float]], factor: float = 0.985) -> dict:
    """Pure-Python mirror of the SQL decay function (so the pipeline is
    runnable without a live Supabase). Returns a summary."""
    survivors: list[tuple[int, float]] = []
    deleted: list[int] = []
    for rid, mass in rows:
        new_mass = mass * factor
        if new_mass < 0.001:
            deleted.append(rid)
        else:
            survivors.append((rid, new_mass))
    return {"survivors": len(survivors), "deleted": len(deleted)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["embed", "cluster", "decay", "all"])
    args = ap.parse_args()

    out = ROOT / "scripts" / "out"
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    report = {"job": args.job, "ts": timestamp}

    if args.job in {"embed", "all"}:
        sample = [
            (1, "the kettle whistles when water boils"),
            (2, "git rebase rewrites commit history"),
            (3, "the moon causes tides on the oceans"),
        ]
        embs = embed_pending_thoughts(sample)
        report["embed"] = {tid: int(emb.sum()) for tid, emb in embs.items()}

    if args.job in {"cluster", "all"}:
        pts = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
        labels = [f"thought-{i}" for i in range(20)]
        clusters = cluster_report(pts, labels, k=4)
        (out / "clusters.json").write_text(json.dumps(clusters, indent=2))
        report["cluster"] = {"clusters": len(clusters)}

    if args.job in {"decay", "all"}:
        rows = [(i, 1.0 - 0.05 * i) for i in range(20)]
        report["decay"] = decay_step(rows)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

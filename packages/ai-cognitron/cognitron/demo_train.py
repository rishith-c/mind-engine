"""End-to-end training smoke test for Cognitron.

Demonstrates that PGD can learn a meaningful particle layout: associate
"queries" with their semantically-correct memories so that a wave from the
query reaches the right memory first.

Loss design: for each (query, true-memory) pair, the loss is the negative
absorbed energy at the true memory plus a penalty proportional to absorbed
energy on incorrect memories. Minimising it physically pulls correct pairs
together and pushes incorrect pairs apart.

Run: python -m cognitron.demo_train

The analyst flagged "fake AI risk" — a from-scratch model that demos badly
is worse than no model. This script targets a task that is provably solvable
by particle motion alone and prints loss-down curves the demo can show live.
"""

from __future__ import annotations

import numpy as np

from cognitron.hdc import HDCEncoder
from cognitron.pgd import (
    ParticleGradientDescent,
    PGDConfig,
    SGDConfig,
    SGDFallback,
)
from cognitron.pnn import ParticleNetwork


# ---- toy dataset -----------------------------------------------------------

PAIRS = [
    ("how do I bake bread", "bread requires flour yeast water salt and an oven"),
    ("what is photosynthesis", "plants convert sunlight into sugar using chlorophyll"),
    ("rocket fuel chemistry", "kerosene and liquid oxygen combust producing thrust"),
    ("sql injection attack", "untrusted input concatenated into a query lets attackers execute arbitrary sql"),
    ("how to ferment kimchi", "salt cabbage with chili and let lactic bacteria do their work"),
    ("what causes ocean tides", "the moon's gravity pulls water creating bulges that rotate with earth"),
]


def build_loss(network: ParticleNetwork, query_ids: list[int], memory_ids: list[int]):
    """Negative correct-energy + half * incorrect-energy."""
    def loss(_net: ParticleNetwork) -> float:
        total = 0.0
        for q, true_mem in zip(query_ids, memory_ids):
            absorbed = _net.forward([q], energy=1.0)
            true_e = absorbed.get(true_mem, 0.0)
            wrong_e = sum(absorbed.get(m, 0.0) for m in memory_ids if m != true_mem)
            total += -true_e + 0.5 * wrong_e
        return total / max(len(query_ids), 1)

    return loss


def evaluate_top1(network: ParticleNetwork, query_ids: list[int], memory_ids: list[int]) -> float:
    correct = 0
    for q, true_mem in zip(query_ids, memory_ids):
        absorbed = network.forward([q], energy=1.0)
        candidates = sorted(
            ((m, absorbed.get(m, 0.0)) for m in memory_ids), key=lambda kv: kv[1], reverse=True
        )
        if candidates and candidates[0][0] == true_mem:
            correct += 1
    return correct / max(len(query_ids), 1)


def run(use_pgd: bool = True, steps: int = 40) -> None:
    enc = HDCEncoder(dim=2048, use_ngrams=True, ngram_size=2)
    net = ParticleNetwork(embedding_dim=2048, radius=0.5, propagation_steps=4)

    query_ids: list[int] = []
    memory_ids: list[int] = []
    for q, m in PAIRS:
        q_p = net.add_particle(enc.encode(q), text=q, mass=1.0)
        m_p = net.add_particle(enc.encode(m), text=m, mass=1.0)
        query_ids.append(q_p.id)
        memory_ids.append(m_p.id)

    loss_fn = build_loss(net, query_ids, memory_ids)
    optim_name = "PGD" if use_pgd else "SGD-fallback"
    if use_pgd:
        opt = ParticleGradientDescent(net, loss_fn, PGDConfig(diffusion=0.01))
    else:
        opt = SGDFallback(net, loss_fn, SGDConfig())

    print(f"== Cognitron training demo  ({optim_name}) ==")
    print(f"   particles={len(net)}  pairs={len(PAIRS)}")
    initial_acc = evaluate_top1(net, query_ids, memory_ids)
    print(f"   initial top-1 accuracy = {initial_acc:.2%}")

    for step in range(steps):
        loss = opt.step()
        if step % 5 == 0 or step == steps - 1:
            acc = evaluate_top1(net, query_ids, memory_ids)
            print(f"   step {step:3d}  loss={loss:+.4f}  top-1={acc:.2%}")

    final_acc = evaluate_top1(net, query_ids, memory_ids)
    print(f"   FINAL top-1 accuracy = {final_acc:.2%}")
    if final_acc < initial_acc:
        print("   WARNING: training regressed — consider switching to SGDFallback.")


if __name__ == "__main__":
    run(use_pgd=True, steps=30)

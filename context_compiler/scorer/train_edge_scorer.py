"""
Phase 5: Train the edge scorer.

Trains on the edge-level examples from build_edge_training_data.py. At the
scale this project can realistically generate data (a handful of
auto-generated tasks — see that file's own caveats), this is a smoke test
of the training mechanism, not a claim of a well-trained model. Scaling up
means more tasks -> ablation_engine.py -> more training examples, which is
itself bounded by ablation runtime (still fast, since the proxy is
structural, not a real agent rerun).

Usage:
    python -m context_compiler.scorer.train_edge_scorer edge_training_data.jsonl --epochs 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

from context_compiler.scorer.edge_scorer import EdgeScorer, build_task_vocab, featurize_example, collate_edge_batch


def load_examples(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def train_val_split(examples: list[dict], val_frac: float = 0.2, seed: int = 0) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def run_epoch(model, examples, task_vocab, optimizer, device, batch_size, train: bool, pos_weight: torch.Tensor):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for i in range(0, len(examples), batch_size):
            batch_raw = examples[i:i + batch_size]
            featurized = [featurize_example(ex, task_vocab) for ex in batch_raw]
            batch = {k: v.to(device) for k, v in collate_edge_batch(featurized).items()}

            logits = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["label"], pos_weight=pos_weight
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch_raw)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].long().cpu().tolist())

    avg_loss = total_loss / len(examples)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="path to edge_training_data.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="edge_scorer.pt")
    args = parser.parse_args()

    examples = load_examples(args.data)
    if len(examples) < 10:
        print(f"WARNING: only {len(examples)} examples — this is a mechanism smoke test, "
              f"not enough data for a model that generalizes. Generate more tasks first.")

    train_examples, val_examples = train_val_split(examples)
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")

    task_vocab = build_task_vocab([ex["task_description"] for ex in examples])
    print(f"Task vocab size: {len(task_vocab)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeScorer(task_vocab_size=len(task_vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # class imbalance is expected (most candidate edges are padding, not
    # essential — see build_edge_training_data.py's own label-balance
    # printout) — weight the positive class up to compensate
    n_pos = sum(ex["label"] for ex in train_examples)
    n_neg = len(train_examples) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)

    best_val_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_examples, task_vocab, optimizer, device, args.batch_size, train=True, pos_weight=pos_weight
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_examples, task_vocab, optimizer, device, args.batch_size, train=False, pos_weight=pos_weight
        )
        print(f"Epoch {epoch:02d} | train loss {train_loss:.3f} acc {train_acc:.2f} f1 {train_f1:.2f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.2f} f1 {val_f1:.2f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save({"state_dict": best_state, "task_vocab": task_vocab}, args.out)
    print(f"\nBest val F1: {best_val_f1:.2f} — saved to {args.out}")


if __name__ == "__main__":
    main()

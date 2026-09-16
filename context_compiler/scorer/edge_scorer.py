"""
Phase 5: Edge Scorer.

A small MLP-based binary classifier that predicts whether traversing a
particular (frontier_node, edge_type, candidate_node) triple is "worth it"
for a given task description.

Input features (per example):
  - task description -> bag-of-words vector over a fixed vocabulary
  - frontier node type (one-hot over: function/method/class/file/test)
  - edge type (one-hot over: imports/calls/inherits/defines/test_covers/co_change)
  - candidate node type (same one-hot as frontier)
  - lexical overlap between candidate name and task (scalar)

This is deliberately small enough to train in seconds on the ~100-example
datasets this project generates (see build_edge_training_data.py). It's a
proof-of-mechanism, not an architecture choice — the design doc envisions
eventually training on thousands of examples from dozens of repos, at
which point a bigger model (or a GNN over the local subgraph) would be
warranted.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from context_compiler.compiler.seed_retrieval import _tokenize  # reuse the same tokenizer as the lexical baseline


NODE_TYPES = ["function", "method", "class", "file", "test", "external", "?"]
EDGE_TYPES = ["imports", "calls", "inherits", "defines", "test_covers", "co_change", "unknown"]


def _onehot(value: str, categories: list[str]) -> list[float]:
    idx = categories.index(value) if value in categories else len(categories) - 1
    vec = [0.0] * len(categories)
    vec[idx] = 1.0
    return vec


def build_task_vocab(task_descriptions: list[str], max_vocab: int = 200) -> dict[str, int]:
    from collections import Counter
    counts: Counter = Counter()
    for desc in task_descriptions:
        counts.update(_tokenize(desc))
    vocab = {word: i for i, (word, _) in enumerate(counts.most_common(max_vocab))}
    return vocab


def featurize_example(example: dict, task_vocab: dict[str, int]) -> dict[str, torch.Tensor]:
    task_bow = [0.0] * len(task_vocab)
    for token in _tokenize(example["task_description"]):
        if token in task_vocab:
            task_bow[task_vocab[token]] = 1.0

    features = (
        task_bow
        + _onehot(example["frontier_node_type"], NODE_TYPES)
        + _onehot(example["edge_type"], EDGE_TYPES)
        + _onehot(example["candidate_node_type"], NODE_TYPES)
        + [float(example["candidate_name_task_overlap"])]
    )
    return {
        "features": torch.tensor(features, dtype=torch.float32),
        "label": torch.tensor(float(example["label"]), dtype=torch.float32),
    }


def collate_edge_batch(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.stack([e["features"] for e in examples]),
        "label": torch.stack([e["label"] for e in examples]),
    }


class EdgeScorer(nn.Module):
    def __init__(self, task_vocab_size: int = 200, hidden: int = 64):
        super().__init__()
        input_dim = task_vocab_size + len(NODE_TYPES) + len(EDGE_TYPES) + len(NODE_TYPES) + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["features"]).squeeze(-1)

"""
Phase 6: Feedback Loop.

The actual closed-loop experiment from design.md 4.7 / HANDOFF.md Phase 6:

  1. Run baseline (fixed-depth) compiler on auto-generated tasks, measure
     precision/recall against ablation-derived labels.
  2. Build edge-level training data from the baseline's selections.
  3. Train the edge scorer (Phase 5).
  4. Rerun the compiler with the scorer (score-guided expansion) on the same
     tasks, measure again.
  5. RETRAIN: regenerate training data *from the scorer's own selections*
     (not the baseline's), retrain, measure a third time.

Step 5 is the "feedback loop" proper: the scorer's training data shifts
each round because *its own choices* feed back into what gets ablated. The
hypothesis (confirmed, see BENCHMARK.md) is that this self-corrective
signal pushes precision/recall upward over rounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

from context_compiler.graph.code_graph import build_code_graph
from context_compiler.parser.repo_parser import parse_repo
from context_compiler.compiler.context_compiler import compile_context
from context_compiler.compiler.score_guided_expansion import load_edge_scorer
from context_compiler.ablation.tasks import auto_generate_tasks
from context_compiler.ablation.ablation_engine import run_ablation


def evaluate_strategy(graph, repo_root, tasks, scorer=None, task_vocab=None, token_budget=2000, label=""):
    precisions, recalls, feasible_count = [], [], 0
    for task in tasks:
        bundle = compile_context(
            graph, repo_root, task.task_description, token_budget=token_budget,
            scorer=scorer, task_vocab=task_vocab,
        )
        result = run_ablation(graph, repo_root, task, bundle)

        precisions.append(result.precision)
        recalls.append(result.recall)
        feasible_count += int(result.feasible)

    n = len(tasks)
    print(f"\n[{label}] feasible: {feasible_count}/{n}  "
          f"mean precision: {sum(precisions)/n:.2f}  "
          f"mean recall: {sum(recalls)/n:.2f}")
    return precisions, recalls, feasible_count


if __name__ == "__main__":
    from context_compiler.scorer.build_edge_training_data import build_and_save_dataset
    from context_compiler.scorer.train_edge_scorer import load_examples, train_val_split, run_epoch, EdgeScorer
    import torch

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    g = build_code_graph(parse_repo(repo))
    tasks = auto_generate_tasks(g, min_dependencies=2, max_tasks=10)

    # --- Step 1: baseline evaluation ---
    print("=== Step 1: Baseline (fixed-depth) ===")
    evaluate_strategy(g, repo, tasks, label="baseline")

    # --- Step 2+3: build training data from baseline, train scorer ---
    data_path = "./edge_training_data.jsonl"
    n = build_and_save_dataset(g, repo, tasks, data_path)
    print(f"\nBuilt {n} training examples from baseline selections")

    # inline training (same logic as train_edge_scorer.py, without argparse)
    from context_compiler.scorer.edge_scorer import build_task_vocab, featurize_example, collate_edge_batch
    examples = load_examples(data_path)
    task_vocab = build_task_vocab([ex["task_description"] for ex in examples])
    train_examples, val_examples = train_val_split(examples)
    device = torch.device("cpu")
    model = EdgeScorer(task_vocab_size=len(task_vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    n_pos = sum(ex["label"] for ex in train_examples)
    n_neg = len(train_examples) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)

    print(f"\nTraining scorer (round 1): {len(train_examples)} train, {len(val_examples)} val")
    best_f1, best_state = -1.0, None
    for epoch in range(1, 21):
        _, _, _ = run_epoch(model, train_examples, task_vocab, optimizer, device, 16, True, pos_weight)
        _, _, vf1 = run_epoch(model, val_examples, task_vocab, optimizer, device, 16, False, pos_weight)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "task_vocab": task_vocab}, "./edge_scorer.pt")
    print(f"Round 1 best val F1: {best_f1:.2f}")

    # --- Step 4: evaluate with scorer ---
    print("\n=== Step 4: Score-guided (round 1) ===")
    evaluate_strategy(g, repo, tasks, scorer=model, task_vocab=task_vocab, label="score-guided-r1")

    # --- Step 5: retrain from scorer's own selections ---
    data_path_v2 = "./edge_training_data_v2.jsonl"
    n2 = build_and_save_dataset(g, repo, tasks, data_path_v2, scorer=model, task_vocab=task_vocab)
    print(f"\nRebuilt {n2} training examples from scorer's own selections")

    examples_v2 = load_examples(data_path) + load_examples(data_path_v2)
    task_vocab_v2 = build_task_vocab([ex["task_description"] for ex in examples_v2])

    train_v2, val_v2 = train_val_split(examples_v2)
    model_v2 = EdgeScorer(task_vocab_size=len(task_vocab_v2)).to(device)
    optimizer_v2 = torch.optim.AdamW(model_v2.parameters(), lr=1e-3, weight_decay=0.01)
    n_pos_v2 = sum(ex["label"] for ex in train_v2)
    n_neg_v2 = len(train_v2) - n_pos_v2
    pos_weight_v2 = torch.tensor([n_neg_v2 / max(1, n_pos_v2)], device=device)

    print(f"\nTraining scorer (round 2): {len(train_v2)} train, {len(val_v2)} val")
    best_f1_v2, best_state_v2 = -1.0, None
    for epoch in range(1, 21):
        _, _, _ = run_epoch(model_v2, train_v2, task_vocab_v2, optimizer_v2, device, 16, True, pos_weight_v2)
        _, _, vf1 = run_epoch(model_v2, val_v2, task_vocab_v2, optimizer_v2, device, 16, False, pos_weight_v2)
        if vf1 > best_f1_v2:
            best_f1_v2 = vf1
            best_state_v2 = {k: v.clone() for k, v in model_v2.state_dict().items()}
    model_v2.load_state_dict(best_state_v2)
    torch.save({"state_dict": best_state_v2, "task_vocab": task_vocab_v2}, "./edge_scorer_v2.pt")
    print(f"Round 2 best val F1: {best_f1_v2:.2f}")

    # --- Final: evaluate retrained scorer ---
    print("\n=== Step 5: Score-guided (round 2, retrained) ===")
    evaluate_strategy(g, repo, tasks, scorer=model_v2, task_vocab=task_vocab_v2, label="score-guided-r2")

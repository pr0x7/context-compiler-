"""
Phase 7 support script: runs the ledger across all auto-generated tasks,
for both the fixed-depth baseline and the (round-2) score-guided scorer,
and aggregates token-cost-by-source stats. Not itself a new phase —
this exists purely to produce real numbers for docs/BENCHMARK.md rather
than have that writeup invent illustrative figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from context_compiler.graph.code_graph import build_code_graph
from context_compiler.parser.repo_parser import parse_repo
from context_compiler.compiler.context_compiler import compile_context
from context_compiler.ledger.minimal_agent import run_minimal_agent
from context_compiler.ledger.context_ledger import build_ledger_record
from context_compiler.ablation.tasks import auto_generate_tasks
from context_compiler.ablation.ablation_engine import run_ablation
from context_compiler.compiler.score_guided_expansion import load_edge_scorer


def run_strategy(g, repo_root, tasks, scorer=None, task_vocab=None, seed=0):
    total_compiler_tokens = 0
    total_agent_tokens = 0
    precisions, recalls, feasible_count = [], [], 0

    for i, task in enumerate(tasks):
        bundle = compile_context(
            g, repo_root, task.task_description, token_budget=2000,
            scorer=scorer, task_vocab=task_vocab,
        )
        agent_run = run_minimal_agent(g, repo_root, bundle, seed=seed + i)
        record = build_ledger_record(f"bench-{task.task_id}", bundle, agent_run)
        total_compiler_tokens += record.tokens_by_source["compiler"]
        total_agent_tokens += record.tokens_by_source["agent"]

        ablation = run_ablation(g, task, bundle)
        precisions.append(ablation.precision)
        recalls.append(ablation.recall)
        feasible_count += int(ablation.feasible)

    n = len(tasks)
    return {
        "feasible": feasible_count,
        "total_tasks": n,
        "mean_precision": sum(precisions) / n,
        "mean_recall": sum(recalls) / n,
        "total_compiler_tokens": total_compiler_tokens,
        "total_agent_tokens": total_agent_tokens,
        "mean_compiler_tokens_per_task": total_compiler_tokens / n,
        "mean_agent_tokens_per_task": total_agent_tokens / n,
    }


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    checkpoint_v1 = sys.argv[2] if len(sys.argv) > 2 else "./edge_scorer.pt"
    checkpoint_v2 = sys.argv[3] if len(sys.argv) > 3 else "./edge_scorer_v2.pt"
    out_path = sys.argv[4] if len(sys.argv) > 4 else "./benchmark_summary.json"

    g = build_code_graph(parse_repo(repo))
    tasks = auto_generate_tasks(g, min_dependencies=2, max_tasks=10)

    results = {"baseline": run_strategy(g, repo, tasks, seed=100)}

    scorer_v1, vocab_v1 = load_edge_scorer(checkpoint_v1)
    results["score_guided_round1"] = run_strategy(g, repo, tasks, scorer=scorer_v1, task_vocab=vocab_v1, seed=200)

    scorer_v2, vocab_v2 = load_edge_scorer(checkpoint_v2)
    results["score_guided_round2"] = run_strategy(g, repo, tasks, scorer=scorer_v2, task_vocab=vocab_v2, seed=300)

    Path(out_path).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")

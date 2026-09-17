"""
Real LLM Benchmark Campaign Script.

Runs the Real LLM Agent Harness (Gemini 3.6 Flash) across all auto-generated
benchmark tasks for both fixed-depth baseline expansion and score-guided expansion.

Saves full ledger records into `llm_benchmark_ledger.jsonl` and prints
an aggregate token-cost & precision/recall evaluation table.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from context_compiler.ablation.ablation_engine import run_ablation
from context_compiler.ablation.tasks import auto_generate_tasks
from context_compiler.compiler.context_compiler import compile_context
from context_compiler.compiler.score_guided_expansion import load_edge_scorer
from context_compiler.graph.code_graph import build_code_graph
from context_compiler.ledger.context_ledger import append_to_ledger, build_ledger_record
from context_compiler.ledger.llm_agent import run_llm_agent
from context_compiler.parser.repo_parser import parse_repo


def run_benchmark_strategy(
    g: Any,
    repo_root: str,
    tasks: list[Any],
    strategy_name: str,
    scorer: Any = None,
    task_vocab: dict | None = None,
    out_ledger_path: str = "./llm_benchmark_ledger.jsonl",
    seed: int = 42,
) -> dict[str, Any]:
    """Runs llm_agent across all tasks for a specific compiler strategy."""
    total_compiler_tokens = 0
    total_agent_tokens = 0
    precisions = []
    recalls = []
    feasible_count = 0
    records = []

    print(f"\n--- Running Real LLM Benchmark ({strategy_name}) across {len(tasks)} tasks ---")
    for i, task in enumerate(tasks):
        print(f" Task [{i+1}/{len(tasks)}]: {task.task_id} - '{task.task_description}'")
        bundle = compile_context(
            g,
            repo_root,
            task.task_description,
            token_budget=2000,
            scorer=scorer,
            task_vocab=task_vocab,
        )

        agent_run = run_llm_agent(g, repo_root, bundle, seed=seed + i)
        record = build_ledger_record(
            f"bench-{strategy_name}-{task.task_id}", bundle, agent_run
        )
        append_to_ledger(record, out_ledger_path)
        records.append(record)


        total_compiler_tokens += record.tokens_by_source.get("compiler", 0)
        total_agent_tokens += record.tokens_by_source.get("agent", 0)

        ablation = run_ablation(g, repo_root, task, bundle)

        precisions.append(ablation.precision)
        recalls.append(ablation.recall)
        feasible_count += int(ablation.feasible)

        print(
            f"   -> Compiler tokens: {record.tokens_by_source.get('compiler', 0):4d} | "
            f"Gemini agent tokens: {record.tokens_by_source.get('agent', 0):4d} | "
            f"Extra pulls: {len(agent_run.tool_calls)}"
        )

    n = len(tasks)
    return {
        "strategy": strategy_name,
        "feasible": feasible_count,
        "total_tasks": n,
        "mean_precision": sum(precisions) / n if n else 0.0,
        "mean_recall": sum(recalls) / n if n else 0.0,
        "total_compiler_tokens": total_compiler_tokens,
        "total_agent_tokens": total_agent_tokens,
        "mean_compiler_tokens_per_task": total_compiler_tokens / n if n else 0.0,
        "mean_agent_tokens_per_task": total_agent_tokens / n if n else 0.0,
    }


def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "./edge_scorer.pt"
    out_ledger_path = sys.argv[3] if len(sys.argv) > 3 else "./llm_benchmark_ledger.jsonl"

    print(f"Building code graph for repo at '{repo_root}'...")
    g = build_code_graph(parse_repo(repo_root))
    tasks = auto_generate_tasks(g, min_dependencies=2, max_tasks=10)
    print(f"Generated {len(tasks)} benchmark tasks.")

    results = []

    # 1. Fixed-Depth Baseline Strategy
    baseline_stats = run_benchmark_strategy(
        g, repo_root, tasks, strategy_name="fixed_depth", out_ledger_path=out_ledger_path
    )
    results.append(baseline_stats)

    # 2. Score-Guided Strategy (if model checkpoint exists)
    if os.path.exists(checkpoint_path):
        scorer, task_vocab = load_edge_scorer(checkpoint_path)
        score_guided_stats = run_benchmark_strategy(
            g,
            repo_root,
            tasks,
            strategy_name="score_guided",
            scorer=scorer,
            task_vocab=task_vocab,
            out_ledger_path=out_ledger_path,
        )
        results.append(score_guided_stats)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("                REAL LLM AGENT BENCHMARK EVALUATION TABLE               ")
    print("=" * 80)
    print(
        f"{'Strategy':<15} | {'Feasible':<9} | {'Precision':<10} | {'Recall':<8} | "
        f"{'Compiler Tok/Task':<18} | {'Gemini Agent Tok/Task':<20}"
    )
    print("-" * 80)
    for res in results:
        print(
            f"{res['strategy']:<15} | {res['feasible']}/{res['total_tasks']:<7} | "
            f"{res['mean_precision']:<10.2f} | {res['mean_recall']:<8.2f} | "
            f"{res['mean_compiler_tokens_per_task']:<18.1f} | "
            f"{res['mean_agent_tokens_per_task']:<20.1f}"
        )
    print("=" * 80)
    print(f"Raw run records saved to: {out_ledger_path}")


if __name__ == "__main__":
    main()

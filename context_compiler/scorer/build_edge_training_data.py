"""
Phase 5: Edge-level training data.

Combines Phase 4's ablation labels with Phase 2's edge-tracking expansion
to produce supervised examples of the form:

    (frontier_node, edge_type, candidate_node, task_description) -> label

label = 1 if candidate_node ended up in the ablation-derived minimal
context (i.e. it was actually needed), 0 if it was selected by the
compiler but ablated away as padding.

CAVEAT (inherits proxy_check.py's caveat): since Phase 4's ground truth is
itself derived from the graph's own calls/inherits edges, this training
signal teaches the scorer to predict "is this a direct call/inherit
dependency of the target" — a real but narrow notion of relevance. It does
NOT teach the scorer anything about co_change or import relevance beyond
what direct dependency happens to correlate with. Worth strengthening once
(if) a less circular proxy (e.g. pyflakes-based) exists — see
ablation/proxy_check.py.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

from context_compiler.compiler.expansion import fixed_depth_expansion_with_edges
from context_compiler.compiler.seed_retrieval import lexical_seed_retrieval
from context_compiler.compiler.context_compiler import compile_context
from context_compiler.ablation.tasks import AblationTask, auto_generate_tasks
from context_compiler.ablation.ablation_engine import run_ablation


@dataclass
class EdgeExample:
    task_description: str
    frontier_node_type: str
    edge_type: str
    candidate_node_type: str
    candidate_name_task_overlap: int   # cheap lexical feature, same idea as seed_retrieval's scoring
    label: int   # 1 = needed (in ablation-derived minimal context), 0 = padding


def _lexical_overlap(graph: nx.MultiDiGraph, node_id: str, task_description: str) -> int:
    # reuse seed_retrieval's tokenizer for a consistent notion of "overlap"
    from context_compiler.compiler.seed_retrieval import _tokenize
    task_tokens = set(_tokenize(task_description))
    node_tokens = set(_tokenize(node_id))
    return len(task_tokens & node_tokens)


def build_edge_examples(
    graph: nx.MultiDiGraph, repo_root: str, task: AblationTask, token_budget: int = 2000,
    scorer=None, task_vocab: dict | None = None,
) -> list[EdgeExample]:
    """
    scorer/task_vocab: if given, compiles the bundle using the current
    score-guided expansion (Phase 5/6) instead of the fixed-depth baseline
    — this is what lets feedback_loop.py regenerate training data from the
    scorer's *own* selections for the next retraining round, per
    docs/HANDOFF.md's Phase 6 steps.
    """
    bundle = compile_context(
        graph, repo_root, task.task_description, token_budget=token_budget,
        scorer=scorer, task_vocab=task_vocab,
    )
    ablation_result = run_ablation(graph, repo_root, task, bundle)

    minimal_set = set(ablation_result.minimal_context)

    seeds = [n for n, _ in lexical_seed_retrieval(graph, task.task_description, top_k=8)]
    reached = fixed_depth_expansion_with_edges(graph, seeds, depth=2)

    examples: list[EdgeExample] = []
    for node_id, edge in reached.items():
        if node_id not in {s.node_id for s in bundle.selected}:
            # only edges that actually contributed to the compiler's
            # selection have a meaningful label from this ablation run
            continue
        label = 1 if node_id in minimal_set else 0
        examples.append(EdgeExample(
            task_description=task.task_description,
            frontier_node_type=graph.nodes.get(edge.parent, {}).get("node_type", "?"),
            edge_type=edge.edge_type,
            candidate_node_type=graph.nodes.get(node_id, {}).get("node_type", "?"),
            candidate_name_task_overlap=_lexical_overlap(graph, node_id, task.task_description),
            label=label,
        ))
    return examples


def build_and_save_dataset(
    graph: nx.MultiDiGraph, repo_root: str, tasks: list[AblationTask], out_path: str,
    scorer=None, task_vocab: dict | None = None,
) -> int:
    all_examples: list[EdgeExample] = []
    for task in tasks:
        all_examples.extend(build_edge_examples(graph, repo_root, task, scorer=scorer, task_vocab=task_vocab))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    return len(all_examples)


if __name__ == "__main__":
    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    out_path = sys.argv[2] if len(sys.argv) > 2 else "./edge_training_data.jsonl"

    g = build_code_graph(parse_repo(repo))
    tasks = auto_generate_tasks(g, min_dependencies=2, max_tasks=10)
    n = build_and_save_dataset(g, repo, tasks, out_path)

    print(f"Built {n} edge-level training examples from {len(tasks)} tasks")
    print(f"Saved to {out_path}")

    labels = [json.loads(l)["label"] for l in Path(out_path).read_text().splitlines() if l.strip()]
    pos = sum(labels)
    print(f"Label balance: {pos} positive / {len(labels) - pos} negative ({pos/len(labels):.1%} positive)")

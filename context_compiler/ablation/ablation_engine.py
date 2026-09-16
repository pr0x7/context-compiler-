"""
Phase 4: Ablation Engine.

Greedy removal heuristic (per design.md 4.6, explicitly preferred over
exhaustive subset search): starting from a compiled ContextBundle, repeatedly
try dropping the lowest-scored remaining item and rerun the proxy check
(see proxy_check.py and its caveat about what this actually measures).
If the check still passes without that item, it's genuinely dropped; if not,
it's restored. What's left when every item has been tried is the
minimal-sufficient-context label for that task.

Also computes precision/recall of the *original* compiler selection against
that minimal label — design.md 4.7's "precision/recall of compiler
selection against the minimal-sufficient-context label" eval metric. This
is the number that actually tells you whether Phase 2's baseline compiler
is over-including (low precision) or under-including (low recall)
relative to what the ablation says was truly needed.

Note: because the proxy check's "required dependencies" are a subset of
what score-guided/fixed-depth expansion would include anyway, the minimal
set found here will typically undershoot what a real task would need in
practice (e.g. it won't require test files or usage examples the agent
might genuinely want). Read the precision/recall numbers with that caveat
in mind — see proxy_check.py's module docstring for the full picture.
Note: because "required dependencies" is a fixed, deterministic set derived
directly from the graph (not discovered through search), the greedy-removal
loop below is somewhat degenerate — it will always converge to exactly
{target} ∪ required_dependencies when the compiler's selection covers that
set. Its real value isn't "discovering" the minimal set (that's already
known in closed form) — it's confirming which of the *specific* nodes the
compiler chose were essential vs. droppable padding, which does need the
actual selection to check. Read the precision/recall numbers as "how much
of what the compiler picked was necessary" and "how much of what was
necessary did the compiler pick" — not as evidence of a non-trivial search
process.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from context_compiler.ablation.tasks import AblationTask
from context_compiler.ablation.proxy_check import check_context_sufficient
from context_compiler.compiler.context_compiler import ContextBundle, compile_context


@dataclass
class AblationEvent:
    node_id: str
    kept: bool          # True if removal broke the check (so it was restored)
    order: int           # the order in which this removal was attempted


@dataclass
class AblationResult:
    task_id: str
    commit_hash: str
    target_node: str
    original_selection: list[str]
    minimal_context: list[str]
    feasible: bool       # False if the compiler's own selection never satisfied the proxy check
    events: list[AblationEvent] = field(default_factory=list)
    precision: float = 0.0   # of the original compiler selection, how much was actually needed
    recall: float = 0.0      # of what was actually needed, how much did the compiler select
    created_at: float = field(default_factory=time.time)


def run_ablation(graph: nx.MultiDiGraph, repo_root: str, task: AblationTask, bundle: ContextBundle) -> AblationResult:
    original_ids = {s.node_id for s in bundle.selected}
    theoretical_minimal = {task.target_node} | task.required_dependencies

    # the target itself must always be present for the task to make sense —
    # force it in if the compiler's seed retrieval didn't happen to select it
    working_set = set(original_ids)
    working_set.add(task.target_node)

    feasible = check_context_sufficient(graph, repo_root, task, working_set)
    events: list[AblationEvent] = []

    if feasible:
        # sort by ascending compiler score so lowest-value items get tried
        # for removal first
        scores = {s.node_id: s.score for s in bundle.selected}
        removable_order = sorted(
            (n for n in working_set if n != task.target_node),
            key=lambda n: scores.get(n, 0.0),
        )
        for i, node_id in enumerate(removable_order):
            trial_set = working_set - {node_id}
            if check_context_sufficient(graph, repo_root, task, trial_set):
                working_set = trial_set
                events.append(AblationEvent(node_id=node_id, kept=False, order=i))
            else:
                events.append(AblationEvent(node_id=node_id, kept=True, order=i))
        minimal = working_set
    else:
        # can't meaningfully ablate down from a set that's already
        # insufficient — report the true requirement instead, so
        # precision/recall still mean something rather than being
        # hardcoded to zero
        minimal = theoretical_minimal

    # always measure against the true requirement (theoretical_minimal),
    # not the ablation-derived `minimal`, so the metric is well-defined and
    # comparable across both the feasible and infeasible branches
    overlap = original_ids & theoretical_minimal
    precision = len(overlap) / len(original_ids) if original_ids else 0.0
    recall = len(overlap) / len(theoretical_minimal) if theoretical_minimal else 1.0

    return AblationResult(
        task_id=task.task_id, commit_hash=graph.graph.get("commit_hash", "?"),
        target_node=task.target_node, original_selection=sorted(original_ids),
        minimal_context=sorted(minimal), feasible=feasible, events=events,
        precision=precision, recall=recall,
    )


def save_ablation_result(result: AblationResult, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def load_ablation_results(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo
    from context_compiler.ablation.tasks import auto_generate_tasks

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    out_path = sys.argv[2] if len(sys.argv) > 2 else "./ablation_labels.jsonl"
    token_budget = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    g = build_code_graph(parse_repo(repo))
    tasks = auto_generate_tasks(g, min_dependencies=2, max_tasks=10)
    print(f"Running ablation on {len(tasks)} auto-generated tasks...\n")

    precisions, recalls = [], []
    for task in tasks:
        bundle = compile_context(g, repo, task.task_description, token_budget=token_budget)
        result = run_ablation(g, repo, task, bundle)
        save_ablation_result(result, out_path)

        precisions.append(result.precision)
        recalls.append(result.recall)
        feas = "yes" if result.feasible else "NO "
        print(f"{task.task_id:30s} feasible={feas} orig={len(result.original_selection):3d} "
              f"minimal={len(result.minimal_context):3d}  "
              f"precision={result.precision:.2f}  recall={result.recall:.2f}")

    if precisions:
        print(f"\nMean precision: {sum(precisions)/len(precisions):.2f}")
        print(f"Mean recall:    {sum(recalls)/len(recalls):.2f}")
    print(f"\nLabels saved to {out_path}")

"""
Phase 4: Ablation tasks.

Design.md 4.6/HANDOFF.md require "a small task set" to run ablation
against. Rather than hand-authoring tasks (real work, and arguably a
separate project), tasks here are auto-generated from the graph itself:
pick symbols that have a meaningful number of internal dependencies (calls
to other symbols defined in this repo), and treat "understand/modify this
symbol" as the task, described in natural language for seed retrieval to
work off of.

This is a deliberate scope choice — see the caveat in
ablation_engine.py's module docstring about what this proxy actually
measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass
class AblationTask:
    task_id: str
    task_description: str
    target_node: str
    required_dependencies: set[str]  # node_ids the target directly calls/inherits (internal only)


def required_dependencies_of(graph: nx.MultiDiGraph, node_id: str) -> set[str]:
    """
    Direct internal dependencies of a symbol: everything it `calls` or
    `inherits` from that resolved to a real symbol in this repo (edges to
    unresolved/external names were never added to the graph in the first
    place — see graph/code_graph.py).
    """
    deps = set()
    for _, target, data in graph.out_edges(node_id, data=True):
        if data.get("edge_type") in ("calls", "inherits"):
            deps.add(target)
    return deps


def auto_generate_tasks(
    graph: nx.MultiDiGraph, min_dependencies: int = 2, max_tasks: int = 10,
) -> list[AblationTask]:
    """
    Symbols with >= min_dependencies internal calls/inherits make
    reasonable ablation targets — enough structure for removal to matter,
    but nothing too sprawling. Task description is auto-phrased from the
    symbol's own (qualified) name so seed_retrieval has something to match.
    """
    candidates: list[tuple[str, set[str]]] = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") not in ("function", "method"):
            continue
        deps = required_dependencies_of(graph, node_id)
        if len(deps) >= min_dependencies:
            candidates.append((node_id, deps))

    # prefer symbols with more dependencies first — more interesting ablations
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    tasks = []
    for i, (node_id, deps) in enumerate(candidates[:max_tasks]):
        # for methods, include the parent class in the description too —
        # "understand and modify init" is useless for seed retrieval when
        # ten classes all have an __init__
        parts = node_id.split(".")
        short_name = parts[-1]
        readable_name = short_name.replace("_", " ").strip()
        if attrs_kind(graph, node_id) == "method" and len(parts) >= 2:
            readable_name = f"{parts[-2]} {readable_name}".strip()

        tasks.append(AblationTask(
            task_id=f"auto-{i:03d}-{short_name}",
            task_description=f"understand and modify {readable_name}",
            target_node=node_id,
            required_dependencies=deps,
        ))
    return tasks


def attrs_kind(graph: nx.MultiDiGraph, node_id: str) -> str:
    return graph.nodes[node_id].get("node_type", "")


if __name__ == "__main__":
    import sys

    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    g = build_code_graph(parse_repo(repo))
    tasks = auto_generate_tasks(g)

    print(f"Generated {len(tasks)} tasks:")
    for t in tasks:
        print(f"  {t.task_id}: target={t.target_node}  deps={len(t.required_dependencies)}")

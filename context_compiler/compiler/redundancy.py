"""
Phase 2: Redundancy removal.

Baseline-scope redundancy handling (design.md 4.3 calls for more — dedup
overlapping spans, collapse near-identical call sites — but those need the
learned scorer's notion of "near-identical" to do well; this is the cheap,
structural version that's still worth doing before token-budget optimization):

  - If a whole file is a candidate AND one or more specific symbols from
    that same file are also candidates, drop the whole-file entry. The
    specific symbols are more targeted and the file's docstring-level
    content is generally redundant once you're including its functions
    directly.

Deliberately NOT collapsed: a class and one of its own methods both being
selected. That's not redundancy — a method's context often only makes
sense next to its class definition — so both are kept.
"""

from __future__ import annotations

import networkx as nx


def remove_redundant(graph: nx.MultiDiGraph, candidate_scores: dict[str, float]) -> dict[str, float]:
    files_with_symbols_selected: set[str] = set()
    for node_id in candidate_scores:
        attrs = graph.nodes.get(node_id, {})
        if attrs.get("node_type") in ("function", "method", "class"):
            files_with_symbols_selected.add(attrs.get("file_path"))

    filtered = {}
    dropped = []
    for node_id, score in candidate_scores.items():
        attrs = graph.nodes.get(node_id, {})
        if attrs.get("node_type") in ("file", "test") and node_id in files_with_symbols_selected:
            dropped.append(node_id)
            continue
        filtered[node_id] = score

    if dropped:
        print(f"[redundancy] dropped {len(dropped)} whole-file entries superseded by specific symbols: {dropped}")

    return filtered

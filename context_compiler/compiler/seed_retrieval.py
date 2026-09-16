"""
Phase 2: Seed retrieval.

Cheap lexical match between a task description and node identifiers/
docstrings, to produce an initial candidate set (the "entry points" that
fixed-depth expansion then grows outward from).

This is deliberately dumb (token overlap, no embeddings) — that's the
point of a *baseline* compiler. Phase 5 replaces/augments this with a
learned scorer; this module gives you something to compare it against.
"""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Common English + code stopwords that shouldn't count as meaningful overlap.
_STOPWORDS = {
    "the", "a", "an", "in", "on", "for", "to", "of", "and", "or", "is", "are",
    "this", "that", "with", "as", "at", "by", "from", "it", "be", "should",
    "add", "fix", "update", "make", "def", "self", "return", "if", "else",
}


def _tokenize(text: str) -> list[str]:
    """Split identifiers on case/underscore boundaries too, e.g. `GCNBaseline` -> gcn, baseline."""
    words = _WORD_RE.findall(text)
    tokens: list[str] = []
    for w in words:
        # split snake_case
        parts = w.split("_")
        for p in parts:
            # split CamelCase within each part
            camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", p)
            tokens.extend(c.lower() for c in (camel_parts or [p]))
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


def lexical_seed_retrieval(
    graph: nx.MultiDiGraph, task_description: str, top_k: int = 10,
    node_types: tuple[str, ...] = ("function", "method", "class", "file"),
) -> list[tuple[str, float]]:
    """
    Score every eligible node by token overlap between the task description
    and (qualified name + docstring), return the top_k as (node_id, score).

    Score = weighted Jaccard-ish overlap: name-token matches count double
    docstring-token matches, since a name match is a stronger signal.
    """
    task_tokens = Counter(_tokenize(task_description))
    if not task_tokens:
        return []

    scored: list[tuple[str, float]] = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") not in node_types:
            continue

        name_tokens = Counter(_tokenize(node_id))
        doc_tokens = Counter(_tokenize(attrs.get("docstring", "") or ""))

        name_overlap = sum((task_tokens & name_tokens).values())
        doc_overlap = sum((task_tokens & doc_tokens).values())

        score = 2.0 * name_overlap + 1.0 * doc_overlap
        if score > 0:
            scored.append((node_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    import sys

    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else "graph transformer attention bias"

    g = build_code_graph(parse_repo(repo))
    results = lexical_seed_retrieval(g, query, top_k=10)
    print(f"Query: {query!r}")
    for node_id, score in results:
        print(f"  {score:.1f}  {node_id}")

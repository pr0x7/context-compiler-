"""
Phase 2: Token-budget optimization.

Greedy knapsack-style selection: maximize total relevance score within a
fixed token budget. True optimal knapsack is exact-DP-solvable at this
scale, but greedy-by-score-density (score/token) is the standard practical
approximation and is what "knapsack-style" means in design.md 4.3 — good
enough for a baseline, and fast.

Also handles reading the actual source text for a node (needed both to
count its tokens and to actually include it in the compiled bundle).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

# No tokenizer dependency for the baseline — chars/4 is the standard rough
# approximation for English/code text. Swap in a real tokenizer (tiktoken,
# the target model's own tokenizer) if you need exact counts later.
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def get_node_source(graph: nx.MultiDiGraph, repo_root: str, node_id: str) -> str:
    """
    Read the actual source text for a node:
      - function/method/class: the lines [lineno, end_lineno] of its file
      - file: the whole file's content
      - anything else (e.g. external): empty string
    """
    attrs = graph.nodes[node_id]
    node_type = attrs.get("node_type")

    if node_type == "file" or node_type == "test":
        file_path = node_id
    elif node_type in ("function", "method", "class"):
        file_path = attrs.get("file_path")
    else:
        return ""

    full_path = Path(repo_root) / file_path
    if not full_path.exists():
        return ""

    text = full_path.read_text(encoding="utf-8", errors="replace")
    if node_type in ("function", "method", "class"):
        lines = text.splitlines()
        start = max(0, attrs.get("lineno", 1) - 1)
        end = min(len(lines), attrs.get("end_lineno", start + 1))
        return "\n".join(lines[start:end])
    return text


@dataclass
class SelectedNode:
    node_id: str
    score: float
    tokens: int
    source: str


def select_within_budget(
    graph: nx.MultiDiGraph,
    repo_root: str,
    scored_candidates: dict[str, float],
    token_budget: int,
) -> list[SelectedNode]:
    """
    Greedy fill: sort candidates by score/token density descending, take
    each until the budget runs out. Returns selections in that same
    density-descending order (a reasonable default read order — most
    "bang for buck" context first).
    """
    scored_with_tokens: list[tuple[str, float, int, str]] = []
    for node_id, score in scored_candidates.items():
        source = get_node_source(graph, repo_root, node_id)
        if not source.strip():
            continue
        tokens = estimate_tokens(source)
        scored_with_tokens.append((node_id, score, tokens, source))

    # density = score per token — this is the greedy knapsack heuristic
    scored_with_tokens.sort(key=lambda x: x[1] / x[2], reverse=True)

    selected: list[SelectedNode] = []
    remaining = token_budget
    for node_id, score, tokens, source in scored_with_tokens:
        if tokens <= remaining:
            selected.append(SelectedNode(node_id, score, tokens, source))
            remaining -= tokens

    return selected


if __name__ == "__main__":
    import sys

    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo
    from context_compiler.compiler.seed_retrieval import lexical_seed_retrieval
    from context_compiler.compiler.expansion import fixed_depth_expansion

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else "graph transformer attention bias"
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    g = build_code_graph(parse_repo(repo))
    seeds = [n for n, _ in lexical_seed_retrieval(g, query, top_k=5)]
    distances = fixed_depth_expansion(g, seeds, depth=2)

    # naive score from distance alone, for this standalone smoke test
    # (context_compiler.py combines this properly with seed scores)
    candidate_scores = {n: 1.0 / (1 + d) for n, d in distances.items()}

    selected = select_within_budget(g, repo, candidate_scores, token_budget=budget)
    total_tokens = sum(s.tokens for s in selected)
    print(f"Selected {len(selected)}/{len(candidate_scores)} candidates, {total_tokens}/{budget} tokens")
    for s in selected:
        print(f"  {s.tokens:5d} tok  score={s.score:.2f}  {s.node_id}")

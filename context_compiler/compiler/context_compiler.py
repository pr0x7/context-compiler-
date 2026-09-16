"""
Phase 2/6: Context Compiler.

Wires together: seed retrieval -> expansion -> redundancy removal ->
token-budget optimization, per design.md section 4.3/7 phase 2.

As of Phase 6, expansion can be either the Phase 2 fixed-depth baseline
(default, no extra setup needed) or Phase 5's score-guided expansion (pass
`scorer`/`task_vocab`, loaded via score_guided_expansion.load_edge_scorer).
This is the actual "swap it in" step HANDOFF.md's Phase 6 section called
for — both paths produce the same ContextBundle shape, so nothing
downstream (the ledger, the ablation engine) needs to know which expansion
strategy produced a given bundle.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from context_compiler.graph.code_graph import build_code_graph
from context_compiler.parser.repo_parser import parse_repo
from context_compiler.compiler.seed_retrieval import lexical_seed_retrieval
from context_compiler.compiler.expansion import fixed_depth_expansion
from context_compiler.compiler.redundancy import remove_redundant
from context_compiler.compiler.token_budget import select_within_budget, SelectedNode


@dataclass
class ContextBundle:
    task_description: str
    commit_hash: str
    token_budget: int
    selected: list[SelectedNode] = field(default_factory=list)
    seed_nodes: list[str] = field(default_factory=list)
    expansion_strategy: str = "fixed_depth"   # "fixed_depth" | "score_guided" — logged so runs are comparable

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.selected)

    def render(self) -> str:
        """Concatenate selected source snippets into one context string, in selection order."""
        parts = []
        for s in self.selected:
            parts.append(f"# --- {s.node_id} (score={s.score:.2f}, ~{s.tokens} tok) ---\n{s.source}")
        return "\n\n".join(parts)


def compile_context(
    graph: nx.MultiDiGraph,
    repo_root: str,
    task_description: str,
    top_k_seeds: int = 8,
    expansion_depth: int = 2,
    token_budget: int = 4000,
    scorer=None,
    task_vocab: dict | None = None,
    max_expansion_nodes: int = 30,
) -> ContextBundle:
    """
    scorer/task_vocab: if both given (see score_guided_expansion.load_edge_scorer),
    uses Phase 5's learned best-first expansion instead of the Phase 2
    fixed-depth baseline. Leave both None for the original baseline behavior.
    """
    seed_results = lexical_seed_retrieval(graph, task_description, top_k=top_k_seeds)
    seed_nodes = [n for n, _ in seed_results]
    seed_scores = dict(seed_results)

    if not seed_nodes:
        return ContextBundle(task_description, graph.graph.get("commit_hash", "?"), token_budget, [], [])

    if scorer is not None and task_vocab is not None:
        from context_compiler.compiler.score_guided_expansion import score_guided_expansion
        reached = score_guided_expansion(
            graph, seed_nodes, task_description, scorer, task_vocab, max_nodes=max_expansion_nodes,
        )
        candidate_scores: dict[str, float] = dict(seed_scores)
        for node, r in reached.items():
            candidate_scores[node] = r.score
        strategy = "score_guided"
    else:
        distances = fixed_depth_expansion(graph, seed_nodes, depth=expansion_depth)

        # combine: seeds keep their lexical score; expanded nodes get a
        # distance-decayed score (this is the "fixed-depth" baseline's stand-in
        # for a learned relevance score — score_guided replaces this decay
        # function with the edge scorer's actual predicted relevance)
        candidate_scores = {}
        for node, hop in distances.items():
            if node in seed_scores:
                candidate_scores[node] = seed_scores[node]
            else:
                base = min(seed_scores.values())
                candidate_scores[node] = base / (1 + hop)
        strategy = "fixed_depth"

    candidate_scores = remove_redundant(graph, candidate_scores)
    selected = select_within_budget(graph, repo_root, candidate_scores, token_budget)

    return ContextBundle(
        task_description=task_description,
        commit_hash=graph.graph.get("commit_hash", "?"),
        token_budget=token_budget,
        selected=selected,
        seed_nodes=seed_nodes,
        expansion_strategy=strategy,
    )


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else "graph transformer attention bias"
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    scorer_checkpoint = sys.argv[4] if len(sys.argv) > 4 else None

    g = build_code_graph(parse_repo(repo))

    scorer, task_vocab = None, None
    if scorer_checkpoint:
        from context_compiler.compiler.score_guided_expansion import load_edge_scorer
        scorer, task_vocab = load_edge_scorer(scorer_checkpoint)

    bundle = compile_context(g, repo, query, token_budget=budget, scorer=scorer, task_vocab=task_vocab)

    print(f"Task: {query!r}  (strategy: {bundle.expansion_strategy})")
    print(f"Seeds: {bundle.seed_nodes}")
    print(f"Selected {len(bundle.selected)} nodes, {bundle.total_tokens}/{bundle.token_budget} tokens\n")
    for s in bundle.selected:
        print(f"  {s.tokens:5d} tok  score={s.score:.2f}  {s.node_id}")

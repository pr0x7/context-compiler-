"""
Phase 5: Score-guided expansion.

Replaces expansion.py's fixed_depth_expansion with a best-first traversal
driven by the trained EdgeScorer: at each step, expand the single
highest-scored untried (frontier_node, edge_type, candidate) triple,
rather than exhausting every neighbor at each hop. This is the actual
design.md 4.3 differentiator ("score-guided expansion ... not a
fixed-radius BFS-then-filter").

Same input/output shape as fixed_depth_expansion (seed nodes in, a
{node_id: <info>} map out) so it's a drop-in replacement in
context_compiler.py once you're ready to switch — see
docs/HANDOFF.md's "wiring point" note.

CAVEAT: the scorer was trained on the tiny smoke-test dataset from
build_edge_training_data.py (order of ~100 examples from 10
auto-generated tasks) — its rankings should not be trusted as "good" yet.
This module proves the architecture (learned scores driving traversal
order and a stopping condition) works end-to-end; making the rankings
actually good requires the larger training set noted in HANDOFF.md's
"What Phase 5 needs".
"""

from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch

from context_compiler.scorer.edge_scorer import EdgeScorer, featurize_example, collate_edge_batch


@dataclass
class ScoredReach:
    parent: str
    edge_type: str
    score: float


def load_edge_scorer(checkpoint_path: str) -> tuple[EdgeScorer, dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = EdgeScorer(task_vocab_size=len(ckpt["task_vocab"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["task_vocab"]


def _score_edge(model: EdgeScorer, task_vocab: dict, task_description: str,
                 frontier_type: str, edge_type: str, candidate_type: str, overlap: int) -> float:
    example = {
        "task_description": task_description,
        "frontier_node_type": frontier_type,
        "edge_type": edge_type,
        "candidate_node_type": candidate_type,
        "candidate_name_task_overlap": overlap,
        "label": 0,  # unused at inference
    }
    featurized = [featurize_example(example, task_vocab)]
    batch = collate_edge_batch(featurized)
    with torch.no_grad():
        logit = model(batch)
    return torch.sigmoid(logit).item()


def score_guided_expansion(
    graph: nx.MultiDiGraph,
    seed_nodes: list[str],
    task_description: str,
    model: EdgeScorer,
    task_vocab: dict[str, int],
    max_nodes: int = 20,
    score_threshold: float = 0.0,
) -> dict[str, ScoredReach]:
    """
    Best-first expansion: maintain a max-heap of (score, frontier, edge_type,
    candidate) triples for every unreached neighbor of every reached node,
    always expand the highest-scoring one next, and push its own new
    neighbors onto the heap. Stops at max_nodes reached, or when nothing on
    the heap scores above score_threshold.
    """
    from context_compiler.compiler.seed_retrieval import _tokenize  # local import to avoid a hard dependency at module load time

    undirected = graph.to_undirected(as_view=True)
    reached: dict[str, ScoredReach] = {}
    seed_set = {s for s in seed_nodes if s in undirected}

    def _overlap(node_id: str) -> int:
        task_tokens = set(_tokenize(task_description))
        return len(task_tokens & set(_tokenize(node_id)))

    # heap entries: (-score, tie_breaker, candidate, parent, edge_type)
    heap: list[tuple[float, int, str, str, str]] = []
    counter = 0

    def _push_neighbors(node: str):
        nonlocal counter
        node_type = undirected.nodes[node].get("node_type", "?")
        for neighbor in undirected.neighbors(node):
            if neighbor in reached or neighbor in seed_set:
                continue
            if undirected.nodes[neighbor].get("node_type") == "external":
                continue
            edge_datas = undirected.get_edge_data(node, neighbor) or {}
            types_here = {d.get("edge_type") for d in edge_datas.values()} or {"unknown"}
            candidate_type = undirected.nodes[neighbor].get("node_type", "?")
            for edge_type in types_here:
                score = _score_edge(model, task_vocab, task_description, node_type, edge_type,
                                     candidate_type, _overlap(neighbor))
                counter += 1
                heapq.heappush(heap, (-score, counter, neighbor, node, edge_type))

    for s in seed_set:
        _push_neighbors(s)

    while heap and len(reached) < max_nodes:
        neg_score, _, candidate, parent, edge_type = heapq.heappop(heap)
        score = -neg_score
        if score < score_threshold:
            break
        if candidate in reached:
            continue  # may have been reached via a different (lower-priority) heap entry already popped
        reached[candidate] = ScoredReach(parent=parent, edge_type=edge_type, score=score)
        _push_neighbors(candidate)

    return reached


if __name__ == "__main__":
    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo
    from context_compiler.compiler.seed_retrieval import lexical_seed_retrieval

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else "graph transformer attention bias"
    checkpoint = sys.argv[3] if len(sys.argv) > 3 else "./edge_scorer.pt"
    max_nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 15

    g = build_code_graph(parse_repo(repo))
    model, vocab = load_edge_scorer(checkpoint)
    seeds = [n for n, _ in lexical_seed_retrieval(g, query, top_k=5)]

    reached = score_guided_expansion(g, seeds, query, model, vocab, max_nodes=max_nodes)
    print(f"Score-guided expansion reached {len(reached)} nodes (max_nodes={max_nodes}):")
    for node, r in sorted(reached.items(), key=lambda x: -x[1].score):
        print(f"  {r.score:.3f}  {node}  (via {r.edge_type} from {r.parent})")

"""
Phase 2: Fixed-depth expansion.

Grows outward from seed nodes by a fixed number of hops, following graph
edges undirected (a caller and its callee are both "relevant context" for
each other, regardless of edge direction). This is the explicit strawman
baseline that Phase 5's score-guided expansion is meant to beat — see
design.md section 4.3 and the "fixed-depth graph BFS" baseline in 4.7.

`external` nodes (library imports) are never expanded through — they're
not real code we can include as context.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


def fixed_depth_expansion(
    graph: nx.MultiDiGraph,
    seed_nodes: list[str],
    depth: int = 2,
    edge_types: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """
    BFS outward from seed_nodes up to `depth` hops on the undirected version
    of the graph. Returns {node_id: hop_distance} for every reached node
    (seeds themselves get distance 0).

    edge_types: if given, only traverse edges whose 'edge_type' attribute is
    in this set (e.g. restrict expansion to just ("calls", "imports") and
    skip "co_change" if you want a stricter notion of "related").
    """
    undirected = graph.to_undirected(as_view=True)

    distances: dict[str, int] = {s: 0 for s in seed_nodes if s in undirected}
    frontier = set(distances.keys())

    for hop in range(1, depth + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in undirected.neighbors(node):
                if neighbor in distances:
                    continue
                if undirected.nodes[neighbor].get("node_type") == "external":
                    continue
                if edge_types is not None:
                    edge_datas = undirected.get_edge_data(node, neighbor) or {}
                    types_here = {d.get("edge_type") for d in edge_datas.values()}
                    if not (types_here & set(edge_types)):
                        continue
                distances[neighbor] = hop
                next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return distances


@dataclass
class ExpansionEdge:
    parent: str
    edge_type: str
    hop: int


def fixed_depth_expansion_with_edges(
    graph: nx.MultiDiGraph,
    seed_nodes: list[str],
    depth: int = 2,
    edge_types: tuple[str, ...] | None = None,
) -> dict[str, ExpansionEdge]:
    """
    Same traversal as fixed_depth_expansion, but also records *which* edge
    (parent node + edge type) first reached each node — needed to build
    edge-level training examples for Phase 5's learned scorer (a node's
    ablation label tells you whether it mattered, but only this function
    tells you which specific edge traversal to credit/blame for reaching
    it). Seeds are not included (they weren't reached via any edge).

    When multiple edge types connect the same node pair, one is picked
    arbitrarily (not scored/ranked) — good enough for generating training
    examples, not for anything that needs to pick the "best" edge type.
    """
    undirected = graph.to_undirected(as_view=True)

    reached: dict[str, ExpansionEdge] = {}
    seed_set = {s for s in seed_nodes if s in undirected}
    frontier = set(seed_set)

    for hop in range(1, depth + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in undirected.neighbors(node):
                if neighbor in reached or neighbor in seed_set:
                    continue
                if undirected.nodes[neighbor].get("node_type") == "external":
                    continue
                edge_datas = undirected.get_edge_data(node, neighbor) or {}
                types_here = {d.get("edge_type") for d in edge_datas.values()}
                if edge_types is not None and not (types_here & set(edge_types)):
                    continue
                chosen_type = next(iter(types_here)) if types_here else "unknown"
                reached[neighbor] = ExpansionEdge(parent=node, edge_type=chosen_type, hop=hop)
                next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return reached


if __name__ == "__main__":
    import sys

    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo
    from context_compiler.compiler.seed_retrieval import lexical_seed_retrieval

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else "graph transformer attention bias"
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    g = build_code_graph(parse_repo(repo))
    seeds = [n for n, _ in lexical_seed_retrieval(g, query, top_k=5)]
    print(f"Seeds: {seeds}")

    distances = fixed_depth_expansion(g, seeds, depth=depth)
    print(f"\nExpanded to {len(distances)} nodes within depth {depth}:")
    for node, d in sorted(distances.items(), key=lambda x: x[1]):
        print(f"  hop {d}  {node}")

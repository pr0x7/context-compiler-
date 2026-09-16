"""
Parse source code into a multi-edge-type graph:
  - AST edges       (parent -> child, from the syntax tree itself)
  - Next-token edges (sequential edges between consecutive leaf/token nodes,
                       like code2vec/CodeBERT style — gives the model a
                       "reading order" signal alongside pure tree structure)
  - Data-flow edges  (identifier -> identifier, connecting each use of a
                       variable back to its most recent definition/previous
                       use — a lightweight def-use heuristic, not a full
                       dataflow analysis)

This is intentionally a heuristic, dependency-light approach (no real
compiler/CFG library) so it stays easy to run and easy to explain in a
writeup. The three edge types are also what you'll ablate later to show
which structural signal matters most.
"""

from dataclasses import dataclass, field
from typing import Optional
from tree_sitter_languages import get_parser


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------

@dataclass
class CodeGraph:
    node_types: list[str] = field(default_factory=list)   # AST node type per node, e.g. "if_statement"
    node_tokens: list[Optional[str]] = field(default_factory=list)  # source text for leaf nodes, else None
    edges: list[tuple[int, int, str]] = field(default_factory=list)  # (src, dst, edge_type)

    @property
    def num_nodes(self) -> int:
        return len(self.node_types)


# Edge type vocabulary — keep this stable, the model embeds edge types by index.
EDGE_TYPES = {"ast": 0, "next_token": 1, "data_flow": 2}

# Node types tree-sitter reports for variable-like identifiers.
# (Adjust per language grammar if you switch off Python/C/JS.)
IDENTIFIER_NODE_TYPES = {"identifier"}


def parse_function_to_graph(source_code: str, language: str = "python") -> CodeGraph:
    """
    Parse a single function's source into a CodeGraph.

    language: any grammar tree_sitter_languages ships, e.g. "python", "c", "javascript".
    """
    parser = get_parser(language)
    tree = parser.parse(bytes(source_code, "utf8"))
    root = tree.root_node

    graph = CodeGraph()
    leaf_order: list[int] = []          # node indices of leaves, in source order (for next_token edges)
    identifier_last_def: dict[str, int] = {}  # var name -> node index of most recent occurrence

    def add_node(ts_node) -> int:
        idx = graph.num_nodes
        graph.node_types.append(ts_node.type)
        if ts_node.child_count == 0:
            token_text = source_code[ts_node.start_byte:ts_node.end_byte]
            graph.node_tokens.append(token_text)
        else:
            graph.node_tokens.append(None)
        return idx

    def walk(ts_node, parent_idx: Optional[int]):
        idx = add_node(ts_node)

        if parent_idx is not None:
            graph.edges.append((parent_idx, idx, "ast"))

        is_leaf = ts_node.child_count == 0
        if is_leaf:
            leaf_order.append(idx)

            # --- lightweight data-flow heuristic ---
            if ts_node.type in IDENTIFIER_NODE_TYPES:
                name = source_code[ts_node.start_byte:ts_node.end_byte]
                if name in identifier_last_def:
                    graph.edges.append((identifier_last_def[name], idx, "data_flow"))
                identifier_last_def[name] = idx

        for child in ts_node.children:
            walk(child, idx)

    walk(root, None)

    # --- next-token edges: sequential chain over leaves in source order ---
    for a, b in zip(leaf_order[:-1], leaf_order[1:]):
        graph.edges.append((a, b, "next_token"))

    return graph


if __name__ == "__main__":
    # quick smoke test
    sample = """
def add(a, b):
    result = a + b
    return result
"""
    g = parse_function_to_graph(sample, language="python")
    print(f"nodes: {g.num_nodes}")
    print(f"edges: {len(g.edges)}")
    for etype in EDGE_TYPES:
        count = sum(1 for _, _, t in g.edges if t == etype)
        print(f"  {etype}: {count}")

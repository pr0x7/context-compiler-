"""
Phase 1: Code Graph.

Builds a NetworkX MultiDiGraph from a RepoParseResult, with typed nodes
(file / function / method / class / test / external) and typed edges
(imports / calls / inherits / defines / test_covers / co_change).

Versioned per commit: save_graph()/load_graph() key the JSON snapshot by
commit hash, so a run is always reproducible against a fixed graph state
rather than a moving repo (see design doc section 6 — this project also
freezes the repo snapshot for its duration rather than handling live
re-snapshotting, per the resolved-decisions section).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

from context_compiler.parser.repo_parser import RepoParseResult, parse_repo


def build_code_graph(parsed: RepoParseResult, co_change_history_limit: int = 200) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.graph["commit_hash"] = parsed.commit_hash
    g.graph["repo_root"] = parsed.repo_root

    # --- nodes: files ---
    for f in parsed.files:
        g.add_node(f, node_type="test" if f in parsed.test_files else "file")

    # --- nodes: symbols (functions/methods/classes), + "defines" edges from their file ---
    for qualname, sym in parsed.symbols.items():
        g.add_node(qualname, node_type=sym.kind, file_path=sym.file_path,
                    lineno=sym.lineno, end_lineno=sym.end_lineno,
                    docstring=sym.docstring or "")
        if sym.file_path in g:
            g.add_edge(sym.file_path, qualname, edge_type="defines")

    # --- edges: imports (file -> file, or file -> external placeholder) ---
    for src, dst in parsed.import_edges:
        if not g.has_node(dst) and dst.startswith("external:"):
            g.add_node(dst, node_type="external")
        g.add_edge(src, dst, edge_type="imports")

    # --- edges: calls (now exact via Jedi in repo_parser.py) ---
    for caller_qualname, target_qualname in parsed.call_edges:
        if g.has_node(caller_qualname) and g.has_node(target_qualname):
            g.add_edge(caller_qualname, target_qualname, edge_type="calls")

    # --- edges: inherits (class -> base, still name-based) ---
    # Need to compute short_name map just for inherits since calls no longer need it.
    by_short_name: dict[str, list[str]] = {}
    for qualname in parsed.symbols:
        short = qualname.rsplit(".", 1)[-1]
        by_short_name.setdefault(short, []).append(qualname)

    for cls_qualname, base_name in parsed.inherit_edges:
        short_base = base_name.rsplit(".", 1)[-1]
        targets = by_short_name.get(short_base, [])
        if targets:
            for target in targets:
                g.add_edge(cls_qualname, target, edge_type="inherits")
        # if the base isn't in our own symbol table (e.g. nn.Module), we
        # deliberately don't fabricate a node for it — inheriting from
        # external/library classes isn't tracked as a graph edge, only
        # noted in the symbol's `bases` field.

    # --- edges: test_covers (test file -> source file, import-based heuristic) ---
    for test_file, source_module in parsed.test_covers_edges:
        g.add_edge(test_file, source_module, edge_type="test_covers")

    # --- edges: co_change (files that changed together in recent commits) ---
    co_change_edges = _compute_co_change_edges(Path(parsed.repo_root), co_change_history_limit)
    for (a, b), count in co_change_edges.items():
        if g.has_node(a) and g.has_node(b):
            g.add_edge(a, b, edge_type="co_change", weight=count)
            g.add_edge(b, a, edge_type="co_change", weight=count)

    return g


def _compute_co_change_edges(repo_root: Path, history_limit: int) -> Counter:
    """
    Files that appear together in the same commit, across recent history.
    Returns a Counter keyed by (file_a, file_b) sorted-pair -> co-occurrence count.
    Best-effort: returns empty if this isn't a git repo.
    """
    counts: Counter = Counter()
    try:
        log = subprocess.run(
            ["git", "-C", str(repo_root), "log", f"-{history_limit}", "--name-only", "--pretty=format:__COMMIT__"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return counts

    current_files: list[str] = []
    for line in log.splitlines():
        if line == "__COMMIT__":
            _tally_pairs(current_files, counts)
            current_files = []
        elif line.strip():
            current_files.append(line.strip())
    _tally_pairs(current_files, counts)  # last commit block
    return counts


def _tally_pairs(files: list[str], counts: Counter) -> None:
    py_files = sorted(f for f in files if f.endswith(".py"))
    for i in range(len(py_files)):
        for j in range(i + 1, len(py_files)):
            counts[(py_files[i], py_files[j])] += 1


# ---------------------------------------------------------------------------
# Save / load (versioned by commit hash)
# ---------------------------------------------------------------------------

def save_graph(g: nx.MultiDiGraph, out_dir: str) -> str:
    out_path = Path(out_dir) / f"{g.graph['commit_hash']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(g, edges="edges")
    out_path.write_text(json.dumps(data, indent=2))
    return str(out_path)


def load_graph(path: str) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text())
    return nx.node_link_graph(data, edges="edges", multigraph=True, directed=True)


if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./graph_snapshots"

    parsed = parse_repo(target_repo)
    graph = build_code_graph(parsed)

    print(f"Graph for commit {graph.graph['commit_hash']}:")
    print(f"  nodes: {graph.number_of_nodes()}")
    print(f"  edges: {graph.number_of_edges()}")

    node_type_counts = Counter(d.get("node_type", "?") for _, d in graph.nodes(data=True))
    edge_type_counts = Counter(d.get("edge_type", "?") for _, _, d in graph.edges(data=True))
    print(f"  node types: {dict(node_type_counts)}")
    print(f"  edge types: {dict(edge_type_counts)}")

    saved_path = save_graph(graph, out_dir)
    print(f"Saved snapshot to {saved_path}")

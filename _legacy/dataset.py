"""
Turns parsed CodeGraphs into PyTorch Geometric Data objects.

Expects a dataset file (CSV or JSONL) with at least two columns:
  - "func"   : the raw source code of one function
  - "target" : 0/1 label (e.g. buggy / not buggy)

This matches common formats like Devign / CodeXGLUE defect-detection.
Adjust `LABEL_COL` / `CODE_COL` below if your dataset uses different names.

Also precomputes shortest-path distance matrices per graph, since the
Graphormer-style model needs them for its attention bias. This is done once
at load time (not per training step) to keep training fast.
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from parse_ast import parse_function_to_graph, EDGE_TYPES, CodeGraph

CODE_COL = "func"
LABEL_COL = "target"

MAX_SPD = 20          # cap shortest-path distance (longer -> treated as "far", bucketed together)
MAX_NODES = 256        # skip functions whose graphs are bigger than this (keeps memory/compute sane on 8GB)


def build_type_vocab(node_types_lists: list[list[str]]) -> dict[str, int]:
    """Map every AST node type string seen in training data to an integer id."""
    vocab = {"<unk>": 0}
    for types in node_types_lists:
        for t in types:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def code_graph_to_pyg(graph: CodeGraph, type_vocab: dict[str, int], label: int) -> Data | None:
    if graph.num_nodes == 0 or graph.num_nodes > MAX_NODES:
        return None

    # --- node features: AST type id (token text is intentionally left out of
    #     v1 to keep this lightweight; add a token embedding later as an
    #     easy follow-up experiment) ---
    x = torch.tensor(
        [type_vocab.get(t, 0) for t in graph.node_types], dtype=torch.long
    ).unsqueeze(1)

    # --- edges, split by type, PyG-style (edge_index + edge_type) ---
    if not graph.edges:
        return None
    src = torch.tensor([e[0] for e in graph.edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in graph.edges], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.tensor([EDGE_TYPES[e[2]] for e in graph.edges], dtype=torch.long)

    # --- shortest-path distance matrix (undirected, AST+CFG+DFG combined),
    #     used by the graph transformer's attention bias ---
    nxg = nx.Graph()
    nxg.add_nodes_from(range(graph.num_nodes))
    nxg.add_edges_from((e[0], e[1]) for e in graph.edges)
    spd = np.full((graph.num_nodes, graph.num_nodes), MAX_SPD, dtype=np.int64)
    for src_node, lengths in nx.all_pairs_shortest_path_length(nxg, cutoff=MAX_SPD):
        for dst_node, d in lengths.items():
            spd[src_node, dst_node] = d
    spd_tensor = torch.tensor(spd, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                y=torch.tensor([label], dtype=torch.long))
    data.spd = spd_tensor  # dense [N, N] — fine at MAX_NODES=256, revisit if you raise the cap
    data.num_nodes = graph.num_nodes
    return data


def load_raw_examples(path: str) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix} (use .csv or .jsonl)")


class ASTCodeDataset(Dataset):
    """
    Loads a code+label file, parses every function to a graph, and caches
    PyG Data objects in memory. Fine for the ~10-20k function scale this
    project targets on 8GB; switch to on-disk caching if you scale up.
    """

    def __init__(self, path: str, language: str = "python", type_vocab: dict[str, int] | None = None):
        super().__init__()
        raw = load_raw_examples(path)

        parsed: list[tuple[CodeGraph, int]] = []
        for row in raw:
            code = row.get(CODE_COL)
            label = row.get(LABEL_COL)
            if code is None or label is None:
                continue
            graph = parse_function_to_graph(code, language=language)
            parsed.append((graph, int(label)))

        if type_vocab is None:
            type_vocab = build_type_vocab([g.node_types for g, _ in parsed])
        self.type_vocab = type_vocab

        self.examples: list[Data] = []
        for graph, label in parsed:
            data = code_graph_to_pyg(graph, type_vocab, label)
            if data is not None:
                self.examples.append(data)

        print(f"Loaded {len(self.examples)}/{len(raw)} examples "
              f"(skipped empty/oversized graphs), vocab size={len(type_vocab)}")

    def len(self):
        return len(self.examples)

    def get(self, idx):
        return self.examples[idx]

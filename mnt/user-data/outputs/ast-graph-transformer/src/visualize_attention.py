"""
Visualize which AST nodes the graph transformer attends to for a single
example — turns "I trained a model" into "here's what it learned", and
makes for a good README screenshot.

Usage:
    python visualize_attention.py --checkpoint best_model.pt --source example.py
"""

import argparse

import matplotlib.pyplot as plt
import networkx as nx
import torch

from parse_ast import parse_function_to_graph
from dataset import code_graph_to_pyg
from graph_transformer import GraphTransformer, collate_transformer_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="path to a .py (or other language) file with one function")
    parser.add_argument("--language", default="python")
    parser.add_argument("--layer", type=int, default=-1, help="which transformer layer's attention to visualize")
    parser.add_argument("--head", type=int, default=0, help="which attention head to visualize")
    parser.add_argument("--out", default="attention.png")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    model_args = ckpt["args"]

    model = GraphTransformer(len(vocab), dim=model_args["dim"], num_heads=model_args["heads"],
                              num_layers=model_args["layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    source = open(args.source).read()
    graph = parse_function_to_graph(source, language=args.language)
    data = code_graph_to_pyg(graph, vocab, label=0)  # label unused at inference
    batch = collate_transformer_batch([data])

    with torch.no_grad():
        logits, attn = model.forward_with_attention(batch, layer_idx=args.layer)

    pred = logits.argmax(dim=-1).item()
    print(f"Prediction: {'buggy' if pred == 1 else 'not buggy'} (logits: {logits.tolist()})")

    n = graph.num_nodes
    attn_map = attn[0, args.head, :n, :n].numpy()

    # Build a small networkx graph from AST edges only, for a readable layout
    nxg = nx.DiGraph()
    for i, t in enumerate(graph.node_types):
        label = graph.node_tokens[i] if graph.node_tokens[i] else t
        nxg.add_node(i, label=label[:12])
    for src, dst, etype in graph.edges:
        if etype == "ast":
            nxg.add_edge(src, dst)

    pos = nx.nx_agraph.graphviz_layout(nxg, prog="dot") if _has_graphviz() else nx.spring_layout(nxg, seed=0)

    # Node color = total incoming attention received (how much the model "looked at" this node)
    attention_received = attn_map.sum(axis=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(nxg, pos, with_labels=False, node_size=300, ax=ax,
            node_color=attention_received, cmap="viridis", edge_color="#cccccc", arrows=False)
    labels = {i: nxg.nodes[i]["label"] for i in nxg.nodes}
    nx.draw_networkx_labels(nxg, pos, labels, font_size=7, ax=ax)
    ax.set_title(f"Attention over AST (layer {args.layer}, head {args.head}) — pred: "
                 f"{'buggy' if pred == 1 else 'not buggy'}")
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(attention_received)
    fig.colorbar(sm, ax=ax, label="attention received")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved visualization to {args.out}")


def _has_graphviz() -> bool:
    try:
        import pygraphviz  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()

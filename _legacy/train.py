"""
Train and evaluate either the graph transformer or a GCN/GAT baseline on
the AST-based code graphs.

Example usage:
    python train.py --model transformer --train data/train.jsonl --val data/val.jsonl --edges ast,cfg,dfg
    python train.py --model gcn --train data/train.jsonl --val data/val.jsonl --edges ast

The --edges flag controls the ablation: which edge types to keep in the
graph before training (drop cfg/dfg to see how much they actually help).
"""

import argparse
import copy

import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from dataset import ASTCodeDataset
from graph_transformer import GraphTransformer, collate_transformer_batch
from baselines import GCNBaseline, GATBaseline


def filter_edge_types(dataset: ASTCodeDataset, keep: set[str]):
    """In-place: drop edges/spd not in `keep` for an edge-type ablation."""
    from parse_ast import EDGE_TYPES
    keep_ids = {EDGE_TYPES[k] for k in keep if k in EDGE_TYPES}
    for data in dataset.examples:
        mask = torch.tensor([et.item() in keep_ids for et in data.edge_type])
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        # Note: spd is recomputed from the full graph at parse time; for a
        # strict ablation you'd re-run parse_ast with edges pre-filtered.
        # This filtered edge_index/edge_type is what GCN/GAT baselines see;
        # for the transformer's spd-based bias, re-parse with restricted
        # edges if you want a fully clean ablation.


def run_epoch(model, loader, optimizer, device, is_transformer: bool, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            if is_transformer:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["y"]
                logits = model(batch)
            else:
                batch = batch.to(device)
                labels = batch.y
                logits = model(batch)

            loss = torch.nn.functional.cross_entropy(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "gcn", "gat"], default="transformer")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--language", default="python")
    parser.add_argument("--edges", default="ast,next_token,data_flow",
                         help="comma-separated subset of: ast,next_token,data_flow")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    keep_edges = set(args.edges.split(","))

    print("Loading train set...")
    train_ds = ASTCodeDataset(args.train, language=args.language)
    print("Loading val set (sharing train vocab)...")
    val_ds = ASTCodeDataset(args.val, language=args.language, type_vocab=train_ds.type_vocab)

    if keep_edges != {"ast", "next_token", "data_flow"}:
        print(f"Ablation: keeping only edge types {keep_edges}")
        filter_edge_types(train_ds, keep_edges)
        filter_edge_types(val_ds, keep_edges)

    vocab_size = len(train_ds.type_vocab)
    is_transformer = args.model == "transformer"

    if is_transformer:
        model = GraphTransformer(vocab_size, dim=args.dim, num_heads=args.heads, num_layers=args.layers).to(device)
        train_loader = TorchDataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                        collate_fn=collate_transformer_batch)
        val_loader = TorchDataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=collate_transformer_batch)
    else:
        model_cls = GCNBaseline if args.model == "gcn" else GATBaseline
        model = model_cls(vocab_size, dim=args.dim, num_layers=args.layers).to(device)
        # spd is a dense [N,N] matrix only used by the transformer's attention
        # bias — exclude it here so PyG doesn't try to concat mismatched sizes.
        train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, exclude_keys=["spd"])
        val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, exclude_keys=["spd"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, device, is_transformer, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, optimizer, device, is_transformer, train=False)

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    torch.save({"state_dict": best_state, "vocab": train_ds.type_vocab, "args": vars(args)}, args.out)
    print(f"Best val F1: {best_f1:.3f} — saved to {args.out}")


if __name__ == "__main__":
    main()

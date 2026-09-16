"""
Standard message-passing baselines (GCN, GAT) to compare the graph
transformer against. These use PyG's native sparse batching — much cheaper
than the transformer's dense [B,N,N] attention, which is exactly the
tradeoff worth calling out in your writeup (GNNs scale better, transformers
capture longer-range structure).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GCNBaseline(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, num_layers: int = 4, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.convs = nn.ModuleList([GCNConv(dim, dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, batch) -> torch.Tensor:
        x = self.node_embed(batch.x.squeeze(-1))
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch.edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        pooled = global_mean_pool(x, batch.batch)
        return self.classifier(pooled)


class GATBaseline(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, num_layers: int = 4, num_heads: int = 4,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.node_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.convs = nn.ModuleList([
            GATConv(dim, dim // num_heads, heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, batch) -> torch.Tensor:
        x = self.node_embed(batch.x.squeeze(-1))
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch.edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        pooled = global_mean_pool(x, batch.batch)
        return self.classifier(pooled)

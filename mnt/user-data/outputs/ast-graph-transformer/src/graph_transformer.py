"""
A small Graphormer-style graph transformer.

Core idea (Ying et al., "Do Transformers Really Perform Bad for Graph
Representation?", 2021): run a normal transformer encoder over all nodes,
but bias the attention scores using graph structure — here, shortest-path
distance between each pair of nodes. Nodes far apart in the graph get a
learned penalty; nodes close together (or directly connected by a specific
edge type) get a learned boost. This is what lets a transformer "see" graph
structure without message-passing.

Kept small on purpose (4 layers, dim 128 default) to comfortably fit 8GB
for function-sized graphs (tens to low hundreds of nodes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionBias(nn.Module):
    """Learned bias added to attention logits, based on shortest-path distance."""

    def __init__(self, num_heads: int, max_spd: int = 20):
        super().__init__()
        self.max_spd = max_spd
        # +1 bucket for "disconnected / beyond max_spd"
        self.spd_bias = nn.Embedding(max_spd + 1, num_heads)

    def forward(self, spd: torch.Tensor) -> torch.Tensor:
        # spd: [B, N, N] -> bias: [B, num_heads, N, N]
        spd_clamped = spd.clamp(max=self.max_spd)
        bias = self.spd_bias(spd_clamped)          # [B, N, N, num_heads]
        return bias.permute(0, 3, 1, 2)             # [B, num_heads, N, N]


class GraphTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]  attn_bias: [B, H, N, N]  pad_mask: [B, N] (True = real node)
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, head_dim]

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]
        scores = scores + attn_bias

        # mask out padded nodes (both as query-irrelevant and key-irrelevant)
        key_mask = pad_mask[:, None, None, :]  # [B, 1, 1, N]
        scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)  # rows that were fully masked -> avoid NaN propagation
        attn = self.dropout(attn)

        out = attn @ v  # [B, H, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        x = x + self.dropout(out)

        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class GraphTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, num_heads: int = 8,
                 num_layers: int = 4, num_classes: int = 2, max_spd: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        self.node_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.attn_bias = GraphAttentionBias(num_heads, max_spd=max_spd)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        # batch keys: x [B,N], spd [B,N,N], pad_mask [B,N] (see collate_transformer_batch)
        x = self.node_embed(batch["x"])  # [B, N, D]
        attn_bias = self.attn_bias(batch["spd"])
        pad_mask = batch["pad_mask"]

        for layer in self.layers:
            x = layer(x, attn_bias, pad_mask)
        x = self.final_norm(x)

        # mean-pool over real nodes only
        mask = pad_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)

    def forward_with_attention(self, batch: dict, layer_idx: int = -1):
        """
        Same as forward(), but also returns the last layer's attention map
        for the interpretability step (see visualize_attention.py).
        Recomputes attention manually for a single chosen layer.
        """
        x = self.node_embed(batch["x"])
        attn_bias = self.attn_bias(batch["spd"])
        pad_mask = batch["pad_mask"]

        target_layer = self.layers[layer_idx]
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) + layer_idx if layer_idx < 0 else i == layer_idx:
                # recompute this layer's attention explicitly for return
                B, N, D = x.shape
                h = layer.norm1(x)
                qkv = layer.qkv(h).reshape(B, N, 3, layer.num_heads, layer.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                scores = (q @ k.transpose(-2, -1)) / (layer.head_dim ** 0.5) + attn_bias
                key_mask = pad_mask[:, None, None, :]
                scores = scores.masked_fill(~key_mask, float("-inf"))
                attn_weights = torch.nan_to_num(F.softmax(scores, dim=-1))
            x = layer(x, attn_bias, pad_mask)

        x = self.final_norm(x)
        mask = pad_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(pooled)
        return logits, attn_weights  # attn_weights: [B, H, N, N]


def collate_transformer_batch(data_list) -> dict:
    """
    Pads a list of PyG Data objects (from ASTCodeDataset) into dense
    [B, N, ...] tensors for the graph transformer, which — unlike the
    sparse GCN/GAT baseline — needs a full dense attention matrix per graph.
    """
    max_n = max(d.num_nodes for d in data_list)
    B = len(data_list)

    x = torch.zeros(B, max_n, dtype=torch.long)
    spd = torch.zeros(B, max_n, max_n, dtype=torch.long)
    pad_mask = torch.zeros(B, max_n, dtype=torch.bool)
    y = torch.zeros(B, dtype=torch.long)

    for i, d in enumerate(data_list):
        n = d.num_nodes
        x[i, :n] = d.x.squeeze(-1)
        spd[i, :n, :n] = d.spd
        pad_mask[i, :n] = True
        y[i] = d.y

    return {"x": x, "spd": spd, "pad_mask": pad_mask, "y": y}

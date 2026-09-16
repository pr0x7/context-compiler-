# Legacy Code (ast-graph-transformer)

These files are from the **original `ast-graph-transformer` project** that this
repository evolved from. They are preserved as reference material but are
**not imported or used by any part of the current Context Compiler system**.

## Files

| File | Original Purpose |
|------|-----------------|
| `parse_ast.py` | Tree-sitter AST parsing (replaced by `context_compiler/parser/repo_parser.py`) |
| `dataset.py` | PyG dataset class for the graph transformer |
| `graph_transformer.py` | Graph Transformer model architecture |
| `baselines.py` | GCN/GAT baseline models |
| `train.py` | Training loop for the graph transformer |
| `visualize_attention.py` | Attention weight visualization |

## Dependencies

These files require `torch_geometric`, `tree_sitter`, and `tree_sitter_languages`,
which are **not** required by the current Context Compiler system.

If you want to run them, install the legacy dependencies:
```
pip install torch_geometric tree_sitter==0.21.3 tree_sitter_languages==1.10.2
```

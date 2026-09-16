# Graph Transformers for Source Code Bug Detection

A small research-style project: represent source code as a multi-edge-type
graph built from its AST, and compare a Graphormer-style graph transformer
against standard GNN baselines (GCN, GAT) on a bug/vulnerability detection
task.

## Why this project

- **Structure-aware, not just token-aware.** Code has real structure —
  nesting, control flow, variable def-use chains — that flat token
  sequences (what most code-LLMs see) throw away. This project tests
  whether making that structure explicit, and letting a transformer attend
  over it directly, helps.
- **Small and from-scratch.** Runs comfortably on modest hardware (no GPU
  required for the toy scale in `data/toy.jsonl`; a real dataset like
  Devign/CodeXGLUE needs a single consumer GPU or even CPU with patience).
- **Ablatable.** Three edge types — AST structure, next-token order, and a
  lightweight data-flow heuristic — can be turned on/off independently to
  see which structural signal actually matters for the task.

## How it works

1. **Parse** — `src/parse_ast.py` uses `tree-sitter` to parse a function
   into an AST, then builds a graph with three edge types:
   - `ast`: parent → child syntax tree edges
   - `next_token`: sequential edges between leaf tokens in source order
   - `data_flow`: identifier → identifier, connecting each variable use back
     to its previous occurrence (a heuristic, not a real dataflow analysis)

2. **Encode** — `src/dataset.py` converts each graph into a PyTorch
   Geometric `Data` object: node features are AST node-type embeddings,
   plus a precomputed shortest-path-distance matrix used by the
   transformer's attention bias.

3. **Model** — `src/graph_transformer.py` implements a small
   Graphormer-style model: a standard transformer encoder, but attention
   logits get a learned bias based on shortest-path distance between node
   pairs. `src/baselines.py` implements GCN and GAT baselines on the same
   graphs for comparison.

4. **Train & ablate** — `src/train.py` trains any of the three models, and
   `--edges` lets you drop edge types to ablate their contribution.

5. **Interpret** — `src/visualize_attention.py` renders which AST nodes the
   trained transformer attends to most for a given function.

## Getting a real dataset

The toy data in `data/toy.jsonl` (8 hand-written examples) is only for
verifying the pipeline runs end-to-end. For real results, use a public
labeled defect-detection dataset such as:

- **Devign** — ~27k labeled C functions (buggy / not buggy)
- **CodeXGLUE Defect Detection** — similar task, HuggingFace `datasets` hub
- **Big-Vul** — larger C/C++ vulnerability dataset if you want more data

Reformat whichever you pick into JSONL with `{"func": "...", "target": 0/1}`
per line (see `dataset.py` — change `CODE_COL`/`LABEL_COL` if your columns
are named differently), and set `--language` to match (`c`, `python`, etc.
via `tree_sitter_languages`).

## Quickstart

```bash
pip install -r requirements.txt

# sanity check the pipeline on the tiny toy set
cd src
python train.py --model transformer --train ../data/toy.jsonl --val ../data/toy.jsonl \
    --epochs 5 --batch_size 4 --dim 64 --heads 4 --layers 2

# once you have a real dataset:
python train.py --model transformer --train ../data/train.jsonl --val ../data/val.jsonl \
    --language c --epochs 15

python train.py --model gcn --train ../data/train.jsonl --val ../data/val.jsonl --language c
python train.py --model gat --train ../data/train.jsonl --val ../data/val.jsonl --language c

# ablation: AST structure only, no next-token or data-flow edges
python train.py --model transformer --train ../data/train.jsonl --val ../data/val.jsonl \
    --language c --edges ast

# visualize what the trained transformer attends to
python visualize_attention.py --checkpoint best_model.pt --source example.c --language c
```

## What to report in a writeup

- Accuracy/F1 table: GCN vs GAT vs Graph Transformer (note: bug-detection
  datasets are usually class-imbalanced — F1 matters more than accuracy)
- Ablation table: full edge set vs AST-only vs AST+data-flow, etc.
- 2-3 attention visualizations with a short note on what each highlights
- Honest limitations: the data-flow heuristic is not a real dataflow
  analysis (no branch-aware liveness, no interprocedural tracking); the
  toy scale here is for pipeline verification, not a claimed result

## Known constraints / notes

- `tree_sitter_languages==1.10.2` requires `tree_sitter==0.21.3` — newer
  `tree_sitter` versions changed their API and will break `get_parser()`.
  This is pinned in `requirements.txt`.
- The graph transformer uses dense `[B, N, N]` attention, so it scales
  worse than the GNN baselines on very large graphs — this is itself worth
  a sentence in your writeup as a tradeoff, not just a limitation.
- `MAX_NODES` in `dataset.py` (default 256) skips unusually large functions
  to keep memory bounded on modest hardware — tune based on your dataset
  and GPU.

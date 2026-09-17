# Context Compiler

An intelligent, graph-based context-retrieval engine for AI coding agents. 

Instead of dumping entire files or relying on simple embedding-based Retrieval-Augmented Generation (RAG), this project builds an explicit **Code Graph** of the repository (tracking imports, function calls, class hierarchies, and test coverage) to determine the *minimal sufficient context* an AI agent needs to solve a task.

## Key Research Contributions

1. **Graph-Based Score-Guided Expansion**: Moves beyond fixed-depth BFS by using a learned neural edge scorer to navigate the code graph, selecting only the most relevant dependency paths.
2. **Ablation-Driven Feedback Loop**: Automatically derives ground-truth "minimal context" labels by systematically ablating context and verifying structural completeness, which is then used to retrain the relevance scorer.
3. **Hybrid Semantic + Lexical Retrieval**: Combines exact-match lexical search with dense vector embeddings (`all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF) for highly robust seed retrieval.

## How It Works

When an AI agent is given a coding task, the Context Compiler:
1. **Parses the repository** to build a rich semantic graph of all Python symbols and their relationships using AST extraction.
2. **Retrieves seed nodes** using hybrid semantic and lexical retrieval based on the task description.
3. **Expands the context** by executing a best-first traversal over the code graph, guided by a trained multi-layer perceptron (MLP) edge scorer.
4. **Optimizes a token budget** using exact token counting (`tiktoken`) to produce a chunked "Context Bundle" containing exactly what the agent needs.

## Getting Started

### Prerequisites
- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation
```bash
git clone https://github.com/pr0x7/context-compiler-.git
cd context-compiler-
pip install -r requirements.txt
```

### Usage
Run the minimal agent simulator against the current repository:
```bash
export PYTHONPATH=.
python3 context_compiler/ledger/minimal_agent.py . "understand and modify run_ablation"
```

Run the internal test suite:
```bash
python3 -m pytest -v tests/
```

## System Architecture & Evaluation

The system consists of 7 completed phases, thoroughly documented in our research and development logs:
1. **Repository Parser & Code Graph**: Graph extraction via AST parsing.
2. **Baseline Compiler**: Fixed-depth expansion and token budgeting.
3. **Context Ledger**: Observability tool tracking agent context pulls.
4. **Ablation Engine**: Derives minimal-context ground truth.
5. **Learned Edge Scorer**: Neural model predicting traversal relevance.
6. **Feedback Loop**: Retraining the scorer on ablation-derived data.
7. **Evaluation**: See `BENCHMARK.md` for a full breakdown of precision/recall metrics across strategies.

For a deep dive into the architecture, design decisions, and known limitations, please refer to:
- [Design Document](docs/design.md) (Architecture & SDD)
- [Benchmark Results](BENCHMARK.md) (Quantitative Evaluation)
- [Development Log](DEV_README.md) (Implementation details & caveats)

## License
MIT License.

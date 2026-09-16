# SDD — AST/Graph-Based Context Compiler + Context Ledger

**Project:** graph-transformer-ast (extension)
**Scope of this doc:** design for the "1 + 2" combination — a structure-aware context compiler for coding agents, plus a context observability ledger and ablation-based eval loop.
**Status:** draft, for guiding implementation

---

## 1. Problem statement

Coding agents currently select context via embedding RAG or full-file dumping, neither of which uses the actual structure of the codebase (call graph, import graph, AST relationships). There is also no standard way to observe *why* an agent run succeeded or failed based on what context it had, or to know what the *minimal sufficient* context for a task actually was.

This project builds:
1. A **context compiler** that uses a graph-transformer relevance scorer over the code graph to select context for a coding task.
2. A **context ledger** that logs the full union of compiler-selected and agent-pulled context per run.
3. An **ablation-based eval loop** that derives ground-truth minimal context per task and feeds it back into the scorer.

## 2. Non-goals

- Not building a general-purpose agent harness (assume Claude Code / OpenHands / a minimal custom loop as the executor).
- Not building a new LLM or fine-tuning the base agent model.
- Not targeting arbitrary languages at launch — pick one (Python recommended) for the parser/graph layer, design the graph schema to be language-agnostic in principle.

## 3. Architecture overview

```
Task
  ↓
Repository Parser
  ↓
Code Graph (versioned per commit)
  ↓
Context Compiler
  ├─ Seed Retrieval        (lexical/embedding hit → entry-point nodes)
  ├─ Score-Guided Expansion (graph transformer drives traversal, not fixed-depth BFS)
  ├─ Redundancy Removal
  └─ Token-Budget Optimization
  ↓
LLM Agent (executes task, may pull additional context via tools)
  ↓
Context Ledger (compiler-selected ∪ agent-pulled, timestamped)
  ↓
Ablation Engine → minimal-sufficient-context labels
  ↓
Evaluation (task success, token cost, latency, context-efficiency)
  ↓
Feedback → retrains/updates relevance scorer
```

Key correction from the earlier pipeline sketch: **relevance scoring drives expansion, it isn't a filter applied after a fixed-depth traversal.** And **evaluation feeds back into the compiler** — this is a loop, not a linear pipeline.

## 4. Components

### 4.1 Repository Parser
- Input: repo at a given commit.
- Output: AST per file, symbol table, import graph, call graph, test-to-source mapping, git blame/history metadata.
- Existing AST graph-transformer encoder work is reused here as the graph representation layer.

### 4.2 Code Graph
- Nodes: files, functions/methods, classes, symbols, tests.
- Edges: imports, calls, inherits, defines, references, co-change (from git history), test-covers.
- **Versioned per commit** — snapshot the graph at each commit so runs are reproducible against a fixed graph state, not a moving repo.

### 4.3 Context Compiler
- **Seed retrieval**: cheap lexical/embedding match between task description and node identifiers/docstrings → initial candidate set.
- **Score-guided expansion**: the graph transformer scores edges/neighbors from the current frontier and expands along high-relevance paths (e.g. beam search or top-k attention-weighted walk), not a fixed-radius BFS-then-filter. This is the component that differentiates the approach from RAG.
- **Redundancy removal**: dedupe overlapping symbol/file spans, collapse near-identical context (e.g. multiple call sites of the same function).
- **Token-budget optimization**: knapsack-style selection to maximize predicted relevance within a fixed token budget.
- Output: an ordered, budgeted context bundle + the compiler's own confidence/relevance scores per included node (needed later for eval).

### 4.4 Agent execution
- Any existing harness. Requirement: it must expose a tool-call log (what it read/grepped/searched beyond what the compiler gave it).

### 4.5 Context Ledger
- Logs the **union** of compiler-selected and agent-pulled context, not just the compiler's output — this is what makes it an observability tool rather than a compiler log.
- Per-run record includes:
  - Files/symbols retrieved (source: compiler vs. agent tool call)
  - Relevance scores assigned by the compiler
  - Tokens consumed, broken down by source
  - Compaction events (what was dropped/summarized, when)
  - Tool results
  - Timestamps for all of the above, so two runs can be diffed step-by-step
- Storage: structured (e.g. JSONL or SQLite) per run, keyed by task ID + commit hash + run ID, so runs are comparable and diffable.

### 4.6 Ablation Engine
- For each completed task, systematically remove subsets of the retrieved context and rerun (or use a cheaper proxy: mask context and check if a held-out test still passes / a checker still validates).
- Produces a **minimal-sufficient-context label** per task: the smallest context set that still yields success.
- This label is the ground truth for two things:
  1. Precision/recall of the compiler's selection against the minimal set (eval metric).
  2. Training signal for the relevance scorer (feedback loop).
- Note: full ablation is expensive; start with a greedy removal heuristic (drop lowest-scored nodes first, stop when success breaks) rather than exhaustive subset search.

### 4.7 Evaluation
Metrics to report per task and in aggregate:
- **Task success rate** (pass/fail against test suite or checker).
- **Token cost** (total, and split compiler vs. agent-pulled).
- **Latency** (wall clock, and compiler-only latency separately from agent latency).
- **Context efficiency** = useful tokens / total tokens, where "useful" = tokens overlapping the ablation-derived minimal set.
- **Precision/recall of compiler selection** against the minimal-sufficient-context label.

Baselines to compare against: full-file dump, embedding-only RAG, fixed-depth graph BFS (no learned scoring) — this last baseline isolates the value of the graph transformer specifically.

## 5. Feedback loop (what closes the pipeline into a system)

Ablation-derived minimal-context labels → supervised signal for the relevance scorer → retrain/fine-tune periodically → redeploy compiler. This is what turns the project from "a context pipeline" into a genuine research contribution: the ablation-labeled dataset is itself a reusable artifact, independent of the rest of the tool.

## 6. Reproducibility requirements

- Code graph snapshotted per commit (see 4.2).
- Every ledger record keyed by (task ID, commit hash, run ID, compiler version, scorer model version).
- Ablation labels stored alongside the commit snapshot so they don't go stale as the repo moves.

## 7. Suggested build phases

1. **Parser + code graph** for one language (Python), stored per-commit.
2. **Baseline compiler**: seed retrieval + fixed-depth expansion + token budget (no learned scoring yet) — gives a working end-to-end pipeline and a baseline for comparison.
3. **Ledger**: wire up compiler + agent tool-call logging, unioned and timestamped.
4. **Ablation engine**: greedy-removal heuristic, minimal-context labels on a small task set.
5. **Learned relevance scorer**: plug in the existing graph transformer, replace fixed-depth expansion with score-guided expansion.
6. **Feedback loop**: retrain scorer on ablation labels, re-evaluate.
7. **Benchmark writeup**: compiler vs. RAG vs. full-dump vs. fixed-depth-graph baselines, on task success / cost / context-efficiency.

## 8. Resolved decisions (updated from original draft)

- **Ablation cost budget**: use a cheap static-check/linter proxy instead of full agent reruns for most tasks, given the 8GB local RAM/VRAM constraint. Full-agent-rerun ablation is out of scope unless a small API budget is available later.
- **Repo versioning**: freeze one repo snapshot for the duration of the project rather than handling live re-snapshotting. Simpler, and fully adequate for a portfolio-scale system.
- **Scorer architecture**: to be resolved when Phase 5 starts — score-guided *expansion* is fundamentally about edges ("which neighbor is worth traversing to"), so an edge-scoring formulation may fit better than reusing the node-classification-style transformer from the original AST project as-is.

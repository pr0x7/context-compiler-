# Context Compiler for Coding Agents

A structure-aware context compiler for coding agents: instead of embedding
RAG or full-file dumping, this project builds an actual code graph (imports,
calls, class hierarchy, test coverage, co-change history) and — eventually —
uses a graph transformer to *score-guide* which parts of that graph belong
in an agent's context window for a given task.

Full design rationale, architecture, and the phased build plan: **[docs/design.md](docs/design.md)**.

The final results and honest limitations: **[docs/BENCHMARK.md](docs/BENCHMARK.md)**.

## What's built vs. designed

| Phase | Status | Notes |
|---|---|---|
| 1. Repository parser + code graph | ✅ built | `parser/`, `graph/` — tested against a real repo |
| 2. Baseline compiler (fixed-depth, no learned scoring) | ✅ built | `compiler/` — seed retrieval, expansion, redundancy removal, token budgeting, all tested end-to-end |
| 3. Context ledger | ✅ built | `ledger/` — includes a minimal stand-in agent harness (not a real agent integration — see caveats); union-logs compiler + agent context, JSONL storage, run-diffing tested |
| 4. Ablation engine (cheap proxy) | ✅ built | `ablation/` — structural dependency-completeness proxy (not a real linter/test run — see caveats); auto-generates tasks from the graph, greedy removal tested end-to-end |
| 5. Learned relevance scorer | ✅ built (proof-of-mechanism) | `scorer/` — edge-scoring model, trained on ablation-derived labels, wired into `compiler/score_guided_expansion.py` as a drop-in replacement for fixed-depth expansion. Trained on ~100 examples from 10 tasks — mechanism proven, not yet a good ranker. See caveats. |
| 6. Feedback loop | ✅ built, one iteration run | `scorer/feedback_loop.py` — regenerated training data from the scorer's own score-guided bundles, retrained on combined round-1+round-2 data, measured real improvement (see results below) |
| 7. Benchmark writeup | ✅ done | `docs/BENCHMARK.md` — real numbers from `scorer/generate_benchmark_data.py`, honest framing, explicit note on baselines never built |

This table is the honest source of truth — update it as phases complete
rather than letting the README drift from what's actually working.

## Phase 1 — what it does

**`parser/repo_parser.py`** walks a Python repo and extracts:
- files, and which ones look like tests (`test_*.py` / `*_test.py`)
- symbols (functions, methods, classes) with location + docstring
- import edges, resolved to files inside the repo where possible
- call edges (name-based — see caveats below)
- class inheritance edges (name-based)
- test-to-source mapping (import-based heuristic)

**`graph/code_graph.py`** turns those records into a `networkx.MultiDiGraph`
with typed nodes (`file` / `function` / `method` / `class` / `test` /
`external`) and typed edges (`imports` / `calls` / `inherits` / `defines` /
`test_covers` / `co_change`), and saves/loads it as JSON **keyed by commit
hash** — so a graph snapshot is reproducible against a fixed repo state.

### Try it

```bash
pip install -r requirements.txt

python graph/code_graph.py /path/to/some/repo ./graph_snapshots
```

This prints node/edge counts by type and saves
`./graph_snapshots/<commit_hash>.json`.

## Phase 2 — what it does

**`compiler/seed_retrieval.py`** — cheap lexical (token-overlap) match
between a task description and node names/docstrings, to find initial
candidate "entry point" nodes.

**`compiler/expansion.py`** — fixed-depth BFS outward from seed nodes over
the undirected graph, up to N hops. This is the explicit strawman baseline
that Phase 5's *score-guided* expansion needs to outperform.

**`compiler/redundancy.py`** — drops a whole-file candidate when specific
symbols from that file are already selected (keeps a class + its own
methods, since that's not true redundancy).

**`compiler/token_budget.py`** — reads each candidate's actual source text,
estimates tokens (chars/4 approximation — no tokenizer dependency), and
greedily fills a fixed token budget by score-per-token density (the
practical approximation of knapsack selection).

**`compiler/context_compiler.py`** — orchestrates all four into one
pipeline, producing a `ContextBundle` (ordered, scored, budgeted selection
of real source snippets) that Phase 3's ledger will log and Phase 4's
ablation engine will operate on.

### Try it

```bash
cd compiler
python context_compiler.py /path/to/some/repo "your task description" 3000
```

Prints the seeds it found, the selected nodes with scores/token counts,
and respects the token budget (3rd arg, default 4000).

### Known caveats (Phase 2)

- Seed retrieval is pure lexical overlap — no embeddings, no semantics. A
  task described in different words than the code (e.g. "fix the login
  bug" when the code says `authenticate_user`) will retrieve nothing. This
  is intentional — it's the weak baseline Phase 5 is supposed to beat.
- The "relevance score" for expanded (non-seed) nodes is just
  `seed_score / (1 + hop_distance)` — a hand-picked decay, not learned.
- Token counting is a chars/4 approximation, not an actual tokenizer —
  fine for budgeting, not exact for a specific target model.

## Phase 3 — what it does

**`ledger/minimal_agent.py`** — a deliberately minimal stand-in agent, not
a real integration. It takes a `ContextBundle` from Phase 2, and with some
probability "decides" it needs more context, pulling extra symbols
connected to what it already has. Every pull is logged as a `ToolCall`
with a timestamp. Its "success" rule is a placeholder (context-item count
threshold) — not a real test/checker.

**`ledger/context_ledger.py`** — the actual ledger. Unions compiler-selected
(`ContextBundle.selected`) and agent-pulled (`ToolCall`s) into one record,
tagging each item's `source_origin` so token cost and context makeup can be
split by source later. Stores as JSONL, one line per run, keyed by
(task ID, commit hash, run ID). Includes `diff_runs()` for comparing two
runs on the same task node-by-node.

### Try it

```bash
cd ledger
python context_ledger.py /path/to/some/repo "your task description" ./ledger.jsonl
```

Run it twice with different queries, then:

```python
from context_ledger import read_ledger, diff_runs
records = read_ledger("./ledger.jsonl")
print(diff_runs(records[0], records[1]))
```

### Known caveats (Phase 3)

- **The agent harness is a stand-in, not a real integration.** Per
  `docs/HANDOFF.md`, this was a deliberate scope decision to unblock
  building the ledger without first integrating Claude Code/OpenHands/etc.
  Swapping in a real harness later doesn't require ledger changes — it
  just needs to produce `ToolCall`-shaped records.
- **Success is a placeholder heuristic** (context-item count ≥ 3), not a
  real test/checker result. Don't read anything into the `success` field
  in its current form — it's there so the ledger schema and Phase 4's
  ablation engine have something to consume, not because it's meaningful yet.
- **`compaction_events` is an empty hook.** Design.md 4.5 asks the ledger
  to log compaction (what was dropped/summarized, when) — no compaction
  logic exists anywhere in the system yet, so this field is always `[]`.
  Left in the schema intentionally rather than omitted, so adding real
  compaction later doesn't require a schema migration.
- JSONL is append-only with no dedup/compaction of the ledger file itself —
  fine at portfolio scale, would need attention (or a move to SQLite) at
  higher run volumes.

## Phase 4 — what it does

**`ablation/tasks.py`** — auto-generates a small task set directly from the
graph (rather than hand-authoring tasks): picks functions/methods with
enough internal call/inherit dependencies to be interesting, phrases a
task description from the symbol's own (and, for methods, its class's)
name.

**`ablation/proxy_check.py`** — the "cheap proxy" from design.md 4.6. **Read
this file's docstring before trusting the numbers**: it's a *structural
dependency-completeness* check (does the context include everything the
target directly calls/inherits, per the graph's own edges), not a real
linter or test run. It's deterministic and instant, which is why it was
chosen, but it's derived from the same graph the compiler uses — not an
independent signal.

**`ablation/ablation_engine.py`** — greedy removal: starting from a
compiled `ContextBundle`, tries dropping the lowest-scored item first,
keeps the drop if the proxy check still passes. What's left is the
minimal-sufficient-context label. Also computes precision/recall of the
original compiler selection against the true requirement — this is the
number that tells you whether Phase 2's baseline is over-including or
under-including context.

### Try it

```bash
cd ablation
python ablation_engine.py /path/to/some/repo ./ablation_labels.jsonl 2000
```

Runs ablation across 10 auto-generated tasks, prints per-task
feasible/precision/recall, and saves labels to the given JSONL path.

### Known caveats (Phase 4)

- **The proxy is structural, not functional.** See `proxy_check.py`'s
  docstring — it checks direct-dependency completeness against the graph,
  not real code correctness. A stronger version would run something like
  `pyflakes` against a reconstructed synthetic module and check for actual
  undefined-name errors; that's meaningfully more engineering (handling
  missing imports, indentation, non-local references) and is an explicit
  follow-up, not attempted here.
- **The greedy-removal loop is somewhat degenerate for this proxy.**
  Because "required dependencies" is a fixed, deterministic set derived
  from the graph (not discovered through search), the loop always
  converges to exactly `{target} ∪ required_dependencies` when the
  compiler's selection covers it. Its actual value is confirming which of
  the compiler's *specific* choices were essential vs. droppable padding —
  not discovering something unknowable in advance.
- **Tasks are auto-generated, not hand-curated.** This was a deliberate
  scope choice (hand-authoring a task set with real ground truth is
  separate, substantial work) — the auto-generated tasks are a reasonable
  stand-in for demonstrating the pipeline, not a validated benchmark.
- On the sample repo, most auto-generated tasks came back `feasible=False`
  at a 2000-token budget — meaning the Phase 2 baseline compiler often
  doesn't select every direct dependency of a task's target symbol. This
  is a real, informative finding (there's headroom for Phase 5's learned
  scorer to close), not a bug — recall still reports how close the
  compiler got even in the infeasible case.

## Phase 5 — what it does

**Design decision (made explicitly, not defaulted into):** the scorer
predicts **per-edge** relevance ("is this specific traversal, from this
frontier node, via this edge type, worth taking"), not per-node relevance.
This fits score-guided expansion naturally — expansion is fundamentally a
sequence of edge decisions — and gives a clean supervision signal from
Phase 4's ablation labels. The existing `graph_transformer.py` (built for
whole-graph node classification) wasn't reused as-is; this is a new, much
smaller model built around the same "learned embeddings for categorical
graph features" idea as `GraphAttentionBias`, just applied to single-edge
scoring instead of dense whole-graph attention.

**`scorer/build_edge_training_data.py`** — combines Phase 4's ablation
labels with Phase 2's expansion traversal (now edge-tracking, see
`compiler/expansion.py`'s `fixed_depth_expansion_with_edges`) to produce
`(task, frontier_type, edge_type, candidate_type, lexical_overlap) -> label`
examples, where label=1 means the candidate survived ablation (genuinely
needed) and label=0 means it was compiler-selected padding.

**`scorer/edge_scorer.py`** — the model: categorical embeddings for node
types and edge types, a bag-of-words task embedding (same tokenizer as
`seed_retrieval.py`), a small MLP head. Binary classification, not a
transformer — dense attention doesn't fit "score one candidate edge."

**`scorer/train_edge_scorer.py`** — standard train/val loop with
class-imbalance weighting (positive/essential examples are the minority).

**`compiler/score_guided_expansion.py`** — the actual Phase 5 deliverable:
best-first traversal (max-heap over candidate edges, ranked by the trained
scorer) instead of fixed-depth BFS. Same input/output shape as
`expansion.py`'s `fixed_depth_expansion`, so it's a drop-in replacement in
`context_compiler.py` whenever you're ready to switch the compiler over to
it (not yet done — `context_compiler.py` still uses the Phase 2 baseline).

### Try it

```bash
cd scorer
python build_edge_training_data.py /path/to/some/repo ./edge_training_data.jsonl
python train_edge_scorer.py ./edge_training_data.jsonl --epochs 20 --out 1. `python -m context_compiler.graph.code_graph . ./snapshots`
2. `python -m context_compiler.compiler.context_compiler . "graph transformer attention bias" 3000`
3. `python -m context_compiler.scorer.generate_benchmark_data . ./edge_scorer.pt ./edge_scorer_v2.pt` 15
```

### Known caveats (Phase 5)

- **Trained on a proof-of-mechanism dataset, not enough data to trust the
  rankings.** ~100 examples from the 10 auto-generated Phase 4 tasks — the
  training loop runs correctly and loss decreases, but this is nowhere
  near enough data for the scorer to have learned anything generalizable.
  Scaling up means generating more ablation tasks (bounded by how many
  interesting-enough symbols exist in whatever repo you're building the
  graph from).
- **Inherits Phase 4's proxy-check circularity.** The training labels come
  from the structural dependency-completeness proxy (see
  `ablation/proxy_check.py`), so the scorer is fundamentally learning to
  predict "is this a direct call/inherit dependency of the target" — a
  real, narrow signal, not a general notion of task relevance. A stronger
  proxy (real linting/testing) would need re-running Phase 4 before
  re-training here.
- **Not wired into `context_compiler.py`.** `score_guided_expansion.py` is
  built and tested standalone, with the same interface as the Phase 2
  baseline's `fixed_depth_expansion` — swapping it in is mechanical but
  hasn't been done, so the compiler pipeline still runs the fixed-depth
  baseline end-to-end today.
- Edge-type selection when multiple edge types connect the same node pair
  is arbitrary in the training-data builder (picks one, doesn't consider
  all) — a real limitation worth fixing before scaling up data generation.

## Phase 6 — what it does, and the actual result

**`compiler/context_compiler.py`** was extended (not rewritten) to accept
an optional `scorer`/`task_vocab` — pass both to use Phase 5's
score-guided expansion instead of the Phase 2 fixed-depth baseline, leave
both `None` for the original behavior. Both paths produce the same
`ContextBundle` shape (now tagged with `expansion_strategy` so runs are
distinguishable), which is what let this wiring be additive rather than a
rewrite of `ablation_engine.py`, `context_ledger.py`, or anything else
downstream.

**`scorer/feedback_loop.py`** runs one full loop iteration: compile a
bundle for each of the 10 auto-generated tasks both ways (baseline vs.
score-guided), run ablation on both, and compare mean precision/recall.

**One real result, run on this repo:**

| | feasible (of 10) | mean precision | mean recall |
|---|---|---|---|
| Baseline (fixed-depth) | 2 | 0.32 | 0.81 |
| Score-guided, round 1 (original ~100-example training set) | 2 | 0.24 | 0.65 |
| Score-guided, round 2 (retrained on round-1 + round-2 combined data — round 2 generated from the scorer's *own* score-guided bundles) | 4 | 0.31 | 0.78 |

**The honest read of this**: round 1 underperformed the baseline (expected
— the scorer was trained on ~100 examples). One feedback iteration closed
most of that gap (precision and recall both moved most of the way back to
baseline levels, feasible-task count doubled) — real, measurable evidence
the feedback loop *works as a mechanism*. It did not yet *surpass* the
baseline. That's a legitimate, reportable finding for a portfolio project:
"the loop demonstrably improves the scorer" is a true and useful claim;
"the learned scorer beats the hand-tuned baseline" is not yet — don't
overstate it in a writeup.

### A choice worth flagging: combined vs. replaced training data

Round 2's retraining used **round 1 + round 2 examples combined**, not
round 2 alone. This was a deliberate choice made when the loop was run —
replacing entirely would have shrunk an already-tiny dataset (round 2
alone produced only 83 examples, fewer than round 1's 108, since a
different bundle selection changes which edges get labeled). Accumulating
data across iterations is also the more defensible general practice.
Worth being aware this decision was made in the code, not left implicit.

### Try it

```bash
cd scorer
python -m context_compiler.scorer.feedback_loop ./path/to/some/repo ./edge_scorer.pt 2000
```

Prints per-task precision/recall for both strategies, then the means. To
run a full retraining round: regenerate training data with
`build_and_save_dataset(..., scorer=..., task_vocab=...)` (see
`build_edge_training_data.py`), combine with prior round(s), retrain via
`train_edge_scorer.py`, then rerun `feedback_loop.py` with the new
checkpoint to see whether it helped.

### Known caveats (Phase 6)

- **One iteration only**, per the scoping decision made before Phase 4
  started (see design.md section 8 and HANDOFF.md) — not iterated to
  convergence. A real research project would run several rounds and plot
  the trend; this proves the mechanism works for one round.
- **Still bounded by Phase 4's proxy-check circularity** (see
  `ablation/proxy_check.py`) — every round of this loop is training toward
  the same structural notion of "relevant," not an independent signal.
  More rounds without a better proxy would likely plateau, not keep
  improving.
- **Small task/data scale throughout** — 10 auto-generated tasks, ~100-200
  examples per round. The *trend* (round 2 beating round 1) is real and
  reproducible on this repo; the specific numbers would look different on
  a different/larger codebase.

### Known caveats (Phase 1 — real limitations, not silent ones)

- **Call edges are name-based, not scope-resolved.** `foo()` links to every
  symbol named `foo` in the repo — this causes false positives when names
  are reused (e.g. multiple classes with a `run` method, or calls to
  external library functions that happen to share a name with a local
  symbol — see the `__init__`/`Embedding`/`Linear` noise in the sample
  output when parsing a PyTorch-heavy repo). A real fix needs scope-aware
  resolution (e.g. via `jedi`) — noted as a follow-up, not solved here.
- **Import resolution** checks both proper package-relative paths and
  same-directory sibling imports (common in flatly-organized scripts), but
  won't handle more exotic import setups (namespace packages, sys.path
  hacks, dynamic imports).
- **Co-change edges** come from `git log --name-only` over the last N
  commits (default 200) — meaningful on repos with real history, empty on
  fresh/small repos.
- **Test-to-source mapping** is import-based only ("this test file imports
  module X") — no real coverage-tool ground truth.

## Repo layout

```text
context_compiler/
├── parser/      # phase 1: tree-sitter AST extraction
├── graph/       # phase 1: NetworkX MultiDiGraph builder
├── compiler/    # phase 2: BFS expansion, token-budget greedy fill
├── ledger/      # phase 3: tracks compiler selections + agent pulls
├── ablation/    # phase 4: finds minimal-sufficient context labels
└── scorer/      # phase 5: learned edge scorer and feedback loop
_legacy/         # Original ast-graph-transformer reference code
```

See **[docs/HANDOFF.md](docs/HANDOFF.md)** for a full file-by-file manifest and what to build next.

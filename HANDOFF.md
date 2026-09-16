# Handoff — Context Compiler Project

Read this first if you're picking this project back up. All 7 phases are
built and tested — this file tells you what exists, what's actually been
verified (vs. just written), and what would be genuine future work if you
want to keep extending it (see the end of this file).

## Where the AST/graph-transformer files went

**Update: this section's original plan changed once Phase 5 was actually
built — read to the end.** The original `ast-graph-transformer` project
(parse_ast.py, dataset.py, graph_transformer.py, baselines.py, train.py,
visualize_attention.py) was copied as-is into `scorer/` in *this* repo,
with the plan to reuse it for Phase 5. That didn't end up happening: dense
whole-graph attention (what `graph_transformer.py` does) doesn't fit
"score one candidate edge during traversal," so Phase 5 built a new, much
smaller edge-scoring model instead (`edge_scorer.py`). **None of the
original copied files are imported or used by anything in this project.**
They're historical/reference material at this point — worth either
deleting from `scorer/` so the directory reflects what's actually live, or
keeping as a visible "considered this architecture, here's why it didn't
fit" artifact for a portfolio reviewer. Your call.

The two projects are otherwise independent:
- `ast-graph-transformer/` — standalone bug-detection project, complete on
  its own terms (parser → graph → transformer/GCN/GAT → training → attention viz)
- `context-compiler/` — this project, which copied those files into
  `scorer/` early on but ended up not using them (see above)

## Full file manifest

```
context-compiler/
├── README.md                    status table + usage — keep this in sync as phases land
├── requirements.txt              deps for phases 1, 2, and 5 (scorer)
├── docs/
│   ├── design.md                 the full SDD — architecture, all 7 phases, resolved decisions
│   └── HANDOFF.md                this file
├── parser/
│   └── repo_parser.py            Phase 1 — walks a repo, extracts symbols/imports/calls/tests
├── graph/
│   └── code_graph.py             Phase 1 — builds + saves/loads the versioned NetworkX graph
├── compiler/
│   ├── seed_retrieval.py         Phase 2 — lexical task→node matching
│   ├── expansion.py               Phase 2 — fixed-depth BFS from seeds
│   ├── redundancy.py             Phase 2 — drops whole-file entries superseded by symbols
│   ├── token_budget.py           Phase 2 — reads source, estimates tokens, greedy budget fill
│   ├── context_compiler.py       Phase 2/6 — orchestrates the above into one ContextBundle; accepts optional scorer/task_vocab to use score-guided expansion instead of the baseline
│   └── score_guided_expansion.py Phase 5 — best-first expansion using the trained edge scorer; same interface as expansion.py's fixed_depth_expansion; wired into context_compiler.py as of Phase 6
├── ledger/
│   ├── minimal_agent.py          Phase 3 — stand-in agent loop, produces a ToolCall log (not a real agent integration)
│   └── context_ledger.py         Phase 3 — unions compiler+agent context, JSONL storage, run diffing
├── ablation/
│   ├── tasks.py                   Phase 4 — auto-generates a task set from the graph's own dependency structure
│   ├── proxy_check.py            Phase 4 — structural dependency-completeness check (NOT a real linter — read its docstring)
│   └── ablation_engine.py        Phase 4 — greedy removal, precision/recall of compiler selection vs. true requirement
├── scorer/
│   ├── build_edge_training_data.py  Phase 5 — combines ablation labels + edge-tracked expansion into training examples
│   ├── edge_scorer.py               Phase 5 — the model: categorical embeddings + bag-of-words task embedding + small MLP
│   ├── train_edge_scorer.py         Phase 5 — training loop, class-imbalance weighted
│   ├── feedback_loop.py             Phase 6 — regenerates training data from score-guided bundles, compares baseline vs. score-guided precision/recall
│   ├── parse_ast.py, dataset.py, graph_transformer.py, baselines.py,
│   │   train.py, visualize_attention.py   — original ast-graph-transformer files, carried over, NOT used by the edge scorer (see below)
├── scorer/                       Phase 5 (not yet wired in) — copied from ast-graph-transformer
│   ├── parse_ast.py              tree-sitter function-level AST→graph (3 edge types)
│   ├── dataset.py                PyG Data conversion, SPD precomputation
│   ├── graph_transformer.py      Graphormer-style model + GraphAttentionBias
│   ├── baselines.py              GCN/GAT baselines
│   ├── train.py                  training loop, supports edge-type ablation
│   └── visualize_attention.py    attention-over-AST visualization
├── ledger/                       Phase 3 — empty, not yet built
├── ablation/                     Phase 4 — empty, not yet built
└── tests/                        empty, not yet populated
```

## What's actually been tested (not just written)

Every file below was run against a real repo (the `ast-graph-transformer`
project itself, git-initialized so commit-hash versioning could be
verified) during this build:

- `repo_parser.py` — ran `parse_repo()`, inspected symbols/imports/calls/
  inherits by hand. Found and fixed a real bug: sibling-style imports
  (`from dataset import X` with no package prefix) weren't resolving —
  fixed by also checking same-directory-as-importer.
- `code_graph.py` — ran `build_code_graph()`, verified node/edge type
  counts look sane, verified `save_graph()`/`load_graph()` roundtrip
  preserves all node/edge attributes and the commit-hash key.
- `seed_retrieval.py` — ran against the query "graph transformer attention
  bias", confirmed `GraphAttentionBias` correctly ranks #1.
- `expansion.py` — ran 2-hop expansion from those seeds, confirmed it
  reaches related classes/functions and (correctly, per the design's own
  critique of fixed-depth BFS) picks up some tangential nodes via
  co-change edges — that noise is expected and is the thing Phase 5 should
  reduce.
- `token_budget.py` — confirmed budget is respected, greedy density-first
  ordering behaves as intended.
- `redundancy.py` — confirmed whole-file entries get dropped when specific
  symbols from that file are already candidates.
- `context_compiler.py` — ran the full pipeline end-to-end on two
  different queries, confirmed `.render()` produces real, readable,
  labeled source snippets — the actual thing a coding agent would receive
  as context.
- `minimal_agent.py` — ran against a compiled bundle with two different
  random seeds; confirmed it produces zero tool calls on one seed and a
  real extra `read_symbol` pull on another, so the "sometimes pulls extra
  context" branch is exercised, not just the trivial path.
- `context_ledger.py` — ran two full runs (different queries) into the
  same JSONL ledger, confirmed both records parse back as valid JSON,
  confirmed `tokens_by_source` correctly splits compiler vs. agent tokens,
  and confirmed `diff_runs()` produces a sane only-in-A/only-in-B/in-both
  breakdown between the two runs.
- `tasks.py` — ran auto-generation, caught and fixed a real bug: task
  descriptions for methods only used the method name (e.g. "init"),
  making several tasks lexically indistinguishable to seed retrieval —
  fixed by including the parent class name in the description.
- `ablation_engine.py` — ran across all 10 auto-generated tasks on the
  sample repo. Caught and fixed a real bug here too: the initial version
  hardcoded precision/recall to 0.0 whenever the compiler's selection
  didn't fully satisfy the proxy check, which hid how close the compiler
  actually got. Fixed to always measure against the true requirement
  (`{target} ∪ required_dependencies`), so precision/recall are meaningful
  whether or not the task was "feasible." Also hand-verified one feasible
  task's greedy-removal trace by eye and confirmed the `kept` items were
  exactly that target's real dependencies — the mechanism does what it
  claims.

**Not tested**: multi-repo runs, very large repos (the code graph is built
fully in memory — fine for small/medium repos, untested at scale), edge
cases like empty repos or repos with syntax-broken files beyond the
try/except already in `repo_parser.py`, ledger behavior at high run volume
(hundreds+ of records in one JSONL file), everything in `ledger/` was only
tested against the *stand-in* agent never a real one, and the ablation
engine has only been run against auto-generated tasks — never against a
hand-curated task with independently-verified ground truth.

**Phase 5, additionally tested:**
- `fixed_depth_expansion_with_edges` (added to `compiler/expansion.py`) —
  confirmed by inspection that parent/edge_type/hop attribution is correct.
- `build_edge_training_data.py` — ran across all 10 auto-generated tasks,
  produced 108 examples, confirmed the label-balance printout (21%
  positive) matches the expected pattern (most compiler-selected padding
  gets ablated away).
- `train_edge_scorer.py` — ran a full training loop, confirmed loss
  decreases over epochs and the checkpoint saves/reloads correctly.
- `score_guided_expansion.py` — loaded the trained checkpoint, ran
  best-first traversal on a real query, confirmed `max_nodes` is respected
  and the heap-based priority ordering behaves as designed.

**Not tested (Phase 5 specifically)**: whether the learned scores are
actually *good* — expected, given the tiny training set (see caveats
below). Also not tested: actually swapping `score_guided_expansion` into
`context_compiler.py` in place of `fixed_depth_expansion` — built to the
same interface, but the swap itself and the resulting end-to-end pipeline
haven't been run.

## What Phase 5 (learned relevance scorer) needed — resolved

This section is kept for history. The decision made: **edge-scoring**, not
per-node scoring — see the reasoning captured in `README.md`'s Phase 5
section. Built: `scorer/build_edge_training_data.py` (turns Phase 4
ablation labels into edge-level examples), `scorer/edge_scorer.py` (a new,
small categorical-embedding + MLP model — NOT a reuse of
`graph_transformer.py`, which doesn't fit single-edge scoring), and
`compiler/score_guided_expansion.py` (best-first traversal driven by the
trained scorer, same interface as `fixed_depth_expansion`).

**Known limitations carried into Phase 6**, not yet resolved:
- Trained on ~100 examples from 10 tasks — a proof-of-mechanism, not
  enough data for the rankings to be trustworthy.
- Inherits Phase 4's proxy-check circularity (see `ablation/proxy_check.py`)
  — the scorer is learning to predict direct-dependency graph edges, a
  real but narrow notion of relevance.
- Edge-type selection when multiple edge types connect the same node pair
  is arbitrary in the training-data builder — picks one, doesn't consider
  all. Worth fixing before generating more training data.

## What Phase 6 (feedback loop) needed — resolved, with a real result

`score_guided_expansion` was wired into `context_compiler.py` (additive —
pass `scorer`/`task_vocab` to opt in, `None`/`None` keeps the Phase 2
baseline behavior). One full loop iteration was run and measured:

| | feasible (/10) | mean precision | Phase | Status | Key Files |
|-------|--------|-----------|
| **1: Parser & Graph** | Done | `context_compiler/parser/repo_parser.py`, `context_compiler/graph/code_graph.py` |
| **2: Baseline Compiler** | Done | `context_compiler/compiler/context_compiler.py` |
| **3: Context Ledger** | Done | `context_compiler/ledger/context_ledger.py` |
| **4: Ablation Engine** | Done | `context_compiler/ablation/ablation_engine.py` |
| **5: Edge Scorer** | Done | `context_compiler/scorer/edge_scorer.py`, `context_compiler/scorer/train_edge_scorer.py` |
| **6: Feedback Loop** | Done | `context_compiler/scorer/feedback_loop.py` |
| **7: Benchmark** | Done | `BENCHMARK.md`, `context_compiler/scorer/generate_benchmark_data.py` |

## History

- **Phase 1-7 complete**: Built the end-to-end pipeline from parser to feedback loop.
- **Project Structure Reorganized**: Migrated the flat directory into proper `context_compiler/` sub-packages, quarantined legacy code to `_legacy/`, and cleaned up all `sys.path` import hacks.closed most of the gap between the scorer and the baseline. It did
not surpass the baseline. Don't oversell this in the eventual writeup:
"the loop improves the scorer" is supported; "the learned approach beats
the hand-tuned one" is not, yet. More rounds might close the rest of the
gap, might plateau (see the proxy-circularity caveat below), or might need
more training tasks per round to keep improving — genuinely unknown
without running more rounds, which is explicitly out of scope here (see
design.md section 8).

**One implementation choice worth flagging**: round 2's retraining used
round-1 + round-2 examples *combined*, not round-2 alone — round 2 alone
only produced 83 examples (fewer than round 1's 108), and replacing
entirely would have shrunk an already-small dataset further. This was a
deliberate choice, not a default; see `README.md`'s Phase 6 section for
the same note.

## What Phase 7 (benchmark writeup) needed — resolved

Written: **[docs/BENCHMARK.md](BENCHMARK.md)**, generated from real data
via `scorer/generate_benchmark_data.py` (not invented figures). Covers the
Phase 6 comparison table, a token-cost-by-source table (with an honest
caveat that the downward agent-token trend isn't a verified causal
finding — see the file), an explicit note that the full-dump and
embedding-RAG baselines design.md originally asked for were never built,
and a collected limitations section pulling together caveats already
written throughout this project rather than restating them fresh.

**All 7 phases are now built and tested.** This project is feature-complete
at the scope decided across this build — see docs/BENCHMARK.md's "Honest
overall framing" for what that does and doesn't mean. Anything past this
point (a real agent harness, a stronger proxy check, more feedback
iterations, the missing baselines, a larger task set) is genuine future
work, not something left unfinished by oversight — each was a scoping
decision made and recorded along the way.

## What Phase 3 (context ledger) needed — resolved

This section is kept for history / to show the decision that got made:
Phase 3 used a hand-rolled minimal agent stand-in (`minimal_agent.py`)
rather than integrating a real harness (Claude Code, OpenHands), to avoid
blocking ledger development on that integration work. This is a deliberate
scope decision, not an oversight — swap in a real harness later without
changing `context_ledger.py`, since it only depends on the `ToolCall` shape.

## Resolved decisions already baked into the code (don't re-litigate)

- Ablation (Phase 4) will use a cheap static-check/linter proxy, not full
  agent reruns — per the 8GB local hardware constraint.
- The repo snapshot is frozen for the project's duration — no live
  re-snapshotting logic exists or is planned.
- Phase 3 uses a hand-rolled minimal agent stand-in, not a real agent
  harness integration — see "What Phase 3 needed" above.
- Phase 5's scorer predicts per-edge relevance, not per-node — a new small
  model, not a reuse of `graph_transformer.py` — see "What Phase 5 needed"
  above.
- Phase 6 ran exactly one feedback iteration, retrained on round-1+round-2
  combined data (not round-2 alone) — see "What Phase 6 needed" above for
  the actual result.
- These are recorded in `docs/design.md` section 8 as well.

## Known technical debt (real, not hidden)

- Call/inherit edge resolution in `repo_parser.py` is name-based only, not
  scope-resolved — documented in the file's own docstring and in the
  README, but worth restating: this is the single biggest source of graph
  noise right now (see the `__init__`/`Embedding`/`Linear` false-positive
  calls in the sample output).
- `estimate_tokens()` in `token_budget.py` is a chars/4 approximation, not
  a real tokenizer.
- No tests/ directory populated yet — everything above was verified via
  ad-hoc `python3 -c "..."` smoke tests during the build, not via a
  checked-in test suite. Writing that suite (pytest, using the same
  `ast-graph-transformer` repo as a fixture) is worth doing before adding
  more phases, so regressions in Phase 1/2 get caught automatically as
  Phase 3+ starts depending on them.

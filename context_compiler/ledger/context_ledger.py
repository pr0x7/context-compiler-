"""
Phase 3: Context Ledger.

Per design.md 4.5: logs the UNION of compiler-selected and agent-pulled
context, not just the compiler's output — this is what makes it an
observability tool rather than a compiler log.

Storage: JSONL, one line per run, keyed by (task ID, commit hash, run ID) —
chosen over SQLite per HANDOFF.md's guidance (simpler, append-only, good
enough at portfolio scale; revisit if you need cross-run SQL queries).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from context_compiler.compiler.context_compiler import ContextBundle
from context_compiler.ledger.minimal_agent import AgentRunResult


@dataclass
class LedgerContextItem:
    node_id: str
    source_origin: str      # "compiler" | "agent"
    score: float | None     # None for agent-pulled items (no compiler score)
    tokens: int
    timestamp: float


@dataclass
class LedgerRecord:
    task_id: str
    commit_hash: str
    run_id: str
    task_description: str
    compiler_version: str
    context_items: list[LedgerContextItem] = field(default_factory=list)
    tokens_by_source: dict[str, int] = field(default_factory=dict)
    compaction_events: list[dict] = field(default_factory=list)  # not yet populated — no compaction logic exists yet, kept as an honest empty hook
    success: bool = False
    created_at: float = field(default_factory=time.time)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


COMPILER_VERSION = "phase2-baseline-v1"  # bump this string whenever compiler/ logic changes materially


def build_ledger_record(
    task_id: str,
    bundle: ContextBundle,
    agent_run: AgentRunResult,
) -> LedgerRecord:
    """
    Union compiler-selected (from `bundle`) and agent-pulled (from
    `agent_run.tool_calls`) context into one record, each item tagged with
    its source_origin so later analysis can split token cost / precision
    by source (per design.md 4.7's "token cost split compiler vs. agent-pulled").
    """
    compiler_start_time = agent_run.tool_calls[0].timestamp if agent_run.tool_calls else time.time()

    items: list[LedgerContextItem] = []
    for s in bundle.selected:
        items.append(LedgerContextItem(
            node_id=s.node_id, source_origin="compiler", score=s.score,
            tokens=s.tokens, timestamp=compiler_start_time,
        ))
    for tc in agent_run.tool_calls:
        node_id = tc.args.get("node_id", "?")
        tokens = 0
        if "read" in tc.result_summary and "tokens" in tc.result_summary:
            try:
                tokens = int(tc.result_summary.split()[1])
            except (IndexError, ValueError):
                tokens = 0
        items.append(LedgerContextItem(
            node_id=node_id, source_origin="agent", score=None,
            tokens=tokens, timestamp=tc.timestamp,
        ))

    tokens_by_source = {"compiler": 0, "agent": 0}
    for item in items:
        tokens_by_source[item.source_origin] += item.tokens

    return LedgerRecord(
        task_id=task_id,
        commit_hash=bundle.commit_hash,
        run_id=agent_run.run_id,
        task_description=bundle.task_description,
        compiler_version=COMPILER_VERSION,
        context_items=items,
        tokens_by_source=tokens_by_source,
        compaction_events=[],
        success=agent_run.success,
    )


def append_to_ledger(record: LedgerRecord, ledger_path: str) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.to_json_line() + "\n")


def read_ledger(ledger_path: str) -> list[dict]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def diff_runs(record_a: dict, record_b: dict) -> dict:
    """
    Step-by-step diff between two ledger records for the same task —
    per design.md 4.5's requirement that runs be diffable. Returns which
    nodes were only in A, only in B, and in both, plus a token delta.
    """
    ids_a = {item["node_id"] for item in record_a["context_items"]}
    ids_b = {item["node_id"] for item in record_b["context_items"]}

    total_tokens_a = sum(record_a["tokens_by_source"].values())
    total_tokens_b = sum(record_b["tokens_by_source"].values())

    return {
        "only_in_a": sorted(ids_a - ids_b),
        "only_in_b": sorted(ids_b - ids_a),
        "in_both": sorted(ids_a & ids_b),
        "total_tokens_a": total_tokens_a,
        "total_tokens_b": total_tokens_b,
        "token_delta": total_tokens_b - total_tokens_a,
        "success_a": record_a["success"],
        "success_b": record_b["success"],
    }


if __name__ == "__main__":
    import sys

    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo
    from context_compiler.compiler.context_compiler import compile_context
    from context_compiler.ledger.minimal_agent import run_minimal_agent

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else "graph transformer attention bias"
    ledger_path = sys.argv[3] if len(sys.argv) > 3 else "./context_ledger.jsonl"

    g = build_code_graph(parse_repo(repo))
    task_id = f"task-{uuid.uuid4().hex[:8]}"

    bundle = compile_context(g, repo, query, token_budget=1500)
    agent_run = run_minimal_agent(g, repo, bundle, seed=7)
    record = build_ledger_record(task_id, bundle, agent_run)
    append_to_ledger(record, ledger_path)

    print(f"Logged run {record.run_id} for task {task_id}")
    print(f"  compiler items: {sum(1 for i in record.context_items if i.source_origin == 'compiler')}")
    print(f"  agent items:    {sum(1 for i in record.context_items if i.source_origin == 'agent')}")
    print(f"  tokens by source: {record.tokens_by_source}")
    print(f"  success: {record.success}")

    all_records = read_ledger(ledger_path)
    print(f"\nLedger now has {len(all_records)} total record(s) at {ledger_path}")

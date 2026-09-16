"""
Phase 3 prerequisite: a minimal agent harness with a tool-call log.

Per design.md 4.4/HANDOFF.md, the ledger needs *something* that exposes a
tool-call log — what the agent read/searched beyond what the compiler gave
it. Rather than integrating a real agent (Claude Code, OpenHands — real
integration work, deferred), this is a deliberately minimal stand-in:

  - It receives a ContextBundle (from Phase 2) as its starting context.
  - It can call exactly one tool: `read_symbol(node_id)` — pull one more
    symbol's source from the code graph, simulating an agent deciding the
    compiler didn't give it enough.
  - Every tool call is logged with a timestamp.
  - It "succeeds" or "fails" based on a trivial rule (see below) — this
    project isn't about building a good agent, it's about proving the
    ledger correctly unions compiler-selected + agent-pulled context.

Swap this out for a real harness later; the ledger code doesn't care which
one produced the tool-call log, as long as it's in this shape.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import networkx as nx

from context_compiler.compiler.context_compiler import ContextBundle
from context_compiler.compiler.token_budget import get_node_source, estimate_tokens


@dataclass
class ToolCall:
    tool_name: str
    args: dict
    result_summary: str
    timestamp: float


@dataclass
class AgentRunResult:
    task_description: str
    context_bundle: ContextBundle
    tool_calls: list[ToolCall] = field(default_factory=list)
    success: bool = False
    run_id: str = ""


def run_minimal_agent(
    graph: nx.MultiDiGraph,
    repo_root: str,
    bundle: ContextBundle,
    max_extra_pulls: int = 3,
    seed: int | None = None,
) -> AgentRunResult:
    """
    Simulated agent loop:
      1. Starts with `bundle`'s selected context.
      2. With some probability, decides it needs more and calls
         `read_symbol` on a node connected to something already in context
         but not yet included (a stand-in for "the agent greps/follows an
         import it wasn't given").
      3. "Succeeds" if, after its pulls, some minimum coverage heuristic is
         met (placeholder — a real harness would run tests/a checker).

    This is intentionally simple. The point of this module is to produce a
    realistic-shaped tool-call log for the ledger to consume, not to be a
    good agent.
    """
    rng = random.Random(seed)
    run_id = f"run-{int(time.time() * 1000)}-{rng.randint(1000, 9999)}"
    result = AgentRunResult(task_description=bundle.task_description, context_bundle=bundle, run_id=run_id)

    selected_ids = {s.node_id for s in bundle.selected}
    undirected = graph.to_undirected(as_view=True)

    # candidate "extra" pulls: neighbors of selected nodes not already selected
    candidates: set[str] = set()
    for node_id in selected_ids:
        if node_id not in undirected:
            continue
        for neighbor in undirected.neighbors(node_id):
            attrs = undirected.nodes[neighbor]
            if neighbor in selected_ids:
                continue
            if attrs.get("node_type") in ("function", "method", "class"):
                candidates.add(neighbor)

    candidates_list = sorted(candidates)
    rng.shuffle(candidates_list)

    num_pulls = rng.randint(0, max_extra_pulls)
    for node_id in candidates_list[:num_pulls]:
        source = get_node_source(graph, repo_root, node_id)
        tokens = estimate_tokens(source) if source else 0
        result.tool_calls.append(ToolCall(
            tool_name="read_symbol",
            args={"node_id": node_id},
            result_summary=f"read {tokens} tokens" if source else "not found",
            timestamp=time.time(),
        ))
        time.sleep(0.001)  # ensure distinct timestamps for the ledger's ordering

    # placeholder success rule: "succeeded" if it ended up with at least 3
    # total context items (compiler + agent-pulled) — stand-in for a real
    # test/checker, matching the "cheap proxy" decision for ablation later
    total_context_items = len(selected_ids) + len(result.tool_calls)
    result.success = total_context_items >= 3

    return result


if __name__ == "__main__":
    import sys

    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo
    from context_compiler.compiler.context_compiler import compile_context

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else "graph transformer attention bias"

    g = build_code_graph(parse_repo(repo))
    bundle = compile_context(g, repo, query, token_budget=1500)
    run = run_minimal_agent(g, repo, bundle, seed=42)

    print(f"Run: {run.run_id}")
    print(f"Compiler selected: {len(bundle.selected)} nodes")
    print(f"Agent pulled {len(run.tool_calls)} extra nodes:")
    for tc in run.tool_calls:
        print(f"  {tc.tool_name}({tc.args}) -> {tc.result_summary}")
    print(f"Success: {run.success}")

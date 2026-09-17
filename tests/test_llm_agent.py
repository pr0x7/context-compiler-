"""
Tests for context_compiler/ledger/llm_agent.py.
"""

from __future__ import annotations

import pytest
import networkx as nx

from context_compiler.parser.repo_parser import parse_repo
from context_compiler.graph.code_graph import build_code_graph
from context_compiler.compiler.context_compiler import compile_context
from context_compiler.ledger.llm_agent import run_llm_agent
from context_compiler.ledger.context_ledger import build_ledger_record


@pytest.fixture
def sample_graph(tmp_path):
    # Create a small dummy Python file structure for testing
    d = tmp_path / "sample_pkg"
    d.mkdir()
    f1 = d / "mod_a.py"
    f1.write_text("def func_a():\n    return 42\n")
    f2 = d / "mod_b.py"
    f2.write_text("from sample_pkg.mod_a import func_a\n\ndef func_b():\n    return func_a() + 1\n")

    records = parse_repo(str(tmp_path))
    graph = build_code_graph(records)
    return graph, str(tmp_path)


def test_run_llm_agent_fallback_mode(sample_graph):
    graph, repo_root = sample_graph
    bundle = compile_context(graph, repo_root, "func_b", token_budget=1000)

    # Force empty API key to test dry-run fallback mode
    run_res = run_llm_agent(graph, repo_root, bundle, api_key="", seed=42)

    assert run_res.task_description == bundle.task_description
    assert isinstance(run_res.tool_calls, list)
    assert run_res.run_id.startswith("run-llm-")

    # Ensure ledger record construction succeeds with LLM run result
    record = build_ledger_record("test-task", bundle, run_res)
    assert record.tokens_by_source["compiler"] > 0
    assert "agent" in record.tokens_by_source

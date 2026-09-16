import json
from context_compiler.ledger.context_ledger import build_ledger_record, append_to_ledger, read_ledger, diff_runs
from context_compiler.compiler.context_compiler import ContextBundle
from context_compiler.compiler.token_budget import SelectedNode
from context_compiler.ledger.minimal_agent import AgentRunResult, ToolCall

def test_ledger_record_and_io(tmp_path):
    # Setup dummy data
    bundle = ContextBundle(
        task_description="dummy task",
        commit_hash="abc",
        token_budget=100,
        selected=[
            SelectedNode("node1", 0.9, 10, "source1"),
            SelectedNode("node2", 0.8, 15, "source2")
        ]
    )
    
    agent_run = AgentRunResult(
        task_description="dummy task",
        context_bundle=bundle,
        tool_calls=[
            ToolCall("read_symbol", {"node_id": "node3"}, "read 20 tokens", 123456.0)
        ],
        success=True,
        run_id="run-1"
    )
    
    record = build_ledger_record("task-1", bundle, agent_run)
    
    assert record.task_id == "task-1"
    assert record.tokens_by_source["compiler"] == 25
    assert record.tokens_by_source["agent"] == 20
    
    # Test IO
    ledger_file = tmp_path / "ledger.jsonl"
    append_to_ledger(record, str(ledger_file))
    
    records = read_ledger(str(ledger_file))
    assert len(records) == 1
    assert records[0]["task_id"] == "task-1"
    assert records[0]["success"] is True

def test_diff_runs():
    record_a = {
        "context_items": [{"node_id": "a"}, {"node_id": "b"}],
        "tokens_by_source": {"compiler": 10, "agent": 5},
        "success": True
    }
    record_b = {
        "context_items": [{"node_id": "b"}, {"node_id": "c"}],
        "tokens_by_source": {"compiler": 10, "agent": 10},
        "success": False
    }
    
    diff = diff_runs(record_a, record_b)
    assert diff["only_in_a"] == ["a"]
    assert diff["only_in_b"] == ["c"]
    assert diff["in_both"] == ["b"]
    assert diff["token_delta"] == 5

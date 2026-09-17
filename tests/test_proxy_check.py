"""
Tests for upgraded proxy_check.py AST and Pyflakes analysis.
"""

from __future__ import annotations

import pytest
from context_compiler.parser.repo_parser import parse_repo
from context_compiler.graph.code_graph import build_code_graph
from context_compiler.ablation.tasks import AblationTask
from context_compiler.ablation.proxy_check import check_context_sufficient, _extract_ast_loaded_names


def test_ast_loaded_names_extraction():
    code = """
def process_data(item):
    val = compute_score(item)
    return FormatHelper.format(val)
"""
    names = _extract_ast_loaded_names(code)
    assert "compute_score" in names
    assert "FormatHelper" in names
    assert "item" in names


def test_proxy_check_with_missing_dependency(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    f1 = d / "helper.py"
    f1.write_text("def helper_func():\n    return 100\n")
    f2 = d / "main.py"
    f2.write_text("from pkg.helper import helper_func\n\ndef main_func():\n    return helper_func() + 5\n")

    records = parse_repo(str(tmp_path))
    graph = build_code_graph(records)

    target_id = [n for n in graph.nodes if "main_func" in n][0]
    helper_id = [n for n in graph.nodes if "helper_func" in n][0]

    task = AblationTask(
        task_id="t1",
        task_description="test main_func",
        target_node=target_id,
        required_dependencies={helper_id},
    )

    # 1. Target node only (missing helper_func) -> Should return False
    assert check_context_sufficient(graph, str(tmp_path), task, {target_id}) is False

    # 2. Both target node and helper_func in context -> Should return True
    assert check_context_sufficient(graph, str(tmp_path), task, {target_id, helper_id}) is True

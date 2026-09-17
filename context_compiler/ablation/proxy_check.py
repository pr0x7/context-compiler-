"""
Phase 4: Proxy check.

AST & Pyflakes Static Analysis Completeness Check.

"Success" means:
1. The task's target node is included in the context.
2. An AST traversal of the target node's source code verifies that all loaded
   identifiers (function calls, variable lookups, class references) belonging to
   the repository are present in the candidate context.
3. A Pyflakes static analysis pass over the synthesized context module finds no
   missing internal repository definitions.
"""

from __future__ import annotations

import ast
import re
from io import StringIO
from typing import Any

import networkx as nx
from pyflakes.api import check
from pyflakes.reporter import Reporter

from context_compiler.ablation.tasks import AblationTask
from context_compiler.compiler.token_budget import get_node_source


class _ASTNameVisitor(ast.NodeVisitor):
    """AST Visitor to extract all loaded variable, function, and class names."""

    def __init__(self) -> None:
        self.loaded_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Also capture attribute names if they match symbol names
        self.loaded_names.add(node.attr)
        self.generic_visit(node)


def _extract_ast_loaded_names(source: str) -> set[str]:
    """Parse python source text into AST and return all loaded identifier names."""
    try:
        # Dedent or wrap if necessary to parse standalone methods/functions
        tree = ast.parse(source)
    except SyntaxError:
        try:
            tree = ast.parse(f"class _DummyScope:\n" + "\n".join("    " + line for line in source.splitlines()))
        except SyntaxError:
            return set()

    visitor = _ASTNameVisitor()
    visitor.visit(tree)
    return visitor.loaded_names


def check_context_sufficient(
    graph: nx.MultiDiGraph,
    repo_root: str,
    task: AblationTask,
    context_node_ids: set[str],
) -> bool:
    """
    True iff the target node is present AND AST name extraction + pyflakes
    static analysis confirms no required internal repository symbols are missing.
    """
    if task.target_node not in context_node_ids:
        return False

    # Extract short names of all internal symbols defined in the repo graph
    internal_short_names: set[str] = set()
    for node in graph.nodes:
        if not str(node).startswith("external:"):
            short = str(node).split(".")[-1]
            if short and not short.startswith("__"):
                internal_short_names.add(short)

    # Extract short names present in the current candidate context
    context_short_names: set[str] = set()
    for node in context_node_ids:
        if not str(node).startswith("external:"):
            short = str(node).split(".")[-1]
            context_short_names.add(short)

    # 1. AST Free-Variable Check on the target node
    target_source = get_node_source(graph, repo_root, task.target_node)
    if target_source:
        loaded_names = _extract_ast_loaded_names(target_source)
        for name in loaded_names:
            if name in internal_short_names and name not in context_short_names:
                # Target references an internal repo symbol that isn't in context!
                return False

    # 2. Pyflakes Static Analysis Check on the Synthesized Module
    snippets = []
    for node_id in context_node_ids:
        source = get_node_source(graph, repo_root, node_id)
        if source:
            snippets.append(source)
    synthetic_code = "\n\n".join(snippets)

    out = StringIO()
    err = StringIO()
    reporter = Reporter(out, err)

    try:
        check(synthetic_code, "synthetic_module.py", reporter)
    except SyntaxError:
        # Ignore syntax errors from method concatenations; AST check above already validated names
        pass

    output = out.getvalue()
    for line in output.splitlines():
        if "undefined name" in line:
            match = re.search(r"undefined name '([^']+)'", line)
            if match:
                missing_name = match.group(1)
                if (
                    missing_name in internal_short_names
                    and missing_name not in context_short_names
                ):
                    return False

    return True

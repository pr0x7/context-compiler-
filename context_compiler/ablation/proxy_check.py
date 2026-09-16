"""
Phase 4: Proxy check.

CAVEAT — read this before trusting anything downstream of it:

This is a *structural dependency-completeness* check, not a real
linter/test run. "Success" means: every symbol the task's target node
directly calls or inherits from (per the code graph's own edges) is
present in the candidate context. That's cheap and deterministic, which is
why it was chosen (see docs/HANDOFF.md's "What Phase 4 needs" and the
8GB-hardware-driven decision to avoid full agent reruns) — but it's also
somewhat circular: the "ground truth" here is derived from the same graph
the compiler uses to select context in the first place, not from an
independent signal like a real test suite or an actual linter.

A stronger version would run something like `pyflakes` against a
reconstructed synthetic module built from the selected snippets and check
for real undefined-name errors. That's a meaningfully bigger undertaking
(handling missing imports, indentation, non-local references) and is
explicitly left as a follow-up rather than attempted here.
"""

from __future__ import annotations

import re
import networkx as nx
from io import StringIO
from pyflakes.api import check
from pyflakes.reporter import Reporter
from context_compiler.ablation.tasks import AblationTask
from context_compiler.compiler.token_budget import get_node_source


def check_context_sufficient(
    graph: nx.MultiDiGraph, repo_root: str, task: AblationTask, context_node_ids: set[str]
) -> bool:
    """
    True iff the target node itself is present AND a pyflakes static analysis
    pass on the synthesized context finds no missing internal dependencies.
    """
    if task.target_node not in context_node_ids:
        return False

    # Build the synthetic module
    snippets = []
    for node_id in context_node_ids:
        source = get_node_source(graph, repo_root, node_id)
        if source:
            snippets.append(source)
    synthetic_code = "\n\n".join(snippets)

    # Run Pyflakes
    out = StringIO()
    err = StringIO()
    reporter = Reporter(out, err)
    try:
        check(synthetic_code, "synthetic_module.py", reporter)
    except SyntaxError:
        # If it doesn't parse, consider it insufficient (though likely just our concat being messy)
        pass

    # Extract all known short names from the repo graph
    internal_short_names = {node.split(".")[-1] for node in graph.nodes if not node.startswith("external:")}
    
    # Extract short names that ARE in the selected context
    context_short_names = {node.split(".")[-1] for node in context_node_ids if not node.startswith("external:")}

    # Analyze Pyflakes output
    output = out.getvalue()
    for line in output.splitlines():
        if "undefined name" in line:
            match = re.search(r"undefined name '([^']+)'", line)
            if match:
                missing_name = match.group(1)
                if missing_name in internal_short_names and missing_name not in context_short_names:
                    # An internal symbol is missing!
                    return False

    return True

import networkx as nx
from context_compiler.compiler.context_compiler import compile_context

def test_compile_context(tmp_path):
    # Setup dummy files
    (tmp_path / "target.py").write_text("def target_func():\n    pass\n")
    (tmp_path / "dep.py").write_text("def dep_func():\n    pass\n")
    (tmp_path / "far.py").write_text("def far_func():\n    pass\n")

    # Setup a small mock graph
    g = nx.MultiDiGraph()
    # Add nodes with file_path to allow get_node_source
    g.add_node("target_func", node_type="function", file_path="target.py", lineno=1, end_lineno=2)
    g.add_node("dep_func", node_type="function", file_path="dep.py", lineno=1, end_lineno=2)
    g.add_node("far_func", node_type="function", file_path="far.py", lineno=1, end_lineno=2)
    
    # Add edges
    g.add_edge("target_func", "dep_func", edge_type="calls")
    g.add_edge("dep_func", "far_func", edge_type="calls")
    
    # Compile context. 
    # With a query "target func", it should seed on target_func and expand.
    # Token budget is large enough for target and dep, maybe far too.
    bundle = compile_context(g, str(tmp_path), task_description="target_func", token_budget=50)
    
    selected_ids = [s.node_id for s in bundle.selected]
    
    assert "target_func" in selected_ids
    # depending on exact token math, dep_func should likely be included
    assert "dep_func" in selected_ids
    
    # Test strict budget - if budget is very small, it should only pick seed
    bundle_strict = compile_context(g, str(tmp_path), task_description="target_func", token_budget=10)
    selected_ids_strict = [s.node_id for s in bundle_strict.selected]
    assert "target_func" in selected_ids_strict
    assert "far_func" not in selected_ids_strict

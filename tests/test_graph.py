import networkx as nx
from context_compiler.parser.repo_parser import RepoParseResult, Symbol
from context_compiler.graph.code_graph import build_code_graph

def test_build_code_graph():
    # Setup mock parse result
    parse_result = RepoParseResult(
        commit_hash="dummy",
        repo_root="/dummy",
        files=["a.py", "b.py"],
        test_files=["test_a.py"],
        symbols={
            "a.Foo": Symbol("a.Foo", "class", "a.py", 1, 10, None),
            "a.Foo.bar": Symbol("a.Foo.bar", "method", "a.py", 2, 5, "doc"),
            "b.baz": Symbol("b.baz", "function", "b.py", 1, 5, None),
        },
        import_edges=[("a.py", "b.py")],
        call_edges=[("a.Foo.bar", "b.baz")],
        inherit_edges=[],
        test_covers_edges=[("test_a.py", "a.py")],
    )

    graph = build_code_graph(parse_result)
    
    assert isinstance(graph, nx.MultiDiGraph)
    
    # Check nodes
    assert "a.py" in graph.nodes
    assert graph.nodes["a.py"]["node_type"] == "file"
    assert "a.Foo" in graph.nodes
    assert graph.nodes["a.Foo"]["node_type"] == "class"
    
    # Check containment edges (file defines symbols)
    assert graph.has_edge("a.py", "a.Foo")
    assert graph.has_edge("a.py", "a.Foo.bar")
    
    # Check structural edges
    # 'a.py' imports 'b.py'
    edges = graph.get_edge_data("a.py", "b.py")
    assert any(e["edge_type"] == "imports" for e in edges.values())
    
    # 'a.Foo.bar' calls 'b.baz'
    edges = graph.get_edge_data("a.Foo.bar", "b.baz")
    assert any(e["edge_type"] == "calls" for e in edges.values())

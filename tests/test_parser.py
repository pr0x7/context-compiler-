import os
from pathlib import Path
from context_compiler.parser.repo_parser import parse_repo

def test_parser_extracts_symbols_and_edges(tmp_path):
    # Setup dummy repo
    (tmp_path / "main.py").write_text(
        "from utils import helper\n"
        "\n"
        "class Processor:\n"
        "    def run(self):\n"
        "        helper()\n"
    )
    (tmp_path / "utils.py").write_text(
        "def helper():\n"
        "    pass\n"
    )
    (tmp_path / "test_main.py").write_text(
        "from main import Processor\n"
        "def test_processor():\n"
        "    p = Processor()\n"
        "    p.run()\n"
    )
    
    # Initialize a dummy git repo just so _get_commit_hash doesn't fail
    # or just rely on the fallback "no-git-unversioned".
    
    result = parse_repo(str(tmp_path))
    
    # Check files
    assert "main.py" in result.files
    assert "utils.py" in result.files
    assert "test_main.py" in result.files
    assert "test_main.py" in result.test_files
    
    # Check symbols
    assert "main.Processor" in result.symbols
    assert "main.Processor.run" in result.symbols
    assert "utils.helper" in result.symbols
    assert "test_main.test_processor" in result.symbols
    
    # Check import edges (ignoring exact path format, just check filename)
    import_targets = [target for src, target in result.import_edges]
    assert any("utils.py" in t for t in import_targets)
    assert any("main.py" in t for t in import_targets)
    
    # Check call edges (now exactly resolved via Jedi)
    call_pairs = set(result.call_edges)
    assert ("main.Processor.run", "utils.helper") in call_pairs
    assert ("test_main.test_processor", "main.Processor.run") in call_pairs
    
    # Check test-covers edges
    test_covers = [src for src, tgt in result.test_covers_edges]
    assert "test_main.py" in test_covers

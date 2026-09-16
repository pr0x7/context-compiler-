"""
Phase 1: Repository Parser.

Walks a Python repository at a given commit and extracts structured records:
  - files
  - symbols (functions, methods, classes) with location + docstring
  - import edges (file -> file, resolved best-effort to files within the repo)
  - call edges (function -> function, name-based resolution — see caveats)
  - inherits edges (class -> base class)
  - test-to-source mapping (heuristic: which source modules a test file imports)

Uses Python's built-in `ast` module rather than tree-sitter for this level —
we need semantic info (resolved imports, call targets, class hierarchy)
that's easier to get from Python's own AST + symbol resolution than from a
generic tree-sitter parse. The fine-grained per-function graphs used by the
scorer (Phase 5) still use the tree-sitter pipeline from the original
AST-classifier project.

CAVEATS (call these out in the writeup, don't let them be silent limitations):
  - Call edges are resolved by *name only* — `foo()` is linked to every
    symbol named `foo` in the repo, not disambiguated by type/scope. This
    causes false positives when names are reused (e.g. multiple classes
    with a `run` method). A real implementation would need proper scope-aware
    resolution or a tool like Jedi.
  - Import edges only resolve imports that point at files inside the repo
    root (checked both as proper package-relative paths and as
    same-directory sibling imports); external packages (numpy, os, etc.)
    are recorded as edges to an "external:<package>" placeholder node, not
    traversed further.
  - Test-to-source mapping is import-based only: "this test file imports
    module X, so it probably tests X." No coverage-tool ground truth.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import jedi


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    qualified_name: str        # e.g. "mypkg.utils.helpers.parse_config"
    kind: str                  # "function" | "method" | "class"
    file_path: str             # relative to repo root
    lineno: int
    end_lineno: int
    docstring: str | None
    bases: list[str] = field(default_factory=list)   # for classes: base class names (unresolved)


@dataclass
class RepoParseResult:
    commit_hash: str
    repo_root: str
    files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    symbols: dict[str, Symbol] = field(default_factory=dict)         # qualified_name -> Symbol
    import_edges: list[tuple[str, str]] = field(default_factory=list)  # (file, target) target may be "external:X"
    call_edges: list[tuple[str, str]] = field(default_factory=list)    # (caller_qualname, callee_name) name-based
    inherit_edges: list[tuple[str, str]] = field(default_factory=list) # (class_qualname, base_name) name-based
    test_covers_edges: list[tuple[str, str]] = field(default_factory=list)  # (test_file, source_module)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_commit_hash(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git-unversioned"


def _is_test_file(rel_path: str) -> bool:
    name = Path(rel_path).name
    return name.startswith("test_") or name.endswith("_test.py")


def _module_name_for_file(repo_root: Path, file_path: Path) -> str:
    """Best-effort dotted module name from a file's path relative to repo root."""
    rel = file_path.relative_to(repo_root).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def _resolve_import_target(repo_root: Path, importing_file: Path, module: str, level: int) -> str:
    """
    Best-effort: if `module` (possibly relative, `level`>0) matches a file
    inside the repo, return its relative path; otherwise "external:<module>".

    Tries, in order: relative-import resolution (level>0), repo-root-relative
    resolution (proper package imports), and same-directory-as-importer
    resolution (common in flatly-organized scripts using sibling imports like
    `from dataset import X` run with that dir on the path).
    """
    candidates: list[Path] = []

    if level > 0:
        base = importing_file.parent
        for _ in range(level - 1):
            base = base.parent
        candidate_parts = module.split(".") if module else []
        candidates.append(base.joinpath(*candidate_parts))
    else:
        candidates.append(repo_root.joinpath(*module.split(".")))
        candidates.append(importing_file.parent.joinpath(*module.split(".")))

    for candidate in candidates:
        for suffix_candidate in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if suffix_candidate.exists():
                return str(suffix_candidate.relative_to(repo_root))
    return f"external:{module}"


# ---------------------------------------------------------------------------
# Per-file AST walk
# ---------------------------------------------------------------------------

class _FileVisitor(ast.NodeVisitor):
    def __init__(self, module_name: str, rel_path: str):
        self.module_name = module_name
        self.rel_path = rel_path
        self.symbols: dict[str, Symbol] = {}
        self.unresolved_calls: list[tuple[str, int, int]] = []
        self.inherit_edges: list[tuple[str, str]] = []
        self._scope_stack: list[str] = []

    def _qualname(self, name: str) -> str:
        prefix = ".".join([self.module_name] + self._scope_stack)
        return f"{prefix}.{name}" if prefix else name

    def visit_ClassDef(self, node: ast.ClassDef):
        qualname = self._qualname(node.name)
        bases = [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases]
        self.symbols[qualname] = Symbol(
            qualified_name=qualname, kind="class", file_path=self.rel_path,
            lineno=node.lineno, end_lineno=getattr(node, "end_lineno", node.lineno),
            docstring=ast.get_docstring(node), bases=bases,
        )
        for base in bases:
            if base:
                self.inherit_edges.append((qualname, base))

        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def _visit_function_like(self, node, kind: str):
        qualname = self._qualname(node.name)
        self.symbols[qualname] = Symbol(
            qualified_name=qualname, kind=kind, file_path=self.rel_path,
            lineno=node.lineno, end_lineno=getattr(node, "end_lineno", node.lineno),
            docstring=ast.get_docstring(node),
        )
        self._scope_stack.append(node.name)
        prev_caller = qualname
        # walk body looking for calls attributed to this function
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # record exact column offset for Jedi
                # end_col_offset points just after the identifier (e.g. before the parens)
                # subtracting 1 places us exactly on the last character of the identifier.
                col = getattr(child.func, "end_col_offset", None)
                lineno = getattr(child.func, "lineno", None)
                if col is not None and lineno is not None:
                    self.unresolved_calls.append((prev_caller, lineno, col - 1))
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        kind = "method" if self._scope_stack else "function"
        self._visit_function_like(node, kind)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        kind = "method" if self._scope_stack else "function"
        self._visit_function_like(node, kind)


# removed _call_target_name


# ---------------------------------------------------------------------------
# Repo-level parse
# ---------------------------------------------------------------------------

def parse_repo(repo_root: str) -> RepoParseResult:
    root = Path(repo_root).resolve()
    commit_hash = _get_commit_hash(root)
    result = RepoParseResult(commit_hash=commit_hash, repo_root=str(root))

    py_files = sorted(p for p in root.rglob("*.py") if ".git" not in p.parts and "__pycache__" not in p.parts)
    
    all_unresolved_calls: dict[str, list[tuple[str, int, int]]] = {}

    for file_path in py_files:
        rel_path = str(file_path.relative_to(root))
        result.files.append(rel_path)
        if _is_test_file(rel_path):
            result.test_files.append(rel_path)

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  [skip] {rel_path}: {e}")
            continue

        module_name = _module_name_for_file(root, file_path)
        visitor = _FileVisitor(module_name, rel_path)
        visitor.visit(tree)

        result.symbols.update(visitor.symbols)
        all_unresolved_calls[rel_path] = visitor.unresolved_calls
        result.inherit_edges.extend(visitor.inherit_edges)

        # imports (module-level only, both `import x` and `from x import y`)
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add((alias.name, 0))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                imported_modules.add((mod, node.level))

        for module, level in imported_modules:
            target = _resolve_import_target(root, file_path, module, level)
            result.import_edges.append((rel_path, target))
            if _is_test_file(rel_path) and not target.startswith("external:"):
                result.test_covers_edges.append((rel_path, target))
                
    # --- Jedi Pass to resolve call edges ---
    project = jedi.Project(path=str(root))
    
    for rel_path, calls in all_unresolved_calls.items():
        if not calls:
            continue
            
        file_path = root / rel_path
        try:
            script = jedi.Script(path=str(file_path), project=project)
        except Exception:
            # Jedi might fail to read the file in extremely rare cases
            continue
            
        for caller, lineno, col in calls:
            try:
                # infer at the exact end of the identifier name
                defs = script.infer(line=lineno, column=col)
            except Exception:
                continue
                
            for d in defs:
                if d.module_path and str(d.module_path).startswith(str(root)):
                    target_qualname = d.full_name
                    if target_qualname in result.symbols:
                        result.call_edges.append((caller, target_qualname))

    return result


if __name__ == "__main__":
    import sys
    import json

    target_repo = sys.argv[1] if len(sys.argv) > 1 else "."
    parsed = parse_repo(target_repo)
    print(f"commit: {parsed.commit_hash}")
    print(f"files: {len(parsed.files)} ({len(parsed.test_files)} test files)")
    print(f"symbols: {len(parsed.symbols)}")
    print(f"import edges: {len(parsed.import_edges)}")
    print(f"call edges: {len(parsed.call_edges)}")
    print(f"inherit edges: {len(parsed.inherit_edges)}")
    print(f"test-covers edges: {len(parsed.test_covers_edges)}")

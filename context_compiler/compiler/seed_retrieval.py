"""
Phase 2: Seed retrieval.

Cheap lexical match between a task description and node identifiers/
docstrings, to produce an initial candidate set (the "entry points" that
fixed-depth expansion then grows outward from).

This is deliberately dumb (token overlap, no embeddings) — that's the
point of a *baseline* compiler. Phase 5 replaces/augments this with a
learned scorer; this module gives you something to compare it against.

As of the semantic-seed upgrade, this module also provides:
  - semantic_seed_retrieval: embedding-based cosine similarity search
  - hybrid_seed_retrieval: reciprocal rank fusion of lexical + semantic
The hybrid retriever is the recommended default; it falls back to
lexical-only if sentence-transformers is not installed.
"""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Common English + code stopwords that shouldn't count as meaningful overlap.
_STOPWORDS = {
    "the", "a", "an", "in", "on", "for", "to", "of", "and", "or", "is", "are",
    "this", "that", "with", "as", "at", "by", "from", "it", "be", "should",
    "add", "fix", "update", "make", "def", "self", "return", "if", "else",
}


def _tokenize(text: str) -> list[str]:
    """Split identifiers on case/underscore boundaries too, e.g. `GCNBaseline` -> gcn, baseline."""
    words = _WORD_RE.findall(text)
    tokens: list[str] = []
    for w in words:
        # split snake_case
        parts = w.split("_")
        for p in parts:
            # split CamelCase within each part
            camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", p)
            tokens.extend(c.lower() for c in (camel_parts or [p]))
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


def lexical_seed_retrieval(
    graph: nx.MultiDiGraph, task_description: str, top_k: int = 10,
    node_types: tuple[str, ...] = ("function", "method", "class", "file"),
) -> list[tuple[str, float]]:
    """
    Score every eligible node by token overlap between the task description
    and (qualified name + docstring), return the top_k as (node_id, score).

    Score = weighted Jaccard-ish overlap: name-token matches count double
    docstring-token matches, since a name match is a stronger signal.
    """
    task_tokens = Counter(_tokenize(task_description))
    if not task_tokens:
        return []

    scored: list[tuple[str, float]] = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") not in node_types:
            continue

        name_tokens = Counter(_tokenize(node_id))
        doc_tokens = Counter(_tokenize(attrs.get("docstring", "") or ""))

        name_overlap = sum((task_tokens & name_tokens).values())
        doc_overlap = sum((task_tokens & doc_tokens).values())

        score = 2.0 * name_overlap + 1.0 * doc_overlap
        if score > 0:
            scored.append((node_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Semantic seed retrieval (sentence-transformers)
# ---------------------------------------------------------------------------

# Lazy-loaded embedder — loads once on first use, None if library absent.
_EMBEDDER = None
_EMBEDDER_LOADED = False


def _get_embedder():
    """Load the sentence-transformer model lazily; returns None if unavailable."""
    global _EMBEDDER, _EMBEDDER_LOADED
    if _EMBEDDER_LOADED:
        return _EMBEDDER
    _EMBEDDER_LOADED = True
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        _EMBEDDER = None
    return _EMBEDDER


def _node_text(node_id: str, attrs: dict) -> str:
    """Build a short text representation of a graph node for embedding."""
    kind = attrs.get("node_type", "symbol")
    # Convert qualified name to something more readable:
    # "context_compiler.parser.repo_parser.parse_repo" -> "parser repo_parser parse_repo"
    readable_name = node_id.replace("/", ".").rsplit(".", 1)[-1]
    doc = (attrs.get("docstring") or "").strip()
    if doc:
        # Truncate very long docstrings to keep embeddings focused
        doc = doc[:300]
        return f"{kind}: {readable_name}\n{doc}"
    return f"{kind}: {readable_name}"


def semantic_seed_retrieval(
    graph: nx.MultiDiGraph, task_description: str, top_k: int = 10,
    node_types: tuple[str, ...] = ("function", "method", "class", "file"),
) -> list[tuple[str, float]]:
    """
    Embed the task description and all eligible nodes using a pretrained
    sentence-transformer, rank by cosine similarity. Returns (node_id, score).

    Returns an empty list if sentence-transformers is not installed.
    """
    embedder = _get_embedder()
    if embedder is None:
        return []

    # Collect eligible nodes
    node_ids: list[str] = []
    node_texts: list[str] = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") not in node_types:
            continue
        node_ids.append(node_id)
        node_texts.append(_node_text(node_id, attrs))

    if not node_ids:
        return []

    # Encode everything in one batch for efficiency
    all_texts = [task_description] + node_texts
    embeddings = embedder.encode(all_texts, convert_to_tensor=False, show_progress_bar=False)

    query_emb = embeddings[0]
    node_embs = embeddings[1:]

    # Cosine similarity (normalize explicitly for robustness)
    import numpy as np
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    norms = np.linalg.norm(node_embs, axis=1, keepdims=True) + 1e-9
    node_normed = node_embs / norms
    similarities = node_normed @ query_norm

    # Rank and return top_k
    scored = list(zip(node_ids, similarities.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Hybrid retrieval (reciprocal rank fusion)
# ---------------------------------------------------------------------------

def hybrid_seed_retrieval(
    graph: nx.MultiDiGraph, task_description: str, top_k: int = 10,
    node_types: tuple[str, ...] = ("function", "method", "class", "file"),
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine lexical and semantic retrieval using Reciprocal Rank Fusion (RRF).

    For each ranker, a node at rank r gets score 1/(rrf_k + r). The RRF
    scores from both rankers are summed, then the top_k are returned.
    This is the standard fusion method used in production search systems
    (e.g. Elasticsearch, Vespa) — it's simple, parameter-free, and
    robust against score-scale differences between rankers.

    Falls back to lexical-only if sentence-transformers is not installed.

    Args:
        rrf_k: RRF constant (default 60, standard in literature). Higher
               values give more weight to lower-ranked results.
    """
    # Fetch more candidates from each ranker than we need, so fusion has
    # enough signal to re-rank properly.
    fetch_k = top_k * 3

    lexical_results = lexical_seed_retrieval(graph, task_description, top_k=fetch_k, node_types=node_types)
    semantic_results = semantic_seed_retrieval(graph, task_description, top_k=fetch_k, node_types=node_types)

    # If semantic is unavailable, fall back to lexical-only
    if not semantic_results:
        return lexical_results[:top_k]

    # RRF fusion
    rrf_scores: dict[str, float] = {}
    for rank, (node_id, _) in enumerate(lexical_results):
        rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (node_id, _) in enumerate(semantic_results):
        rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


if __name__ == "__main__":
    import sys

    from context_compiler.graph.code_graph import build_code_graph
    from context_compiler.parser.repo_parser import parse_repo

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else "graph transformer attention bias"

    g = build_code_graph(parse_repo(repo))

    print(f"Query: {query!r}\n")

    print("--- Lexical ---")
    for node_id, score in lexical_seed_retrieval(g, query, top_k=10):
        print(f"  {score:.3f}  {node_id}")

    print("\n--- Semantic ---")
    sem = semantic_seed_retrieval(g, query, top_k=10)
    if sem:
        for node_id, score in sem:
            print(f"  {score:.3f}  {node_id}")
    else:
        print("  (sentence-transformers not installed, skipping)")

    print("\n--- Hybrid (RRF) ---")
    for node_id, score in hybrid_seed_retrieval(g, query, top_k=10):
        print(f"  {score:.4f}  {node_id}")

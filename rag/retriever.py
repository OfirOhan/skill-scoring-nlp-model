import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from rag.embedder import embedder

CHROMA_PATH = "./chroma_db"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EXPAND_LLM = "qwen3"
FETCH_PER_QUERY = 10

reranker = CrossEncoder(RERANK_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)


# ---------------------------------------------------------------------------
# 1. Query Expansion (optional — LLM-backed, off by default for bulk builds)
# ---------------------------------------------------------------------------

def expand_query(original_query: str, n_variations: int = 3) -> list[str]:
    prompt = (
        f"You are a query-understanding module inside a retrieval pipeline.\n"
        f"Your generated queries will be used to search a document store using "
        f"both keyword matching (BM25) and semantic similarity (vector search).\n\n"
        f"Given the user's question, produce exactly {n_variations} search queries "
        f"that maximize the chance of retrieving the right chunks.\n\n"
        f"Think about:\n"
        f"- What words or phrases likely appear in the stored documents\n"
        f"- Include at least one short keyword-style query (for BM25)\n"
        f"- Include at least one natural-language query (for vector search)\n"
        f"- Cover different angles the answer might be described under\n\n"
        f"User question: \"{original_query}\"\n\n"
        f"Return ONLY the queries, one per line, no numbering, no explanation."
    )

    response = ollama.chat(
        model=EXPAND_LLM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip()
    variations = [line.strip() for line in raw.splitlines() if line.strip()]
    return [original_query] + variations[:n_variations]


# ---------------------------------------------------------------------------
# 2. Fusion primitives — BM25 + Vector search merged with RRF, then re-ranked
# ---------------------------------------------------------------------------

def bm25_search(query: str, chunks: list[str], top_k: int = 10) -> list[str]:
    """Build a one-off BM25 index over `chunks` and return the top matches.

    Convenience for ad-hoc single queries. Hot loops that issue many queries
    against the same corpus should build the index once and use `_bm25_top`.
    """
    if not chunks:
        return []
    bm25 = BM25Okapi([chunk.lower().split() for chunk in chunks])
    return _bm25_top(bm25, chunks, query, top_k)


def _bm25_top(bm25: BM25Okapi, chunks: list[str], query: str, top_k: int) -> list[str]:
    """Top-k chunks for `query` from a *pre-built* BM25 index."""
    scores = bm25.get_scores(query.lower().split())
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0]


def rrf_fusion(*ranked_lists: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion across any number of ranked lists.

    Each list is scored independently — rank 0 in any list gets the
    same 1/(k+1) score regardless of list length or origin.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    if not chunks:
        return []
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ---------------------------------------------------------------------------
# 3. Per-collection state (shared across many queries to the same candidate)
# ---------------------------------------------------------------------------

class _CollectionState:
    """Pre-computed per-candidate retrieval state.

    The expensive-to-repeat work — pulling every chunk out of ChromaDB,
    building the chunk -> doc_id provenance map, and tokenising the BM25 index —
    is done once here and reused for every skill query against that candidate.
    This is what makes per-(persona, skill) dataset construction fast: a persona
    with 12 skills pays the corpus setup once instead of 12 times.
    """

    def __init__(self, candidate_id: str):
        self.collection = client.get_or_create_collection(
            name=candidate_id,
            metadata={"hnsw:space": "cosine"},
        )
        stored = self.collection.get(include=["documents", "metadatas"])
        self.chunks = stored["documents"]
        self.text_to_doc = {
            doc: meta.get("doc_id")
            for doc, meta in zip(self.chunks, stored["metadatas"])
        }
        self.bm25 = (
            BM25Okapi([c.lower().split() for c in self.chunks])
            if self.chunks else None
        )


def _fuse_one(state: _CollectionState, query: str, per_query_vectors: list[list[str]],
              variations: list[str], top_k: int) -> dict:
    """Run BM25 + RRF + rerank for a single query given pre-fetched vector hits."""
    bm25_lists = []
    if state.bm25 is not None:
        bm25_lists = [_bm25_top(state.bm25, state.chunks, v, FETCH_PER_QUERY)
                      for v in variations]

    fused = rrf_fusion(*per_query_vectors, *bm25_lists)
    top_chunks = rerank(query, fused, top_k=top_k)
    return {
        "chunks": top_chunks,
        "doc_ids": [state.text_to_doc.get(c) for c in top_chunks],
        "fused_pool": fused,
    }


# ---------------------------------------------------------------------------
# 4. Training retrieval — fusion pipeline + chunk provenance
# ---------------------------------------------------------------------------

def retrieve_batch_for_training(
    queries: list[str],
    candidate_id: str,
    top_k: int = 3,
    expand: bool = False,
) -> list[dict]:
    """Retrieve for many skill queries against one candidate, efficiently.

    Forces the fusion path (no routing — skill queries are always specific) and
    returns chunk->doc_id provenance so retrieval can be graded against the
    evidence ground truth.

    Speed:
      • per-candidate corpus + BM25 index are built once (see _CollectionState)
      • with expand=False, all query embeddings are computed in a single batched
        encode + a single ChromaDB call, then fused per query.
      • expand=True adds an LLM call per query (much slower) but widens recall.

    Returns one dict per input query: {chunks, doc_ids, fused_pool}.
    """
    state = _CollectionState(candidate_id)
    if not queries:
        return []

    results: list[dict] = []

    if not expand:
        # One batched embed + one vector search for the whole skill set.
        q_embeddings = embedder.encode_queries(queries)
        vres = state.collection.query(
            query_embeddings=q_embeddings, n_results=FETCH_PER_QUERY
        )
        for i, query in enumerate(queries):
            per_query_vectors = [vres["documents"][i]]
            results.append(_fuse_one(state, query, per_query_vectors, [query], top_k))
        return results

    # Expansion path: each query fans out to LLM variations.
    for query in queries:
        variations = expand_query(query)
        q_embeddings = embedder.encode_queries(variations)
        vres = state.collection.query(
            query_embeddings=q_embeddings, n_results=FETCH_PER_QUERY
        )
        per_query_vectors = list(vres["documents"])
        results.append(_fuse_one(state, query, per_query_vectors, variations, top_k))
    return results


def retrieve_for_training(query: str, candidate_id: str,
                          top_k: int = 3, expand: bool = False) -> dict:
    """Single-query convenience wrapper over retrieve_batch_for_training."""
    return retrieve_batch_for_training([query], candidate_id, top_k=top_k, expand=expand)[0]


# ---------------------------------------------------------------------------
# 5. General retrieve — fusion pipeline for app/query use
# ---------------------------------------------------------------------------

def retrieve(query: str, candidate_id: str, top_k: int = 5, expand: bool = True) -> dict:
    """Run the fusion retrieval pipeline for a free-form query.

    Always BM25 + vector + RRF + rerank (the summary index and BROAD/SPECIFIC
    router were removed — every query goes through the same path).

    Returns a dict with:
        - chunks: list[str] — the retrieved text chunks
        - expanded_queries: list[str] — query variations actually used
        - fused_pool: list[str] — the candidate pool before re-ranking
    """
    state = _CollectionState(candidate_id)
    queries = expand_query(query) if expand else [query]

    q_embeddings = embedder.encode_queries(queries)
    vres = state.collection.query(query_embeddings=q_embeddings, n_results=FETCH_PER_QUERY)
    per_query_vectors = list(vres["documents"])

    out = _fuse_one(state, query, per_query_vectors, queries, top_k)
    return {
        "chunks": out["chunks"],
        "expanded_queries": queries,
        "fused_pool": out["fused_pool"],
    }

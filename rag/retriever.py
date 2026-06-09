import chromadb
import ollama
from rank_bm25 import BM25Okapi
from rag.embedder import embedder
from rag.reranker import reranker

CHROMA_PATH = "./chroma_db"
ROUTER_LLM = "qwen3"

client = chromadb.PersistentClient(path=CHROMA_PATH)


# ---------------------------------------------------------------------------
# 1. Query Expansion
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
        model=ROUTER_LLM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip()
    variations = [line.strip() for line in raw.splitlines() if line.strip()]
    return [original_query] + variations[:n_variations]


# ---------------------------------------------------------------------------
# 2. Fusion Retrieval — BM25 + Vector search merged with RRF
# ---------------------------------------------------------------------------

def bm25_search(query: str, chunks: list[str], top_k: int = 10) -> list[str]:
    if not chunks:
        return []
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
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


# ---------------------------------------------------------------------------
# 3. Re-ranking
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list[str], top_k: int = 8) -> list[str]:
    return reranker.rerank(query, chunks, top_k=top_k)


# ---------------------------------------------------------------------------
# 4. Main retrieve pipeline (fusion only — summary index / router removed)
# ---------------------------------------------------------------------------

def retrieve(query: str, candidate_id: str, top_k: int = 8) -> dict:
    """
    Run the fusion retrieval pipeline for a query.

    Always BM25 + vector + RRF + rerank (the summary index and BROAD/SPECIFIC
    router were removed — every query goes through the same path).

    Returns a dict with:
        - chunks: list[str] — the retrieved text chunks
        - expanded_queries: list[str] — query variations actually used
        - fused_pool: list[str] — the candidate pool before re-ranking
    """
    collection = client.get_or_create_collection(
        name=candidate_id,
        metadata={"hnsw:space": "cosine"},
    )

    # --- Step 1: Query Expansion ---
    queries = expand_query(query)

    # --- Step 2: Vector search (per-query ranked lists) ---
    # Each query's results are kept as a separate ranked list so RRF
    # scores them independently — rank 0 in any list gets equal weight.
    fetch_per_query = 10
    per_query_results: list[list[str]] = []

    q_embeddings = embedder.encode_queries(queries)
    results = collection.query(query_embeddings=q_embeddings, n_results=fetch_per_query)

    for chunk_list in results["documents"]:
        per_query_results.append(chunk_list)

    # --- Step 3: BM25 search (full collection) ---
    all_chunks = collection.get(include=["documents"])["documents"]
    bm25_lists = [bm25_search(q, all_chunks, top_k=fetch_per_query) for q in queries]

    # --- Step 4: Fuse all ranked lists with RRF ---
    fused = rrf_fusion(*per_query_results, *bm25_lists)

    # --- Step 5: Re-rank and return the best ---
    top_chunks = rerank(query, fused, top_k=top_k)

    return {
        "chunks": top_chunks,
        "expanded_queries": queries,
        # The fused candidate pool *before* re-ranking. Exposed so evaluation can
        # tell whether a relevant chunk was lost at recall (never entered the pool)
        # vs. at re-ranking (entered the pool but was cut from the top-k).
        "fused_pool": fused,
    }


# ---------------------------------------------------------------------------
# 5. Training retrieval — fusion + chunk provenance (used by build_dataset.py)
# ---------------------------------------------------------------------------

FETCH_PER_QUERY = 10


def _bm25_top(bm25: BM25Okapi, chunks: list[str], query: str, top_k: int) -> list[str]:
    """Top-k chunks for `query` from a *pre-built* BM25 index (avoids rebuilding)."""
    scores = bm25.get_scores(query.lower().split())
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0]


def retrieve_batch_for_training(
    queries: list[str],
    candidate_id: str,
    top_k: int = 3,
    expand: bool = False,
) -> list[dict]:
    """Retrieve for many skill queries against one candidate, efficiently.

    Forces the fusion path (skill queries are always specific) and returns
    chunk -> doc_id provenance so retrieval can be graded against the evidence
    ground truth. The per-candidate corpus + BM25 index are built once and reused
    across all queries; with expand=False all query embeddings are computed in one
    batched call. Returns one dict per query: {chunks, doc_ids, fused_pool}.
    """
    collection = client.get_or_create_collection(
        name=candidate_id, metadata={"hnsw:space": "cosine"},
    )
    stored = collection.get(include=["documents", "metadatas"])
    all_chunks = stored["documents"]
    text_to_doc = {
        doc: (meta or {}).get("doc_id")
        for doc, meta in zip(all_chunks, stored["metadatas"])
    }
    bm25 = BM25Okapi([c.lower().split() for c in all_chunks]) if all_chunks else None

    def _fuse(query, per_query_vectors, variations):
        bm25_lists = ([_bm25_top(bm25, all_chunks, v, FETCH_PER_QUERY) for v in variations]
                      if bm25 is not None else [])
        fused = rrf_fusion(*per_query_vectors, *bm25_lists)
        top = rerank(query, fused, top_k=top_k)
        return {"chunks": top, "doc_ids": [text_to_doc.get(c) for c in top], "fused_pool": fused}

    results: list[dict] = []
    if not queries:
        return results

    if not expand:
        # One batched embed + one vector search for the whole skill set.
        q_embeddings = embedder.encode_queries(queries)
        vres = collection.query(query_embeddings=q_embeddings, n_results=FETCH_PER_QUERY)
        for i, query in enumerate(queries):
            results.append(_fuse(query, [vres["documents"][i]], [query]))
        return results

    for query in queries:
        variations = expand_query(query)
        q_embeddings = embedder.encode_queries(variations)
        vres = collection.query(query_embeddings=q_embeddings, n_results=FETCH_PER_QUERY)
        results.append(_fuse(query, list(vres["documents"]), variations))
    return results


def retrieve_for_training(query: str, candidate_id: str,
                          top_k: int = 3, expand: bool = False) -> dict:
    """Single-query convenience wrapper over retrieve_batch_for_training."""
    return retrieve_batch_for_training([query], candidate_id, top_k=top_k, expand=expand)[0]

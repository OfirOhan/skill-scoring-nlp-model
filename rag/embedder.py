from sentence_transformers import SentenceTransformer

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class NomicEmbedder:
    """
    Wrapper around nomic-embed-text-v1.5 that applies the required task prefixes.

    Nomic was trained asymmetrically — queries and documents live in different
    subspaces. Without these prefixes the embeddings are misaligned and similarity
    scores become unreliable, which is why nomic appeared to perform worse than
    MiniLM in your earlier eval.

    Always use:
        encode_documents() during ingestion
        encode_query() / encode_queries() during retrieval
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

    def encode_documents(self, texts: list[str]) -> list:
        """Encode text chunks for storage. Applies 'search_document:' prefix."""
        prefixed = [f"search_document: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()

    def encode_query(self, text: str) -> list:
        """Encode a single query. Applies 'search_query:' prefix."""
        return self.model.encode(f"search_query: {text}", normalize_embeddings=True).tolist()

    def encode_queries(self, texts: list[str]) -> list:
        """Encode multiple query variations. Applies 'search_query:' prefix to each."""
        prefixed = [f"search_query: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()


# Single shared instance — import this everywhere
embedder = NomicEmbedder()
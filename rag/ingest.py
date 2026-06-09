import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
from rag.embedder import embedder
import os
import re

CHROMA_PATH = "./chroma_db"

client = chromadb.PersistentClient(path=CHROMA_PATH)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def get_collection(candidate_id: str):
    return client.get_or_create_collection(
        name=candidate_id,
        metadata={"hnsw:space": "cosine"},
    )


# -- Section extraction ------------------------------------------------------

_SPACED_HEADER_RE = re.compile(
    r"^[A-Z](\s+[A-Z]){3,}(\s+[A-Z])*\s*$"
)


def _is_spaced_header(text: str) -> bool:
    """Detect spaced-letter headers like 'T E C H N I C A L  S K I L L S'."""
    return bool(_SPACED_HEADER_RE.match(text.strip()))


def _is_data_title(text: str) -> bool:
    """Detect Title elements that are actually content (contain numbers, colons, etc.).

    Examples that should be treated as content, not section names:
        'GPA: 94.3 🏆 Dean's Honor List (1st & 2nd Year)'
        '✉ ofir@gmail.com ☎ +972-54-2863632'
    """
    stripped = text.strip()
    if re.search(r"\d", stripped):
        return True
    if "@" in stripped or "☎" in stripped or "✉" in stripped:
        return True
    return False


def extract_sections(file_path: str) -> list[dict]:
    """Partition the document into sections using Unstructured.

    Returns a list of {text, section} dicts where consecutive elements
    under the same section are merged into a single text block.
    """
    elements = partition(file_path)

    sections = []
    current_section = "general"
    current_texts = []

    def _flush():
        if current_texts:
            sections.append({
                "text": "\n\n".join(current_texts),
                "section": current_section,
            })
            current_texts.clear()

    for element in elements:
        text = element.text.strip() if element.text else ""
        if not text:
            continue

        if _is_spaced_header(text):
            _flush()
            current_section = text
            continue

        if element.category == "Title":
            if _is_data_title(text):
                current_texts.append(text)
            else:
                _flush()
                current_section = text
        else:
            current_texts.append(text)

    _flush()

    return sections


# -- In-memory ingestion (synthetic training data) ---------------------------

def ingest_text(text: str, candidate_id: str, doc_id: str, doc_type: str = "cv"):
    """Ingest a raw text document already in memory (no file, no summary).

    Used to build training data from the synthetic corpus: the document text
    comes straight from documents_db.json, so there is no file to partition and
    no recruiter-summary to generate. Every chunk stores its `doc_id` in the
    metadata so retrieval results can be traced back to their source document —
    this is what makes retrieval evaluation against the evidence ground truth
    possible.

    Chunks are stored verbatim (no 'Section:' prefix) so the text the scorer
    later sees is exactly the document text.
    """
    if not text or not text.strip():
        return 0

    collection = get_collection(candidate_id)
    chunks = text_splitter.split_text(text)
    if not chunks:
        return 0

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metas = [
        {
            "candidate_id": candidate_id,
            "doc_id": doc_id,
            "doc_type": doc_type,
        }
        for _ in chunks
    ]

    # encode_documents applies the 'search_document:' prefix — aligned with
    # encode_query() at retrieval time.
    embeddings = embedder.encode_documents(chunks)
    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metas)
    return len(chunks)


# -- Ingestion pipeline -------------------------------------------------------

def ingest_document(file_path: str, candidate_id: str, doc_type: str = "cv"):
    """File pipeline: file -> sections -> chunks -> embeddings -> ChromaDB.

    Uses encode_documents() so all stored vectors have the 'search_document:'
    prefix baked in — aligned with encode_query() used at retrieval time.
    """
    sections = extract_sections(file_path)
    if not sections:
        print(f"Warning: No content extracted from {file_path}")
        return

    collection = get_collection(candidate_id)
    base = os.path.basename(file_path)

    # --- Chunk index ---
    all_chunks, all_ids, all_metas = [], [], []

    for s_idx, section in enumerate(sections):
        chunks = text_splitter.split_text(section["text"])

        for c_idx, chunk in enumerate(chunks):
            contextualized_chunk = f"Section: {section['section']}\n{chunk}"
            all_chunks.append(contextualized_chunk)
            all_ids.append(f"{base}_s{s_idx}_chunk_{c_idx}")
            all_metas.append({
                "candidate_id": candidate_id,
                "doc_type": doc_type,
                "source_file": base,
                "section": section["section"],
            })

    # encode_documents applies 'search_document:' prefix to every chunk
    all_embeddings = embedder.encode_documents(all_chunks)

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metas,
    )
    print(f"Ingested {len(all_chunks)} chunks from {base} ({len(sections)} sections)")
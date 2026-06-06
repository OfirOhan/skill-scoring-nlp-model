import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
import os

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
SUMMARY_LLM = "qwen3"

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def get_collection(candidate_id: str):
    return client.get_or_create_collection(name=candidate_id)


def get_summary_collection(candidate_id: str):
    return client.get_or_create_collection(name=f"{candidate_id}_summaries")


# -- Section extraction ------------------------------------------------------

import re

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
    # Contains digits → likely data, not a section header
    if re.search(r"\d", stripped):
        return True
    # Contains email-like or phone-like patterns
    if "@" in stripped or "☎" in stripped or "✉" in stripped:
        return True
    return False


def extract_sections(file_path: str) -> list[dict]:
    """Partition the document into sections using Unstructured.

    Returns a list of {text, section} dicts where consecutive elements
    under the same section are merged into a single text block. This
    ensures projects/experiences stay together as one chunk when possible.

    Also handles:
    - Spaced-letter headers (e.g., 'T E C H N I C A L  S K I L L S')
      that Unstructured misses as Title elements
    - Title elements containing data (e.g., 'GPA: 94.3') that should
      be kept as content, not used as section names
    """
    elements = partition(file_path)

    sections = []
    current_section = "general"
    current_texts = []

    def _flush():
        """Save accumulated texts as one merged section."""
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

        # Check for spaced-letter headers in ANY element type
        if _is_spaced_header(text):
            _flush()
            current_section = text
            continue

        if element.category == "Title":
            if _is_data_title(text):
                # This is data disguised as a Title — keep it as content
                current_texts.append(text)
            else:
                # Real section header — flush previous and start new section
                _flush()
                current_section = text
        else:
            current_texts.append(text)

    _flush()  # don't forget the last section

    return sections


# -- Summary generation ------------------------------------------------------

def generate_summary(full_text: str, doc_type: str) -> str:
    """Ask the LLM to summarize the document in 5-6 sentences.

    Used to populate the summary index — searched when queries are broad
    like 'tell me about this candidate' rather than specific skill lookups.
    """
    prompt = (
        f"You are summarizing a {doc_type} document for a recruiter.\n"
        f"Write a concise 5-6 sentence summary that must cover:\n"
        f"1. Candidate's full name and current/most recent role\n"
        f"2. Education: degree(s), institution(s), and graduation year(s)\n"
        f"3. Total years of professional experience\n"
        f"4. Key technical skills and domain expertise\n"
        f"5. Most notable achievement or project\n"
        f"Be factual, no opinions.\n\n"
        f"Document:\n{full_text[:3000]}"  # cap at 3000 chars to avoid huge prompts
    )

    response = ollama.chat(
        model=SUMMARY_LLM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# -- Ingestion pipeline -------------------------------------------------------

def ingest_document(file_path: str, candidate_id: str, doc_type: str = "cv"):
    """Full pipeline: file -> sections -> chunks -> embeddings -> ChromaDB.
    Also generates and stores a document summary in a separate summary index.

    Each chunk is stored with metadata:
        - candidate_id: who this document belongs to
        - doc_type: cv | readme | certificate | recommendation
        - source_file: original filename
        - section: which section of the document this chunk came from

    Summary index stores one summary per document for broad queries.
    """
    sections = extract_sections(file_path)
    if not sections:
        print(f"Warning: No content extracted from {file_path}")
        return

    collection = get_collection(candidate_id)
    base = os.path.basename(file_path)

    # --- Chunk index ---
    all_chunks, all_embeddings, all_ids, all_metas = [], [], [], []

    for s_idx, section in enumerate(sections):
        # Split the body text
        chunks = text_splitter.split_text(section["text"])

        for c_idx, chunk in enumerate(chunks):
            # Prepend the section title to the text chunk for semantic richness
            contextualized_chunk = f"Section: {section['section']}\n{chunk}"

            all_chunks.append(contextualized_chunk)
            all_ids.append(f"{base}_s{s_idx}_chunk_{c_idx}")
            all_metas.append({
                "candidate_id": candidate_id,
                "doc_type": doc_type,
                "source_file": base,
                "section": section["section"],
            })

    all_embeddings = embedder.encode(all_chunks).tolist()
    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metas,
    )
    print(f"Ingested {len(all_chunks)} chunks from {base} ({len(sections)} sections)")

    # --- Summary index ---
    full_text = " ".join(s["text"] for s in sections)
    summary = generate_summary(full_text, doc_type)

    summary_collection = get_summary_collection(candidate_id)
    summary_embedding = embedder.encode([summary]).tolist()
    summary_collection.add(
        documents=[summary],
        embeddings=summary_embedding,
        ids=[base],
        metadatas=[{
            "candidate_id": candidate_id,
            "doc_type": doc_type,
            "source_file": base,
        }],
    )
    print(f"Summary stored for {base}: '{summary[:80]}...'")
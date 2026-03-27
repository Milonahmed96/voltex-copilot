import os
import re
import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
CHROMA_DIR         = Path("chroma_db")
COLLECTION_NAME    = "voltex_knowledge"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"

# Policy document chunking settings
POLICY_CHUNK_SIZE    = 400   # characters per chunk
POLICY_CHUNK_OVERLAP = 80    # characters of overlap between consecutive chunks

# Document type routing
FAQ_DOCUMENTS = {
    "product_faqs.txt"
}

POLICY_DOCUMENTS = {
    "voltcare_policy.txt",
    "repairs_returns_policy.txt",
    "delivery_orders_policy.txt",
    "voltmobile_policy.txt",
}

# Category tags — used as metadata for pre-filtering
DOCUMENT_CATEGORIES = {
    "voltcare_policy.txt"       : "voltcare",
    "repairs_returns_policy.txt": "returns_repairs",
    "product_faqs.txt"          : "products",
    "delivery_orders_policy.txt": "delivery_orders",
    "voltmobile_policy.txt"     : "voltmobile",
}

# ─────────────────────────────────────────────
# CHUNKING FUNCTIONS
# ─────────────────────────────────────────────

def chunk_policy_document(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Sliding window chunker for policy documents.

    Splits on sentence boundaries where possible to avoid cutting mid-sentence.
    Falls back to character boundary if no sentence boundary is found within
    the window. Applies overlap so context is not lost at chunk edges.
    """
    # Clean up excessive whitespace while preserving paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Split into sentences (naively on ". ", "? ", "! ", and newlines)
    # This is intentionally simple — the overlap handles boundary imprecision
    sentence_endings = re.compile(r'(?<=[.!?])\s+|\n')
    sentences = sentence_endings.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence keeps us under the chunk size, add it
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            # Save the current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
            # Start new chunk — include overlap from the end of the previous chunk
            if current_chunk and overlap > 0:
                overlap_text = current_chunk[-overlap:].strip()
                # Find the nearest sentence start in the overlap to avoid
                # starting mid-sentence
                space_pos = overlap_text.find(' ')
                if space_pos > 0:
                    overlap_text = overlap_text[space_pos:].strip()
                current_chunk = (overlap_text + " " + sentence).strip()
            else:
                current_chunk = sentence

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Filter out chunks that are too short to be meaningful (headers, dividers)
    chunks = [c for c in chunks if len(c) > 40]

    return chunks


def chunk_faq_document(text: str) -> list[str]:
    """
    Q&A boundary chunker for the product FAQ document.

    Detects 'Q:' as the start of a new chunk boundary. Each Q&A pair
    is kept as a single atomic retrieval unit. This preserves the complete
    answer alongside the question — critical for retrieval accuracy.

    Also captures section headers as metadata context by prepending them
    to the first Q&A in each section.
    """
    chunks = []
    current_section = ""
    current_chunk = ""

    lines = text.split('\n')

    for line in lines:
        stripped = line.strip()

        # Detect section headers (e.g. "SECTION A — LAPTOPS AND COMPUTING")
        if re.match(r'^(SECTION [A-Z]|[A-Z][A-Z\s]+—)', stripped) and len(stripped) > 10:
            current_section = stripped
            continue

        # Detect start of a new Q&A block
        if stripped.startswith('Q:'):
            # Save the previous chunk if it exists
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Start new chunk — prepend section context
            if current_section:
                current_chunk = f"[{current_section}]\n{stripped}"
            else:
                current_chunk = stripped
        elif stripped.startswith('A:') or (current_chunk and stripped):
            # Continue building the current chunk
            current_chunk = current_chunk + "\n" + stripped
        elif not stripped and current_chunk:
            # Blank line — may be end of answer; don't split yet
            # (some answers have blank lines between paragraphs)
            current_chunk = current_chunk + "\n"

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter very short chunks (section headers with no Q&A)
    chunks = [c for c in chunks if len(c) > 60]

    return chunks


# ─────────────────────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────────────────────

def extract_section_header(chunk: str) -> str:
    """
    Extracts the nearest section or part header from a policy chunk.
    Used to enrich metadata so retrieval can be filtered by section.
    Returns empty string if no header found.
    """
    # Look for PART N or SECTION headers at the start of the chunk
    match = re.search(r'(PART \d+\s*[-—]\s*[^\n]+|SECTION [A-Z]\s*[-—]\s*[^\n]+)', chunk)
    if match:
        return match.group(1).strip()

    # Look for all-caps headers that are likely section titles
    match = re.search(r'^([A-Z][A-Z\s]{10,})\n', chunk, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return ""


# ─────────────────────────────────────────────
# MAIN INGESTION PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Voltex Knowledge Base Ingestion Pipeline")
    print("=" * 60)

    # Verify knowledge base directory exists
    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"ERROR: knowledge_base/ directory not found.")
        print(f"Expected location: {KNOWLEDGE_BASE_DIR.resolve()}")
        return

    # Load the embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    # Initialise ChromaDB with persistent storage
    print(f"\nInitialising ChromaDB at: {CHROMA_DIR.resolve()}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Clear existing collection if it exists (clean rebuild)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Existing collection '{COLLECTION_NAME}' found — deleting for clean rebuild.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # cosine similarity — standard for text
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

    # Process each document
    total_chunks = 0
    all_documents   = []
    all_embeddings  = []
    all_metadatas   = []
    all_ids         = []

    for filename in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        doc_name = filename.name

        if doc_name not in POLICY_DOCUMENTS and doc_name not in FAQ_DOCUMENTS:
            print(f"\nSkipping unrecognised file: {doc_name}")
            continue

        print(f"\nProcessing: {doc_name}")

        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"  File size: {len(text):,} characters")

        # Apply the correct chunking strategy
        if doc_name in FAQ_DOCUMENTS:
            chunks = chunk_faq_document(text)
            strategy = "Q&A boundary"
        else:
            chunks = chunk_policy_document(text, POLICY_CHUNK_SIZE, POLICY_CHUNK_OVERLAP)
            strategy = f"sliding window ({POLICY_CHUNK_SIZE} chars, {POLICY_CHUNK_OVERLAP} overlap)"

        print(f"  Chunking strategy: {strategy}")
        print(f"  Chunks produced: {len(chunks)}")

        # Log a sample chunk for visual inspection
        if chunks:
            sample = chunks[len(chunks) // 2]  # middle chunk
            print(f"  Sample chunk (middle):\n    {sample[:200].replace(chr(10), ' ')}...")

        category = DOCUMENT_CATEGORIES.get(doc_name, "general")

        # Build embeddings and metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_name.replace('.txt', '')}_{i:04d}"

            embedding = model.encode(chunk).tolist()

            section = extract_section_header(chunk)

            metadata = {
                "source"    : doc_name,
                "category"  : category,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "section"   : section,
                "char_count": len(chunk),
                "strategy"  : "faq" if doc_name in FAQ_DOCUMENTS else "policy",
            }

            all_documents.append(chunk)
            all_embeddings.append(embedding)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

        total_chunks += len(chunks)

    # Batch insert into ChromaDB
    # ChromaDB has a practical batch limit — split into batches of 500
    print(f"\nInserting {total_chunks} chunks into ChromaDB...")
    batch_size = 500

    for batch_start in range(0, len(all_ids), batch_size):
        batch_end = batch_start + batch_size
        collection.add(
            documents  = all_documents[batch_start:batch_end],
            embeddings = all_embeddings[batch_start:batch_end],
            metadatas  = all_metadatas[batch_start:batch_end],
            ids        = all_ids[batch_start:batch_end],
        )
        print(f"  Inserted batch {batch_start // batch_size + 1} "
              f"({min(batch_end, len(all_ids)) - batch_start} chunks)")

    print(f"\nIngestion complete.")
    print(f"Total chunks in collection: {collection.count()}")

    # ─────────────────────────────────────────────
    # RETRIEVAL SMOKE TESTS
    # ─────────────────────────────────────────────
    # Run 5 test queries and print the top result for each.
    # This lets you visually inspect retrieval quality before building the copilot.

    print("\n" + "=" * 60)
    print("RETRIEVAL SMOKE TESTS")
    print("=" * 60)

    test_queries = [
        ("voltcare",        "Does VoltCare cover accidental damage on my laptop?"),
        ("returns_repairs", "I want to return my washing machine that arrived damaged"),
        ("products",        "What is the difference between OLED and QLED televisions?"),
        ("delivery_orders", "Can I change my delivery address after I have placed an order?"),
        ("voltmobile",      "I am going to Spain next week, will my phone work?"),
    ]

    for category_filter, query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print(f"Category filter: {category_filter}")

        query_embedding = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"category": category_filter},
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = round(1 - dist, 3)   # cosine distance → similarity
            preview    = doc[:180].replace('\n', ' ')
            print(f"  Rank {rank} | similarity: {similarity} | "
                  f"source: {meta['source']} | section: {meta.get('section', 'n/a')[:50]}")
            print(f"    \"{preview}...\"")

    # Save ingestion summary to a JSON file for reference
    summary = {
        "total_chunks": total_chunks,
        "documents_processed": list(POLICY_DOCUMENTS | FAQ_DOCUMENTS),
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "policy_chunk_size": POLICY_CHUNK_SIZE,
        "policy_chunk_overlap": POLICY_CHUNK_OVERLAP,
    }
    summary_path = Path("evaluation") / "ingestion_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nIngestion summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
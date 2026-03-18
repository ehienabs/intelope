"""
RAG pipeline — build a searchable index from cleaned datasets and retrieve
relevant context at query time.

Uses sentence-transformers for embeddings and FAISS for fast vector search.
"""

from pathlib import Path
import json
import numpy as np

INDEX_DIR = Path("data/rag_index")
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, good quality
CHUNK_LEN = 512  # max characters per chunk for embedding


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def _chunk_text(text: str, max_len: int = CHUNK_LEN) -> list[str]:
    """Split long text into overlapping chunks for better retrieval."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    stride = max_len // 2
    for i in range(0, len(text), stride):
        chunk = text[i : i + max_len]
        if len(chunk.split()) < 8:
            break
        chunks.append(chunk)
    return chunks


def build_index(dataset_names: list[str], datasets_dir: Path | None = None) -> dict:
    """
    Build a FAISS index from one or more cleaned datasets.

    Reads each dataset's clean.jsonl, chunks the text, embeds everything,
    and saves the index + metadata to data/rag_index/.

    Returns stats dict.
    """
    import faiss

    if datasets_dir is None:
        datasets_dir = Path("data/datasets")

    embedder = _get_embedder()

    all_chunks = []   # list of text strings
    all_meta = []     # parallel list of {source, dataset, id}

    for ds_name in dataset_names:
        clean_path = datasets_dir / ds_name / "clean.jsonl"
        if not clean_path.exists():
            continue
        with clean_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("output", "").strip()
                if not text:
                    continue
                source = rec.get("source", "unknown")
                rec_id = rec.get("id", "")
                for chunk in _chunk_text(text):
                    all_chunks.append(chunk)
                    all_meta.append({
                        "source": source,
                        "dataset": ds_name,
                        "id": rec_id,
                    })

    if not all_chunks:
        return {"error": "No data to index.", "chunks": 0}

    # Embed in batches
    embeddings = embedder.encode(
        all_chunks, show_progress_bar=True, batch_size=64,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (inner product since embeddings are normalized)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

    with (INDEX_DIR / "chunks.json").open("w") as f:
        json.dump(all_chunks, f)
    with (INDEX_DIR / "meta.json").open("w") as f:
        json.dump(all_meta, f)
    with (INDEX_DIR / "config.json").open("w") as f:
        json.dump({
            "model": EMBED_MODEL,
            "datasets": dataset_names,
            "total_chunks": len(all_chunks),
            "embedding_dim": dim,
        }, f)

    return {
        "chunks": len(all_chunks),
        "datasets": dataset_names,
        "embedding_dim": dim,
    }


class Retriever:
    """Load a built index and search it."""

    def __init__(self, index_dir: Path | None = None):
        import faiss

        index_dir = index_dir or INDEX_DIR
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with (index_dir / "chunks.json").open() as f:
            self.chunks = json.load(f)
        with (index_dir / "meta.json").open() as f:
            self.meta = json.load(f)
        self.embedder = _get_embedder()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return the top_k most relevant chunks for a query.

        Each result: {text, source, dataset, score}
        """
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.meta[idx].get("source", ""),
                "dataset": self.meta[idx].get("dataset", ""),
                "score": float(score),
            })
        return results


def index_exists(index_dir: Path | None = None) -> bool:
    """Check if a RAG index has been built."""
    d = index_dir or INDEX_DIR
    return (d / "index.faiss").exists()


def get_index_info(index_dir: Path | None = None) -> dict | None:
    """Return config info about the current index, or None."""
    d = index_dir or INDEX_DIR
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        return None
    with cfg_path.open() as f:
        return json.load(f)


def format_context(results: list[dict], max_chars: int = 2000) -> str:
    """Format retrieval results into a context block for the LLM prompt."""
    parts = []
    used = 0
    for r in results:
        text = r["text"]
        if used + len(text) > max_chars:
            remaining = max_chars - used
            if remaining > 100:
                text = text[:remaining]
            else:
                break
        parts.append(f"[Source: {r['source']}]\n{text}")
        used += len(text)
    return "\n\n---\n\n".join(parts)

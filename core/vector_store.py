"""
Approach 1, retrieval half: a dense vector store.

Chunks are turned into 384-dimensional embeddings with a sentence-transformer
and stored in a FAISS index. Vectors are L2-normalised and the index uses inner
product, so the score FAISS returns *is* the cosine similarity - which makes it
directly comparable with the TF-IDF numbers from the traditional approach.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from config import EMBEDDING_MODEL
from core.ingestion import Chunk

_model = None


def get_embedder():
    """Load the sentence-transformer once and reuse it (it is ~90 MB)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """Encode texts into unit-length float32 vectors."""
    model = get_embedder()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        batch_size=32,
    )
    return vectors.astype("float32")


class VectorStore:
    """A FAISS index plus the chunks it was built from."""

    def __init__(self, index, chunks: list[dict]):
        self.index = index
        self.chunks = chunks

    # -- construction -------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorStore":
        texts = [chunk.text for chunk in chunks]
        vectors = embed(texts)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        return cls(index, [chunk.to_dict() for chunk in chunks])

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "vectors.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        index_path = directory / "vectors.faiss"
        chunks_path = directory / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No vector index found in {directory}. Re-upload the document."
            )

        index = faiss.read_index(str(index_path))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        return cls(index, chunks)

    # -- search -------------------------------------------------------------
    def search(self, question: str, top_k: int = 4) -> list[dict]:
        """Return the top_k most semantically similar chunks."""
        if not self.chunks:
            return []

        query_vector = embed([question])
        top_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, position in zip(scores[0], indices[0]):
            if position < 0:
                continue
            chunk = dict(self.chunks[int(position)])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results

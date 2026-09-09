"""
Document lifecycle: save the upload, run the pipeline, build BOTH indexes.

Building the dense index and the TF-IDF model at the same moment matters for
the study - it guarantees the two approaches are always compared over an
identical set of chunks.
"""

import shutil
import time
import uuid
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, INDEX_DIR, UPLOAD_DIR
from core import database as db
from core.ingestion import process_file
from core.tfidf_store import TfidfStore
from core.vector_store import VectorStore


def index_dir_for(user_id: int, doc_id: int) -> Path:
    return INDEX_DIR / f"user_{user_id}" / f"doc_{doc_id}"


def save_upload(user_id: int, filename: str, data: bytes) -> Path:
    """Write the uploaded bytes to disk under a collision-proof name."""
    folder = UPLOAD_DIR / f"user_{user_id}"
    folder.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    path = folder / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    path.write_bytes(data)
    return path


def ingest(user_id: int, filename: str, data: bytes, title: str = "",
           chunk_size: int = CHUNK_SIZE,
           chunk_overlap: int = CHUNK_OVERLAP,
           progress=None) -> dict:
    """
    Full ingestion pipeline for one file.

    `progress` is an optional callable(fraction, label) so the Streamlit upload
    screen can show what stage the pipeline is at.
    """
    def report(fraction: float, label: str) -> None:
        if progress:
            progress(fraction, label)

    timings: dict[str, float] = {}

    report(0.05, "Saving the file...")
    path = save_upload(user_id, filename, data)

    report(0.15, "Extracting and cleaning text...")
    started = time.perf_counter()
    try:
        result = process_file(path, chunk_size, chunk_overlap)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    timings["extract_clean_chunk"] = time.perf_counter() - started

    chunks = result["chunks"]

    report(0.35, f"Created {len(chunks)} chunks. Registering the document...")
    doc_id = db.add_document(
        user_id=user_id,
        title=(title.strip() or Path(filename).stem),
        filename=Path(filename).name,
        filepath=str(path),
        filetype=Path(filename).suffix.lower().lstrip("."),
        num_pages=result["pages"],
        num_chars=result["clean_chars"],
        num_chunks=len(chunks),
        index_dir="",
    )

    directory = index_dir_for(user_id, doc_id)

    try:
        report(0.45, "Approach 1: embedding chunks into the vector store...")
        started = time.perf_counter()
        VectorStore.build(chunks).save(directory)
        timings["vector_index"] = time.perf_counter() - started

        report(0.85, "Approach 2: fitting the TF-IDF model...")
        started = time.perf_counter()
        TfidfStore.build(chunks).save(directory)
        timings["tfidf_index"] = time.perf_counter() - started
    except Exception:
        # Do not leave a half-built document behind.
        db.delete_document(doc_id, user_id)
        shutil.rmtree(directory, ignore_errors=True)
        path.unlink(missing_ok=True)
        raise

    db.execute(
        "UPDATE documents SET index_dir = ? WHERE id = ?", (str(directory), doc_id)
    )

    report(1.0, "Done.")

    return {
        "document_id": doc_id,
        "pages": result["pages"],
        "raw_chars": result["raw_chars"],
        "clean_chars": result["clean_chars"],
        "num_chunks": len(chunks),
        "chunks": chunks,
        "timings": timings,
        "index_dir": str(directory),
    }


def remove(user_id: int, doc_id: int) -> None:
    """Delete a document together with its file and both indexes."""
    document = db.get_document(doc_id, user_id)
    if not document:
        return

    if document["filepath"]:
        Path(document["filepath"]).unlink(missing_ok=True)

    shutil.rmtree(index_dir_for(user_id, doc_id), ignore_errors=True)
    db.delete_document(doc_id, user_id)


# ---------------------------------------------------------------------------
# Loading (cached by the UI layer so a big index is read from disk only once)
# ---------------------------------------------------------------------------
def load_vector_store(user_id: int, doc_id: int) -> VectorStore:
    return VectorStore.load(index_dir_for(user_id, doc_id))


def load_tfidf_store(user_id: int, doc_id: int) -> TfidfStore:
    return TfidfStore.load(index_dir_for(user_id, doc_id))

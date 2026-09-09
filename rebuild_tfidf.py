"""
Re-fit the TF-IDF model of every document already in data/indexes.

Only Approach 2 is touched: the chunks, the uploaded files and the FAISS index
are left exactly as they are, so the comparison still runs over identical
chunks. Run this after changing any TFIDF_* setting in config.py, or after
pulling a change to core/tfidf_store.py, instead of re-uploading the material.

    python rebuild_tfidf.py
"""

import json
import sys
import time

from config import INDEX_DIR
from core.ingestion import Chunk
from core.tfidf_store import TfidfStore


def main() -> int:
    directories = sorted(INDEX_DIR.glob("user_*/doc_*"))
    if not directories:
        print(f"No indexes found under {INDEX_DIR}.")
        return 0

    rebuilt = failed = 0

    for directory in directories:
        label = f"{directory.parent.name}/{directory.name}"
        chunks_path = directory / "tfidf_chunks.json"

        if not chunks_path.exists():
            print(f"  {label:<24} skipped - no chunks on disk")
            continue

        try:
            chunks = [Chunk(**record) for record
                      in json.loads(chunks_path.read_text(encoding="utf-8"))]

            before = ""
            try:
                old = TfidfStore.load(directory)
                before = f"{len(old.vectorizer.get_feature_names_out()):,} terms -> "
            except Exception:
                pass

            started = time.perf_counter()
            store = TfidfStore.build(chunks)
            store.save(directory)
            elapsed = time.perf_counter() - started

            print(f"  {label:<24} {len(chunks):>4} chunks, "
                  f"{len(store.passages):>5} passages, "
                  f"{before}{len(store.vectorizer.get_feature_names_out()):,} terms "
                  f"({elapsed:.2f}s)")
            rebuilt += 1
        except Exception as exc:                      # keep going for the rest
            print(f"  {label:<24} FAILED - {exc}")
            failed += 1

    print(f"\nRebuilt {rebuilt} index(es); {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

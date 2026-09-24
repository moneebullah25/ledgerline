"""Model handles for the evidence store. Loaded once per process (lru_cache)."""

import threading
from functools import lru_cache

import numpy as np

from ledgerline.config import settings

# bge models are trained with this instruction on the query side only; passages go in bare.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# One model call at a time per process: on Apple silicon sentence-transformers runs on the
# Metal (MPS) device, which aborts the process when two threads encode concurrently.
# ponytail: global lock; per-device model copies if parallel inference ever matters.
INFERENCE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    return SentenceTransformer(settings().embed_model)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings().rerank_model)


def embed_passages(texts: list[str]) -> np.ndarray:
    with INFERENCE_LOCK:
        return _embedder().encode(texts, batch_size=32, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    with INFERENCE_LOCK:
        return _embedder().encode(QUERY_PREFIX + query, normalize_embeddings=True)


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    with INFERENCE_LOCK:
        return _reranker().predict([(query, t) for t in texts], batch_size=32).tolist()

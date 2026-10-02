"""Qdrant storage: dense (MiniLM) + sparse (BM25) vectors in one collection."""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from . import config as cfg

logger = logging.getLogger("rag.store")


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if cfg.QDRANT_URL:
        return QdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY, timeout=60)
    if cfg.QDRANT_PATH:
        return QdrantClient(path=cfg.QDRANT_PATH)  # embedded mode, no Docker needed
    return QdrantClient(url="http://localhost:6333", timeout=60)


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.EMBED_MODEL)
    model.max_seq_length = cfg.EMBED_MAX_SEQ_LEN  # MiniLM default is 256 -> long chunks get cut
    return model


@lru_cache(maxsize=1)
def get_sparse():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(cfg.SPARSE_MODEL)


def _sparse(e) -> models.SparseVector:
    return models.SparseVector(indices=[int(i) for i in e.indices], values=[float(v) for v in e.values])


def _source_filter(source: str | None) -> models.Filter | None:
    if not source:
        return None
    return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])


def ensure_collection(recreate: bool = False) -> None:
    client, name = get_client(), cfg.COLLECTION
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        return
    dim = get_embedder().get_sentence_embedding_dimension()
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    if cfg.QDRANT_URL or not cfg.QDRANT_PATH:  # payload indexes only exist on a Qdrant server
        client.create_payload_index(name, "source", models.PayloadSchemaType.KEYWORD)
    logger.info("Created collection '%s' (dim=%d).", name, dim)


def delete_document(source: str) -> None:
    get_client().delete(cfg.COLLECTION, points_selector=models.FilterSelector(filter=_source_filter(source)))


def upsert_document(result: dict[str, Any], batch_size: int = 64) -> int:
    """Embed + store all chunks of a process_pdf() result. Re-ingesting replaces the old copy."""
    chunks, source = result["chunks"], result["document_id"]
    if not chunks:
        return 0
    client, embedder, sparse = get_client(), get_embedder(), get_sparse()
    delete_document(source)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        # texts = [c["text"] for c in batch]
        # dense = embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        # sparse_vecs = list(sparse.embed(texts))
        texts = [c["text"] for c in batch]
        dense = embedder.encode([cfg.EMBED_DOC_PREFIX + t for t in texts],
                                batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        sparse_vecs = list(sparse.embed(texts))   # BM25 stays unprefixed


        points = []
        for c, d, s in zip(batch, dense, sparse_vecs):
            md = c["metadata"]
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{c['chunk_id']}")),
                vector={"dense": [float(x) for x in d], "sparse": _sparse(s)},
                payload={"chunk_id": c["chunk_id"], "text": c["text"], "source": source,
                         "page_start": md["page_start"], "page_end": md["page_end"],
                         "section": md["section"], "subsection": md["subsection"]},
            ))
        client.upsert(cfg.COLLECTION, points=points, wait=True)
        logger.info("Stored %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks))
    return len(chunks)


def count_points() -> int:
    return get_client().count(cfg.COLLECTION, exact=True).count


def search(query: str, limit: int = 10, mode: str = "hybrid",
           prefetch: int = 30, source: str | None = None) -> list[dict[str, Any]]:
    """mode: 'dense' | 'sparse' | 'hybrid' (dense + BM25 fused with RRF)."""
    if mode not in ("dense", "sparse", "hybrid"):
        raise ValueError("mode must be dense, sparse or hybrid")
    client, flt = get_client(), _source_filter(source)
    # dense_q = [float(x) for x in get_embedder().encode(query, normalize_embeddings=True)]
    # sparse_q = _sparse(next(iter(get_sparse().query_embed(query))))
    dense_q = [float(x) for x in get_embedder().encode(
    cfg.EMBED_QUERY_PREFIX + query, normalize_embeddings=True)]
    sparse_q = _sparse(next(iter(get_sparse().query_embed(query))))   # unchanged, no prefix    


    if mode == "dense":
        res = client.query_points(cfg.COLLECTION, query=dense_q, using="dense",
                                  query_filter=flt, limit=limit, with_payload=True)
    elif mode == "sparse":
        res = client.query_points(cfg.COLLECTION, query=sparse_q, using="sparse",
                                  query_filter=flt, limit=limit, with_payload=True)
    else:
        res = client.query_points(
            cfg.COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_q, using="dense", filter=flt, limit=prefetch),
                models.Prefetch(query=sparse_q, using="sparse", filter=flt, limit=prefetch),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True,
        )
    return [{**(p.payload or {}), "score": float(p.score)} for p in res.points]

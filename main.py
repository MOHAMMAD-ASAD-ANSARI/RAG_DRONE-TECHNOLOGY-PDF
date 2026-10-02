"""RAG CLI:  python main.py ingest file.pdf | ask ["question"] | eval dataset.json | stats"""
from __future__ import annotations

import argparse
import logging
import sys


def cmd_ingest(a) -> int:
    from rag import store
    from rag.ingest_pdf import process_pdf

    result = process_pdf(a.pdf, a.max_chunk_size, a.overlap)
    if result.get("needs_ocr"):
        print("This PDF is scanned/image-only. Run OCR first (e.g. ocrmypdf in.pdf out.pdf).")
        return 1
    if not result["chunks"]:
        print("No chunks were produced from this PDF.")
        return 1
    store.ensure_collection(recreate=a.recreate)
    n = store.upsert_document(result)
    print(f"Ingested {n} chunks from {result['document_id']}. Collection total: {store.count_points()}")
    return 0


def cmd_ask(a) -> int:
    from rag.generate import stream_answer
    from rag.retrieve import search

    def run(question: str) -> None:
        hits = search(question, top_k=a.top_k, candidates=a.candidates, mode=a.mode,
                      use_rerank=not a.no_rerank, source=a.source)
        if not hits:
            print("No results found. Did you run `python main.py ingest <pdf>` first?")
            return
        print()
        for piece in stream_answer(question, hits):
            print(piece, end="", flush=True)
        print("\n\nSources:")
        for h in hits:
            pages = h["page_start"] if h["page_start"] == h["page_end"] else f"{h['page_start']}-{h['page_end']}"
            print(f"  - {h['source']} p.{pages} | {h['section']}" + (f" > {h['subsection']}" if h["subsection"] else ""))

    if a.question:
        run(a.question)
        return 0
    print("Ask questions (empty line or 'exit' to quit).")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("", "exit", "quit"):
            return 0
        run(q)


def cmd_eval(a) -> int:
    from rag.evaluate import print_results, run_eval

    configs = ([("dense", False), ("sparse", False), ("hybrid", False), ("hybrid", True)]
               if a.compare else [(a.mode, a.rerank)])
    for mode, rr in configs:
        res = run_eval(a.dataset, mode=mode, use_rerank=rr, source=a.source)
        print_results(f"{mode}{' + rerank' if rr else ''}", res)
    return 0


def cmd_stats(_a) -> int:
    from rag import config, store
    print(f"Collection '{config.COLLECTION}': {store.count_points()} chunks")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="PDF RAG on Qdrant")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="parse a PDF and store it in Qdrant")
    s.add_argument("pdf")
    s.add_argument("--max-chunk-size", type=int, default=None)
    s.add_argument("--overlap", type=int, default=None)
    s.add_argument("--recreate", action="store_true", help="wipe the collection first")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ask", help="ask a question (interactive if omitted)")
    s.add_argument("question", nargs="?")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--candidates", type=int, default=30)
    s.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    s.add_argument("--no-rerank", action="store_true")
    s.add_argument("--source", help="only search this PDF filename")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="retrieval metrics (no LLM calls)")
    s.add_argument("dataset")
    s.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    s.add_argument("--rerank", action="store_true")
    s.add_argument("--compare", action="store_true", help="dense vs sparse vs hybrid vs hybrid+rerank")
    s.add_argument("--source")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("stats", help="number of stored chunks")
    s.set_defaults(fn=cmd_stats)

    args = p.parse_args()
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

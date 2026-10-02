# PDF RAG on Qdrant

PDF → clean → section-aware chunks → **Qdrant** (dense + BM25 vectors) → hybrid search (RRF) → cross-encoder rerank → **Gemini** answer.

## 1. Setup (once)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```
Edit `.env`: put your free Gemini key (https://aistudio.google.com/apikey) in `GEMINI_API_KEY`.

## 2. Start Qdrant (pick one)
| Option | How |
|---|---|
| **A. Docker** (default) | `docker compose up -d` → dashboard at http://localhost:6333/dashboard |
| **B. Qdrant Cloud free tier** | Create a free cluster at cloud.qdrant.io, set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` |
| **C. No Docker** | In `.env` set `QDRANT_URL=` (empty) and `QDRANT_PATH=./qdrant_data` (one process at a time) |

## 3. Run
```bash
python main.py ingest drone_technology.pdf            # parse, embed, store (re-running replaces that PDF's chunks)
python main.py ask "What are drones used for?"        # one question
python main.py ask                                    # interactive chat loop
python main.py stats                                  # chunk count
python main.py eval evaluation_dataset.json --compare # dense vs sparse vs hybrid vs hybrid+rerank (no LLM calls)
```
Useful flags: `ask --top-k 5 --mode hybrid|dense|sparse --no-rerank --source file.pdf`,
`ingest --max-chunk-size 900 --overlap 120 --recreate`.

First run downloads three small models (MiniLM embedder, BM25 tokenizer, cross-encoder). Later runs are cached.

## 4. Verify the install (offline, no downloads/API)
`python tests/smoke_test.py` → should end with `ALL TESTS PASSED`.

## Keeping Gemini free-tier usage low
Only the top 5 chunks (~1.5k tokens) are sent per question, and only `ask` calls Gemini. Ingest and eval run locally. Rate limits get retried automatically. If `gemini-2.5-flash` is unavailable for your key, set `GEN_MODEL` in `.env`.

## Evaluation file format
`[{"question": "...", "ground_truth_chunks": [3, 4]}]` — chunk ids are **0-based ints** (same as your notebook). `"chunk_001"`-style labels are also accepted (1-based).
Chunk ids restart at 0 per PDF, so evaluate one PDF at a time (`--source file.pdf` if you stored several).

## Troubleshooting
- `Connection refused` → Qdrant isn't running (`docker compose up -d`) or the URL in `.env` is wrong.
- `needs OCR` → scanned PDF; run `ocrmypdf in.pdf out.pdf` first.
- Changed embedding model? Re-ingest with `--recreate` (vector size changes).
- Embedded mode lock error → close the other process using `./qdrant_data`.

## Bugs fixed from the notebook
1. BM25 evaluation was scoring the **dense** results; the reranker eval used a stale global `question`, and only reranked the top-k it was about to cut (reranking could never change the result). Now it reranks a 30-candidate pool.
2. The final reranked prompt used the un-reranked `context`, and `context_reranked` picked from the un-reranked list.
3. `process_pdf` overwrote a passed-in config; heading regex treated "Part of the…" and numbered sentences as headings; header/footer stripping now only applies inside the header/footer zone.
4. Chunk `page_start/page_end` covered the whole section; now tracked per paragraph. Overlap no longer cuts mid-word; over-long sentences are hard-split.
5. nDCG used the retrieved list as its ideal ranking; now uses the standard ideal. Eval retrieves once for all K (was 6× the work).
6. MiniLM silently truncated at 256 tokens; max length is now 384. In-memory list + O(n) Python cosine replaced by Qdrant; BM25 now lives in Qdrant too (no rebuild on start).

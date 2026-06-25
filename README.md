# WikiGraph — a self-growing Wikipedia GraphRAG chatbot on InterSystems IRIS

WikiGraph is a chatbot that answers from a knowledge base **that grows by itself**. When it
doesn't know enough about a topic, it fetches **only the matching Wikipedia page** (linked pages
are never followed), embeds it **inside IRIS** with the `EMBEDDING` datatype, extracts entities
and relationships with a local LLM, and stores everything — documents, text, vectors, the graph
and chat state — **exclusively in IRIS**.

The whole pipeline is an Interoperability production written in **Python with
[pyprod](https://github.com/intersystems/pyprod)**, so every step (search, scrape, ingest, LLM
call) is a traceable message in the IRIS Management Portal. GraphRAG schema inspired by
[iris-vector-rag](https://github.com/intersystems-community/iris-vector-rag).

---

## How it works

1. **Ask** — a question arrives (English or Portuguese) through the chat UI.
2. **Search** — IRIS resolves the question to known entities and does a **graph-first**, entity-scoped
   vector search over the chunks it already has.
3. **Learn** — if the best score is below a threshold, it fetches the single best-matching Wikipedia
   page, chunks it, embeds each chunk, and mines entities + relationships into the graph.
4. **Answer** — a local LLM writes a grounded reply from the retrieved chunks and graph facts, in the
   question's language.

The second time the same topic comes up, step 3 is skipped — it's already remembered.

---

## Architecture

Two containers on one Compose network:

```
 Browser — chat UI (EN/PT toggle · ☰ menu: Documentation / Knowledge graph / Manage knowledge)
    │  WSGI  (/wikigraph, Flask hosted inside IRIS)
    │  POST /api/chat  →  just INSERTs a 'pending' GraphKB.ChatRequest row, returns its id
    ▼
 ┌─ IRIS interoperability production (pyprod) ────────────────────────────────────────────┐
 │  chatService (BS)        polls pending rows every 1s, claims them                      │
 │     └─► chatProcess (BP) orchestrator: drives status searching→learning→answering→done │
 │            ├─► knowledgeOperation (BO)   graph-first, entity-scoped vector search      │
 │            ├─► wikipediaOperation (BO)   best-match single page (no link following)    │
 │            ├─► ingestProcess (BP)        chunk → EMBEDDING column → entity extraction  │
 │            └─► llmOperation (BO)         topic distillation · extraction · answers     │
 └────────────────────────────────────────────────────────────────────────────────────────┘
    │  EMBEDDING(text,'wikiembed')  and  chat completions
    ▼
 ollama container — chat model gemma4:e2b  +  embedding model (OpenAI-compatible /v1/embeddings)
```

The frontend polls `GET /api/chat/<id>` and renders a live status bubble; every hop above is also a
message in **Management Portal → Interoperability → Productions**.

### The production, split for one-command compile

`pyprod` only compiles the classes physically present in the file it is handed (imports are not
followed). The production is therefore split into focused modules under `src-iris/production/`, and a
single loader compiles them in order:

| File | Contents |
|---|---|
| `common.py` | shared helpers + `EMBEDDING_CONFIG` — plain Python, **not** compiled by pyprod |
| `messages.py` | the interoperability messages (`JsonSerialize`) |
| `services.py` | `chatPollAdapter` (inbound) + `chatService` |
| `processes.py` | `chatProcess` + `ingestProcess` |
| `operations.py` | `knowledgeOperation` + `wikipediaOperation` + `llmOperation` |
| `wikiGraph.py` | the `Production` definition; imports the rest |
| `loadProduction.sh` | compiles each module in order (Production last) — **the one command to load it all** |

### Storage (all IRIS, namespace `GRAPHRAG`)

| Table | Purpose |
|---|---|
| `GraphKB.Document` | one row per ingested page; unique on `(pageId, lang)` |
| `GraphKB.Chunk` | text chunks; `chunkVector EMBEDDING('wikiembed','embedText')` auto-vectorized, HNSW index. `embedText` is the chunk **prefixed with the article title** so senses separate in vector space; `chunkText` stays clean for display |
| `GraphKB.Entity` | people/places/concepts; `descVector` EMBEDDING + HNSW; deduped on `(entityName, entityType, lang)`; `documentRef` links it to its article |
| `GraphKB.Relationship` | typed edges `(sourceEntityRef, predicate, targetEntityRef)` |
| `GraphKB.EntityMention` | which chunk an entity was mentioned in |
| `GraphKB.ChatRequest` | the chat/queue state machine the frontend polls (`pending → searching → learning → answering → done`/`error`); an `ingest` status queues add-by-title requests |

### Embeddings

Embeddings use the IRIS **`EMBEDDING` datatype** (config `wikiembed`) backed by **`%Embedding.OpenAI`
pointed at Ollama's OpenAI-compatible `/v1/embeddings`** endpoint (1024 dims). Each `EMBEDDING()` is a
quick HTTP call to Ollama — IRIS never loads an embedding model in-process, so there is no per-call
model-reload cost and no interop dead-job problem. `%Embedding.OpenAI` is redirected from
`api.openai.com` to the `ollama` host via the config's `httpConfig` key (`Https=0`), with a dummy SSL
config to satisfy validation since the call is plain HTTP. Similarity is `VECTOR_COSINE` over HNSW
indexes. There is **no AI Hub `%ConfigStore` dependency** — endpoints/models are plain env vars and
live production settings.

---

## Retrieval: graph-first and disambiguating

- **Graph-first, entity-scoped** (`knowledgeOperation`): the query is matched against entity
  description vectors — **and** against any entity whose full name literally appears in the question
  (`nameMatch`), so a multi-entity question ("Armstrong, Aldrin and Collins") keeps every named entity
  even when one embeds weakly. Only entities within a cosine **margin** (`scopeMargin`, 0.12) of the top
  match contribute their documents, and the chunk vector search is **scoped to those documents**, so a
  dominant sense ("Tesla" the company) cleanly beats the rival sense (Tesla the person). A **per-entity
  gather** then pulls each matched entity's best chunks via `EntityMention` (so every named entity
  contributes evidence); results are merged and capped (`maxChunks`), a 1-hop relationship expansion is
  added, and `{chunks, entities, relationships}` is returned as the LLM context. If the scoped result is
  weak (`scopeFloor`, 0.45) it widens to all documents.
- **Title resolution & disambiguation when learning** (`wikipediaOperation`): the LLM first distils the
  question into a clean search title, which is resolved **following redirects** — e.g. "Bruce Wayne" →
  *Batman*, "Anakin Skywalker" → *Darth Vader* — and the redirected article is ingested directly. Only
  if the title is **missing** or resolves to a **disambiguation page** does it fall back to a candidate
  search + intro-cosine rerank (with disambiguation pages excluded) — so "Neil Armstrong" the astronaut
  wins over "RV Neil Armstrong" the research ship. Only that single page is read; content links are
  never followed.

Retrieval, chunking and Wikipedia behaviour are all **live settings** in the Management Portal
(`chatProcess`, `knowledgeOperation`, `ingestProcess`, `wikipediaOperation`) — see the tuning tables in
`Overview2.html`.

---

## Prerequisites

- Docker + Docker Compose
- The **IRIS AI Hub community image** (build `2026.2.0AI.162.0`) available to Docker (set `IRIS_IMAGE`
  in `.env`). Any IRIS ≥ 2025.1 image with interoperability and the `EMBEDDING` datatype works too.
- A GPU is optional — `docker-compose.yaml` reserves an NVIDIA GPU for Ollama; remove that `deploy`
  block to run CPU-only.

## Setup

```bash
# review IRIS_IMAGE / ports / models in .env first
docker compose up -d --build
```

The image build bakes in the schema + EMBEDDING config, **compiles the production with
`loadProduction.sh`, and marks it to auto-start**, so a plain `docker compose up` comes up ready.
Ollama pulls the chat model and the embedding model on first start.

Open the chatbot: **http://localhost:52773/wikigraph/** — pick EN or PT and ask. The whole UI (and the
live status bubbles) switches language with the toggle. The **☰ menu** opens a side panel for:
- **Documentation** — this project overview, served at `/wikigraph/docs/overview.html`;
- **Knowledge graph** — an interactive D3 "constellation" of the stored entities and relationships
  (color by article or entity type, multi-select highlighting, edge predicate labels, refresh),
  served at `/wikigraph/docs/graph.html`;
- **Manage knowledge** — **add an article by Wikipedia title**, browse learned articles, and delete
  one or all (cascading to their chunks, entities, mentions and relationships), with live counts.

While it ingests a new topic the UI shows a 📚 *learning* bubble.

### Recompiling after editing the Python

```bash
# compiles every module (messages, services, processes, operations, then the Production)
docker compose exec iris bash -lc ". /opt/.venv/bin/activate && bash /opt/irisbuild/production/loadProduction.sh"
docker compose exec iris iris session IRIS -U GRAPHRAG \
  'do ##class(Ens.Director).UpdateProduction(10)'
```

> Always compile with `loadProduction.sh`, not a single `intersystems_pyprod wikiGraph.py` — pyprod
> only compiles the classes in the file it is given, so each module must be compiled separately.

---

## HTTP API (under `/wikigraph`)

| Method & path | Purpose |
|---|---|
| `GET /` | the chat single-page app |
| `POST /api/chat` `{question, lang}` | insert a `pending` ChatRequest; returns `{requestId, status}` |
| `GET /api/chat/<id>` | poll `{status, statusDetail, answer, …}` |
| `GET /api/stats` | knowledge-graph counts (`documents/chunks/entities/relationships`) |
| `GET /api/graph` | the whole graph (`nodes`, `links`, `documents`) for the constellation view |
| `GET /api/documents` | learned articles with per-article chunk/entity counts |
| `POST /api/documents` `{title, lang}` | add an article by Wikipedia title (queues an `ingest` request; poll `GET /api/chat/<id>` for progress) |
| `DELETE /api/documents/<id>` | delete an article and everything derived from it |
| `GET /docs/<file>` | static project docs (`overview.html`, `graph.html`, …) |

---

## Configuration (`.env`)

| Variable | Example | Meaning |
|---|---|---|
| `IRIS_IMAGE` | AI Hub community `2026.2.0AI.162.0` | base image for the IRIS container |
| `IRIS_PORT` | `52773` | host port for the IRIS web server / chat UI |
| `IRIS_SPORT` | `51972` | host port for the IRIS superserver (1972) |
| `OLLAMA_MODEL` | `gemma4:e2b` | Ollama chat model (topic distillation, entity extraction, answers) |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama endpoint as seen from IRIS |
| `OLLAMA_PORT` | `11435` | host port for Ollama (container is always 11434 internally) |
| `EMBEDDING_MODEL` | `leoipulsar/harrier-0.6b` | Ollama embedding model behind the `EMBEDDING` datatype — **must be 1024-dim** (multilingual; `bge-m3` or `mxbai-embed-large` also work) |
| `DEFAULT_LANG` | `en` | fallback chat language (UI offers EN/PT per message) |
| `SIMILARITY_THRESHOLD` | `0.55` | best cosine score below which the bot learns from Wikipedia |

The `EMBEDDING` config defaults to `bge-m3` if `EMBEDDING_MODEL` is unset; the shipped `.env` points it
at `leoipulsar/harrier-0.6b`. Ollama chat tuning (`maxTokens`, `temperature`, `topP`, `topK`, `numCtx`,
`keepAlive`, …) lives as live settings on `llmOperation` in the Management Portal.

---

## Notes

- All persistence is IRIS — no external vector store, queue or cache.
- The Wikipedia scraper deliberately fetches a single page per topic and never follows *content* links
  (it does follow title redirects, e.g. Bruce Wayne → Batman).
- gemma is a reasoning model: `llmOperation` sends `think:false` so the token budget isn't spent on a
  hidden `thinking` block (which otherwise returns empty answers) — and so extraction yields clean JSON.
- The frontend never calls the production directly; it only inserts a `pending` row that the polling
  `chatService` claims — avoiding the `<Ens>ErrJobRegistryNotClean` web-worker job-registry race.
- camelCase is used across the Python/ObjectScript code (pyprod framework names like `OnRequest`,
  `MessageMap`, `ADAPTER` excepted).

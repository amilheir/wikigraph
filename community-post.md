# WikiGraph: a self-growing Wikipedia GraphRAG chatbot where IRIS *is* the entire stack

Most RAG demos load a fixed set of documents and stop there. I wanted something more dynamic — and to demonstrate a point I make to customers regularly: **InterSystems IRIS can serve as the entire platform**. Documents, text chunks, **vectors**, the **entity/relationship graph**, chat state, the web application, *and* the orchestration all reside within a **single IRIS instance**. There are no additional services to deploy alongside it.

**WikiGraph** is a chatbot whose knowledge **grows on its own**. Ask it something outside its current knowledge and it reads the matching Wikipedia page on demand, converts it into vectors and a knowledge graph, and answers from what it has just learned. Ask again later and the response is immediate.

> 📸 *(screenshot: the chat answering "Who was the first person on the moon?" with a 📚 "learning…" bubble)*

## Architecture: two containers, one of which does almost everything

```
 Browser — chat UI (EN/PT)
    │  WSGI (/wikigraph, Flask hosted inside IRIS)
    │  POST /api/chat → just INSERTs a 'pending' GraphKB.ChatRequest row
    ▼
 ┌─ IRIS interoperability production (pyprod, Python) ─────────────────────┐
 │  chatService (BS)  polls pending / ingest / resolve rows, claims them   │
 │   ├─ chatProcess (BP)  searching → learning → answering → done          │
 │   │    ├─ knowledgeOperation (BO)  graph-first, entity-scoped search    │
 │   │    ├─ wikipediaOperation (BO)  one page, redirects resolved         │
 │   │    ├─ ingestProcess (BP)       chunk → EMBEDDING column → LINK-KG   │
 │   │    └─ llmOperation (BO)        distill · extract · answer           │
 │   └─ resolutionOperation (BO)      dedup / merge duplicate entities     │
 └─────────────────────────────────────────────────────────────────────────┘
    ▼
 ollama container — chat model + embedding model
```

## The whole pipeline is an Interoperability production — written in Python

This is the core of the project. Every step — searching the graph, scraping a page, ingesting, calling the LLM — is an **interoperability message** between business hosts. As a result, you gain the entire Interoperability toolset (queues, retries, settings, and the **Visual Trace**) at no additional cost, while the components themselves are written in **Python** with [pyprod](https://github.com/intersystems/pyprod).

A business operation is just a Python class with a `MessageMap`:

```python
class llmOperation(BusinessOperation):
    ollamaModel = IRISProperty(default="gemma4:e4b", settings="Ollama")
    temperature = IRISProperty(default="0.3", settings="Ollama Options")

    MessageMap = {"WikiGraph.generateAnswerMsg": "onGenerateAnswer"}

    def onGenerateAnswer(self, request):
        content = self.chat(ANSWER_PROMPT.format(lang=request.lang,
                                                 context=request.contextJson),
                            request.question)
        return Status.OK(), llmResponseMsg(content)
```

…and the production itself wires the hosts together, also in Python:

```python
class wikiGraphProduction(Production):
    services   = [ServiceItem("chatService", "WikiGraph.chatService", ...)]
    processes  = [ProcessItem("chatProcess", "WikiGraph.chatProcess", ...),
                  ProcessItem("ingestProcess", "WikiGraph.ingestProcess", ...)]
    operations = [OperationItem("knowledgeOperation", "WikiGraph.knowledgeOperation"),
                  OperationItem("wikipediaOperation",  "WikiGraph.wikipediaOperation"),
                  OperationItem("llmOperation",        "WikiGraph.llmOperation")]
```

Several aspects of this approach are worth highlighting:

- **Full observability at no additional cost.** Open **Management Portal → Interoperability → Productions** and the entire request appears as a message trace: `chatService → chatProcess → knowledgeOperation → wikipediaOperation → ingestProcess → llmOperation`. When the bot "learns," you can observe it happen, step by step.
- **Every parameter is a live setting.** Those `IRISProperty` lines surface as editable settings in the Portal — retrieval thresholds, chunk sizes, LLM `temperature`/`num_predict` — all tunable without recompilation.
- **Pull, don't push — a key reliability pattern.** The WSGI application never invokes the production directly; it only `INSERT`s a `pending` row. A **polling business service** within the production claims it and drives the pipeline. (Calling Ens from web workers registers each worker as an IRIS job and introduces race conditions; the poller avoids this entirely.)
- **Split for one-command compilation.** pyprod compiles the classes physically within the file it is given, so the production is divided into focused modules (`messages`, `services`, `processes`, `operations`, `wikiGraph`), and a single loader script compiles them in order. One command loads the entire system.

> 📸 *(screenshot: the Visual Trace of one "learning" request in the Management Portal)*

## Embeddings: the IRIS `EMBEDDING` datatype, served by a local Ollama

The second thing I want every customer to see is how little code vectorization requires in IRIS. You **declare** the embedding on the column, and IRIS computes and stores the vector automatically on `%Save()` — there is no embedding code in the application whatsoever:

```sql
CREATE TABLE GraphKB.Chunk (
    documentRef INTEGER, chunkIndex INTEGER,
    chunkText  VARCHAR(30000),
    embedText  VARCHAR(31000),
    chunkVector EMBEDDING('wikiembed','embedText')          -- auto-vectorized on save
)
CREATE INDEX HNSWChunkIdx ON TABLE GraphKB.Chunk (chunkVector) AS HNSW(Distance='Cosine')
```

Retrieval is then plain SQL with `VECTOR_COSINE`:

```sql
SELECT TOP 5 c.chunkText, d.title,
       VECTOR_COSINE(c.chunkVector, EMBEDDING(?, 'wikiembed')) AS score
FROM GraphKB.Chunk c JOIN GraphKB.Document d ON c.documentRef = d.%ID
ORDER BY score DESC
```

The notable part is the `wikiembed` config. The `EMBEDDING` datatype is **pluggable**: it delegates to an embedding provider class. WikiGraph uses **`%Embedding.OpenAI`**, which speaks the open **OpenAI-compatible `/v1/embeddings`** protocol — and that protocol is precisely what a **local Ollama** container exposes. The configuration's `httpConfig` therefore simply points the provider at the Ollama host:

```objectscript
Set config = {"modelName":"harrier-0.6b", "apiKey":"ollama", "sslConfig":"ollama"}
Set config.httpConfig = {"Server":"ollama", "Https":0}   // plain HTTP to the sidecar
Set config.httpConfig.Port = 11434

Do ##class(%SQL.Statement).%ExecDirect(,
  "INSERT INTO %Embedding.Config (Name, Configuration, EmbeddingClass, VectorLength, Description) "_
  "VALUES (?, ?, ?, ?, ?)",
  "wikiembed", config.%ToJSON(), "%Embedding.OpenAI", 1024, "Embeddings via local Ollama")
```

The advantages of this design are significant:

- **No model ever loads inside IRIS.** Each `EMBEDDING()` is a lightweight HTTP call to Ollama, so there is no per-call model-reload cost and no long-running in-process job to disrupt the interoperability engine.
- **Swap models with a single config row.** The embedding model (here, a 1024-dimension multilingual model running in Ollama) is simply `modelName` — change it without touching schema or code, provided the dimension matches the HNSW index.
- **A minor validation detail worth noting:** `%Embedding.OpenAI` validates that `sslConfig` names an existing SSL configuration even when communicating over plain HTTP (`Https=0`). Create a placeholder configuration (`Security.SSLConfigs.Create("ollama")`) and you are set.

The same pattern powers the **graph layer** as well — `GraphKB.Entity.descVector` is another `EMBEDDING` column — so the graph and the vectors share one storage engine and one query language.

## Graph-first retrieval — the graph filters the vectors

A naïve vector search conflates namesakes. WikiGraph runs **graph-first**: it resolves the question to **entities**, retains the dominant sense within a cosine margin, and **scopes the chunk vector search to that sense's documents** — then gathers evidence per entity via an `EntityMention` bridge, so a multi-entity question still returns every relevant result. The graph is not decoration; it is the filter that makes the vector search precise.

> 📸 *(screenshot: the "Knowledge graph" constellation view)*

## Keeping the graph clean — with the same vector engine

LLM extraction is inherently noisy: it spells one entity several ways and labels its type inconsistently. WikiGraph keeps the graph coherent in two complementary ways, both **entirely inside IRIS**.

While reading, ingestion runs **coreference-aware extraction** (LINK-KG style): each chunk is given the canonical entities found earlier in the same article, so mentions like "Luke" or "him" resolve to a single canonical node ("Luke Skywalker"), with the shorter forms kept as *aliases* rather than spawning duplicates. Entity **identity is the canonical name**, not the model's unreliable type — so "R2-D2 (Droid)" and "R2-D2 (Droide)" are one node, not two.

Anything that still slips through is handled by a separate **resolution pass** — and this is the part I like most as an IRIS demonstration: it needs no external entity-resolution tool, because it reuses the exact same vector machinery as retrieval. It **blocks** candidate pairs with an HNSW `VECTOR_COSINE` nearest-neighbour search, **matches** on name *plus* description agreement (so "Princess Leia" and "Leia Organa" merge into "Princess Leia Organa", while "Nikola Tesla" and "Tesla, Inc." stay apart), and **merges** each cluster into one canonical node — repointing its mentions and relationships. A dry-run previews every proposed merge before anything is written.

> 📸 *(screenshot: the "Preview merges" dialog in Manage knowledge)*

## Try it

```bash
# set IRIS_IMAGE / models in .env, then:
docker compose up -d --build
# → http://localhost:52773/wikigraph/
```

One command. The build incorporates the schema, the `EMBEDDING` config, and the production (compiled by the loader script), and marks it to auto-start. Ask a question and watch it learn a page in real time; open the **Knowledge graph** to view the entities it extracted; or add subjects directly by title in **Manage knowledge**.

---

WikiGraph is a *small* application, but it draws on a remarkable amount of the IRIS data platform simultaneously: the `EMBEDDING` datatype, `VECTOR_COSINE` + HNSW, a Python interoperability production with full message tracing, WSGI hosting, vector-powered entity resolution, and a graph model — all within a single instance and a single `docker compose up`. **That consolidation is the story.**


*Scaffolded from [quickpyprod](https://github.com/isc-amilheir/quickpyprod) · GraphRAG schema inspired by [iris-vector-rag](https://github.com/intersystems-community/iris-vector-rag).*

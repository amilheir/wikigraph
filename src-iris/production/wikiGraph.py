"""WikiGraph - Wikipedia GraphRAG chatbot production for InterSystems IRIS (pyprod).

This file holds ONLY the Production definition. The components live in sibling modules so
each concern is in its own file:
    common.py      - shared helpers and constants (plain Python, not compiled by pyprod)
    messages.py    - interoperability message classes (JsonSerialize)
    services.py    - chatPollAdapter + chatService (business service)
    processes.py   - chatProcess + ingestProcess (business processes)
    operations.py  - knowledgeOperation + wikipediaOperation + llmOperation (business operations)

pyprod compiles only the classes physically present in the AST of the file it is handed, so
the imports below do NOT recompile those modules - they only resolve the names referenced in
the Production definition at runtime. Compile every module with loadProduction.sh (which runs
intersystems_pyprod once per module, this file last so item validation sees the components).

Pipeline (every hop is a traceable interoperability message):
    chatService -> chatProcess -> knowledgeOperation (graph-first vector + 1-hop graph search)
                                -> wikipediaOperation (single page fetch, no linked pages)
                                -> ingestProcess -> llmOperation (entity extraction)
                                -> llmOperation (RAG answer generation via Ollama)

Embeddings use the IRIS EMBEDDING datatype with %Embedding.OpenAI pointed at the Ollama
container's OpenAI-compatible /v1/embeddings endpoint (config name 'wikiembed', model bge-m3,
1024 dims). IRIS never loads an embedding model in-process - each EMBEDDING() is a quick HTTP
call to Ollama - so there is no model-reload cost and no interop dead-job problem.
"""

import os

from intersystems_pyprod import Production, ServiceItem, ProcessItem, OperationItem

# pull every component into scope so the Production's config-item class names resolve at runtime
import common                # noqa: F401  (shared helpers; no compiled classes)
from messages import *       # noqa: F401,F403
from services import *       # noqa: F401,F403
from processes import *      # noqa: F401,F403
from operations import *     # noqa: F401,F403

iris_package_name = "WikiGraph"


# --------------------------------------------------------------------------
# Production definition
# --------------------------------------------------------------------------

class wikiGraphProduction(Production):
    description = "Wikipedia GraphRAG chatbot: scrape -> embed (IRIS EMBEDDING via Ollama) -> knowledge graph -> RAG"
    actor_pool_size = 2
    services = [
        ServiceItem("chatService", "WikiGraph.chatService", pool_size=1,
                    host_settings={"targetConfigName": "chatProcess"},
                    adapter_settings={"CallInterval": "1"})
    ]
    processes = [
        ProcessItem("chatProcess", "WikiGraph.chatProcess", pool_size=0,
                    host_settings={
                        "knowledgeTarget": "knowledgeOperation",
                        "wikipediaTarget": "wikipediaOperation",
                        "ingestTarget": "ingestProcess",
                        "llmTarget": "llmOperation",
                        "similarityThreshold": os.environ.get("SIMILARITY_THRESHOLD", "0.55")
                    }),
        ProcessItem("ingestProcess", "WikiGraph.ingestProcess", pool_size=0,
                    host_settings={"llmTarget": "llmOperation"})
    ]
    operations = [
        OperationItem("knowledgeOperation", "WikiGraph.knowledgeOperation", pool_size=1),
        OperationItem("wikipediaOperation", "WikiGraph.wikipediaOperation", pool_size=1,
                      host_settings={"llmTarget": "llmOperation"}),
        OperationItem("llmOperation", "WikiGraph.llmOperation", pool_size=1,
                      host_settings={
                          "ollamaUrl": os.environ.get("OLLAMA_URL", "http://ollama:11434"),
                          "ollamaModel": os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
                      }),
        OperationItem("resolutionOperation", "WikiGraph.resolutionOperation", pool_size=1)
    ]

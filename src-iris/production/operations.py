"""WikiGraph - business operations: GraphRAG retrieval, Wikipedia fetch and the Ollama LLM."""

import json
import time

import requests

from intersystems_pyprod import IRISProperty, BusinessOperation, IRISLog, Status

from common import EMBEDDING_CONFIG, statusIsOk, logErrorAndReturnStatus, sqlExec, aliasListFromJson
from messages import (knowledgeSearchResponse, wikiFetchResponse, extractTopicMsg, llmResponseMsg)

iris_package_name = "WikiGraph"


# --------------------------------------------------------------------------
# Business Operation - GraphRAG retrieval (graph-first, entity-scoped chunks)
# --------------------------------------------------------------------------

class knowledgeOperation(BusinessOperation):

    entityFloor = IRISProperty(
        default="0.3",
        description="Minimum entity cosine score to treat the query as 'about' that entity and scope chunks to its document.",
        settings="Retrieval")
    scopeFloor = IRISProperty(
        default="0.45",
        description="If the entity-scoped chunk search scores below this, widen to all documents (protects recall and the learning trigger).",
        settings="Retrieval")
    scopeMargin = IRISProperty(
        default="0.12",
        description="Only documents of entities within this cosine margin of the best entity are scoped. Smaller = stricter disambiguation (one sense wins); larger = include more co-relevant senses.",
        settings="Retrieval")
    nameMatch = IRISProperty(
        default=1, datatype=int,
        description="1 = also match entities whose full name literally appears in the question (regardless of vector score). Fixes multi-entity questions where one entity embeds weakly, e.g. 'Armstrong, Aldrin and Collins' - Collins is picked by name even if his description vector scores low.",
        settings="Retrieval")
    perEntityChunks = IRISProperty(
        default=2, datatype=int,
        description="Chunks gathered per matched/named entity via EntityMention, so EVERY named entity contributes its own evidence (not just the dominant ones). 0 disables per-entity gathering.",
        settings="Retrieval")
    maxChunks = IRISProperty(
        default=10, datatype=int,
        description="Cap on total chunks placed in the LLM context after merging the document-scoped search with the per-entity gather.",
        settings="Retrieval")
    maxChunkChars = IRISProperty(
        default=1200, datatype=int,
        description="Max characters of each chunk placed in the answer context. Bounds the prompt so it cannot fill num_ctx and starve the answer (a too-large prompt makes the LLM return a single token, done_reason=length).",
        settings="Retrieval")
    maxContextChars = IRISProperty(
        default=16000, datatype=int,
        description="Hard cap on the total characters of the context JSON sent to the answer LLM. Keep it well under num_ctx (chars/4 ~= tokens) so there is room left to generate the answer.",
        settings="Retrieval")

    MessageMap = {
        "WikiGraph.knowledgeSearchRequest": "onSearch"
    }

    def searchEntities(self, query, lang):
        """Graph-first resolution. Returns (entities, entityIds, scopeDocs) for the winning sense:
        only entities within scopeMargin of the top-scoring entity contribute their document,
        so a dominant match (e.g. Tesla the company) scopes out the rival sense (Tesla the person)."""
        embed = "EMBEDDING(?, '" + EMBEDDING_CONFIG + "')"
        floor = float(self.entityFloor)
        matched = []
        for row in sqlExec(
                "SELECT TOP 8 e.%ID, e.entityName, e.entityType, e.description, e.documentRef, "
                "VECTOR_COSINE(e.descVector, " + embed + ") AS score "
                "FROM GraphKB.Entity e WHERE e.lang = ? AND e.descVector IS NOT NULL ORDER BY score DESC",
                query, lang):
            score = float(row[5])
            if score >= floor:
                matched.append((int(row[0]), row[1], row[2], row[3], row[4], score))
        if not matched:
            return [], [], []
        cutoff = matched[0][5] - float(self.scopeMargin)   # rows are ordered by score desc
        inScope = [m for m in matched if m[5] >= cutoff]
        scopeDocs = []
        for m in inScope:
            docRef = m[4]
            if docRef is not None and str(docRef) != "" and int(docRef) not in scopeDocs:
                scopeDocs.append(int(docRef))
        entities = [{"name": m[1], "type": m[2], "description": m[3], "score": round(m[5], 3)} for m in inScope][:5]
        entityIds = [str(m[0]) for m in inScope]
        return entities, entityIds, scopeDocs

    def nameMatchEntities(self, query, lang):
        """Entities whose canonical name OR one of its aliases literally appears in the question,
        regardless of vector score. Catches every explicitly-named entity in a multi-entity question
        (e.g. each astronaut), and - via aliases - resolves a short form like 'Tesla' to whichever
        canonical entities carry it ('Nikola Tesla', 'Tesla, Inc.'), leaving the meaning-based scoping
        in onSearch to pick the right sense. Returns [(id, name, type, description), ...].
        Surface forms shorter than 4 chars are ignored so single-letter tokens don't match noise."""
        upperQuery = (query or "").upper()
        matched, seen = [], set()
        # fetch the language's entities with their aliases and match in Python (alias lists can't be
        # matched with a single SQL LIKE); the per-language entity count is small.
        for row in sqlExec(
                "SELECT TOP 500 e.%ID, e.entityName, e.entityType, e.description, e.aliases "
                "FROM GraphKB.Entity e WHERE e.lang = ?", lang):
            entityId = int(row[0])
            surfaceForms = [row[1]] if (row[1] and len(row[1]) >= 4) else []
            surfaceForms += [a for a in aliasListFromJson(row[4]) if len(a) >= 4]
            for form in surfaceForms:
                if form.upper() in upperQuery:
                    if entityId not in seen:
                        seen.add(entityId)
                        matched.append((entityId, row[1], row[2], row[3]))
                    break
            if len(matched) >= 12:
                break
        return matched

    def searchChunksByEntities(self, query, lang, entityIds, perEntity):
        """For each matched entity, the top chunks among those that MENTION it (via EntityMention),
        ranked by cosine to the query. This is the graph guiding the vector search per entity, so a
        named entity contributes evidence from whichever document actually describes it - even if its
        own entity row points elsewhere (Michael Collins's entity points to Apollo 11, but his birth
        date lives in the dedicated Michael Collins article, reachable through EntityMention)."""
        if perEntity <= 0 or not entityIds:
            return []
        embed = "EMBEDDING(?, '" + EMBEDDING_CONFIG + "')"
        chunks = []
        for entityId in entityIds:
            for row in sqlExec(
                    "SELECT TOP " + str(perEntity) + " c.chunkText, d.title, "
                    "VECTOR_COSINE(c.chunkVector, " + embed + ") AS score "
                    "FROM GraphKB.Chunk c "
                    "JOIN GraphKB.Document d ON c.documentRef = d.%ID "
                    "JOIN GraphKB.EntityMention em ON em.chunkRef = c.%ID "
                    "WHERE em.entityRef = ? AND d.lang = ? AND c.chunkVector IS NOT NULL ORDER BY score DESC",
                    query, int(entityId), lang):
                chunks.append({"title": row[1], "text": row[0], "score": round(float(row[2]), 3)})
        return chunks

    def searchChunks(self, query, lang, topK, scopeDocs):
        """Vector search over chunks, optionally restricted to a set of documents (the graph filter)."""
        embed = "EMBEDDING(?, '" + EMBEDDING_CONFIG + "')"
        docFilter = ""
        if scopeDocs:
            docFilter = " AND c.documentRef IN (" + ",".join(str(d) for d in scopeDocs) + ")"
        rows = sqlExec(
            "SELECT TOP " + str(topK) + " c.chunkText, d.title, "
            "VECTOR_COSINE(c.chunkVector, " + embed + ") AS score "
            "FROM GraphKB.Chunk c JOIN GraphKB.Document d ON c.documentRef = d.%ID "
            "WHERE d.lang = ? AND c.chunkVector IS NOT NULL" + docFilter + " ORDER BY score DESC",
            query, lang)
        # keep the context lean - a small LLM degenerates when flooded with text
        return [{"title": row[1], "text": row[0], "score": round(float(row[2]), 3)} for row in rows]

    def expandRelationships(self, entityIds):
        """1-hop graph expansion among the matched entities."""
        if not entityIds:
            return []
        idList = ",".join(entityIds)
        rows = sqlExec(
            "SELECT s.entityName, r.predicate, t.entityName, r.description "
            "FROM GraphKB.Relationship r "
            "JOIN GraphKB.Entity s ON r.sourceEntityRef = s.%ID "
            "JOIN GraphKB.Entity t ON r.targetEntityRef = t.%ID "
            "WHERE r.sourceEntityRef IN (" + idList + ") OR r.targetEntityRef IN (" + idList + ")")
        relationships, seen = [], set()
        for row in rows:
            triple = (row[0], row[1], row[2])
            if triple in seen:
                continue
            seen.add(triple)
            relationships.append({"source": row[0], "predicate": row[1], "target": row[2], "description": row[3]})
        return relationships[:30]

    def mergeChunks(self, *chunkLists):
        """Union chunk lists, dedup by (title, text) keeping the best score, sorted by score desc."""
        best = {}
        for chunkList in chunkLists:
            for chunk in chunkList:
                key = (chunk["title"], chunk["text"])
                if key not in best or chunk["score"] > best[key]["score"]:
                    best[key] = chunk
        return sorted(best.values(), key=lambda c: c["score"], reverse=True)

    def onSearch(self, request):
        try:
            topK = max(1, int(request.topK))

            # 1. GRAPH FIRST: resolve the query to entities and the documents they belong to.
            #    Vector match (the winning sense) + lexical name match (every entity named in the
            #    question), so a multi-entity question keeps all of its entities even when one of
            #    them embeds weakly.
            IRISLog.Info("Starting knowledge search for query: " + request.query)
            entities, entityIds, scopeDocs = self.searchEntities(request.query, request.lang)

            allIds = list(entityIds)
            mergedEntities = list(entities)
            if int(self.nameMatch):
                for entityId, name, entityType, description in self.nameMatchEntities(request.query, request.lang):
                    if str(entityId) not in allIds:
                        allIds.append(str(entityId))
                        mergedEntities.append({"name": name, "type": entityType, "description": description, "score": None})

            # 2. CHUNKS: the document-scoped vector search (disambiguated) UNION a per-entity gather
            #    that follows EntityMention so every matched/named entity contributes its own evidence.
            scopedChunks = self.searchChunks(request.query, request.lang, topK, scopeDocs)
            entityChunks = self.searchChunksByEntities(request.query, request.lang, allIds, int(self.perEntityChunks))
            chunks = self.mergeChunks(scopedChunks, entityChunks)
            bestScore = max((chunk["score"] for chunk in chunks), default=0.0)
            scoped = bool(scopeDocs)

            # 3. WIDEN: if the scope is empty or weak, search all documents so recall (and the
            #    'do I need to learn?' decision upstream) is not starved by an over-narrow scope
            if scoped and bestScore < float(self.scopeFloor):
                wider = self.searchChunks(request.query, request.lang, topK, [])
                chunks = self.mergeChunks(chunks, wider)
                bestScore = max((chunk["score"] for chunk in chunks), default=0.0)
                scoped = False

            chunks = chunks[:int(self.maxChunks)]

            # 4. graph edges among the matched entities
            relationships = self.expandRelationships(allIds)

            # graph-first context: entities + relationships frame the answer, chunks give the evidence.
            # title/score are only used internally (dedup, scoping, bestScore) - the LLM sees text only.
            # BUDGET THE CONTEXT: num_ctx is shared by prompt + output, so an oversized context leaves
            # no room to generate and the LLM returns a single token (done_reason=length). Cap each
            # chunk, bound entity/relationship descriptions, and stop once the total budget is hit.
            maxChunkChars = int(self.maxChunkChars)
            maxContextChars = int(self.maxContextChars)
            contextChunks, usedChars = [], 0
            for chunk in chunks:
                text = (chunk["text"] or "")[:maxChunkChars]
                if contextChunks and usedChars + len(text) > maxContextChars:
                    break
                contextChunks.append({"text": text})
                usedChars += len(text)
            contextEntities = [{"name": e.get("name"), "type": e.get("type"),
                                "description": (e.get("description") or "")[:300]} for e in mergedEntities[:8]]
            contextRelationships = [{"source": r["source"], "predicate": r["predicate"], "target": r["target"]}
                                    for r in relationships[:20]]
            contextJson = json.dumps({"chunks": contextChunks, "entities": contextEntities,
                                      "relationships": contextRelationships}, ensure_ascii=False)
            IRISLog.Info("knowledge search: " + str(len(mergedEntities)) + " entities ("
                         + str(len(allIds) - len(entityIds)) + " by name), " + str(len(chunks))
                         + " chunks (scoped=" + ("yes" if scoped else "no") + ", " + str(len(scopeDocs))
                         + " docs), best score " + str(round(bestScore, 3)))
            return Status.OK(), knowledgeSearchResponse(bestScore, len(chunks), contextJson)
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in knowledgeOperation onSearch: " + str(e))
            return status, knowledgeSearchResponse(0.0, 0, "{}")


# --------------------------------------------------------------------------
# Business Operation - Wikipedia API (single page only, no linked pages)
# --------------------------------------------------------------------------

class wikipediaOperation(BusinessOperation):

    userAgent = IRISProperty(
        default="WikiGraph/1.0",
        description="User-Agent sent to the Wikipedia API. Wikimedia policy throttles generic/contact-less agents - keep a tool name, a URL and ideally a contact email here.",
        settings="Wikipedia")
    maxRetries = IRISProperty(
        default=4, datatype=int,
        description="Max attempts per Wikipedia API call. 429/503 responses are retried with exponential backoff (honouring Retry-After) instead of failing through to extra requests.",
        settings="Wikipedia")
    minRequestIntervalMs = IRISProperty(
        default=300, datatype=int,
        description="Minimum gap between consecutive Wikipedia API calls (politeness throttle) to stay within rate limits.",
        settings="Wikipedia")
    maxExtractChars = IRISProperty(default=120000, datatype=int, settings="Wikipedia")
    searchCandidates = IRISProperty(
        default=5, datatype=int,
        description="How many search hits to consider. The candidate whose intro best matches the question is ingested - this is what tells 'Neil Armstrong' the person from 'RV Neil Armstrong' the ship.",
        settings="Wikipedia")
    llmTarget = IRISProperty(default="llmOperation", settings="Target:selector?context={Ens.ContextSearch/ProductionItems?targets=1&productionName=@productionId}")
    extractTopic = IRISProperty(
        default=1, datatype=int,
        description="1 = ask the LLM to distil the question into a clean Wikipedia search title before searching (better candidates). 0 = search the raw question.",
        settings="Wikipedia")

    MessageMap = {
        "WikiGraph.wikiFetchRequest": "onFetch"
    }

    def searchTopic(self, question, lang):
        """Distil the question into a clean Wikipedia search title via the LLM (falls back to the question)."""
        if int(self.extractTopic) != 1:
            return question
        try:
            status, resp = self.SendRequestSync(self.llmTarget, extractTopicMsg(question, lang))
            topic = resp.content.strip() if statusIsOk(status) else ""
            if topic:
                IRISLog.Info("LLM search topic: '" + topic + "' (from '" + question + "')")
                return topic
        except Exception as e:
            IRISLog.Warning("topic extraction failed, using raw question: " + str(e))
        return question

    def cosineToQuestion(self, text, question):
        """Semantic similarity of a candidate (title + intro) to the question, via Ollama."""
        try:
            for row in sqlExec(
                    "SELECT VECTOR_COSINE(EMBEDDING(?, '" + EMBEDDING_CONFIG + "'), "
                    "EMBEDDING(?, '" + EMBEDDING_CONFIG + "')) AS sc", text[:4000], question[:1000]):
                return float(row[0]) if row[0] is not None else 0.0
        except Exception as e:
            IRISLog.Warning("candidate rerank failed: " + str(e))
        return 0.0

    def wikiGet(self, apiUrl, params, timeout=(10, 60)):
        """GET the Wikipedia API politely: a descriptive User-Agent (per Wikimedia policy), a minimum
        gap between calls, and exponential backoff on 429/503 (honouring Retry-After). Raises only
        after retries are exhausted, so a transient 429 no longer cascades into the search fallback."""
        headers = {"User-Agent": self.userAgent, "Accept-Encoding": "gzip"}
        interval = max(0, int(self.minRequestIntervalMs)) / 1000.0
        gap = time.time() - getattr(self, "_lastWikiTs", 0.0)
        if gap < interval:
            time.sleep(interval - gap)
        attempts = max(1, int(self.maxRetries))
        response = None
        for attempt in range(attempts):
            response = requests.get(apiUrl, headers=headers, params=params, timeout=timeout)
            self._lastWikiTs = time.time()
            if response.status_code in (429, 503) and attempt < attempts - 1:
                retryAfter = response.headers.get("Retry-After", "")
                wait = float(retryAfter) if retryAfter.isdigit() else min(2 ** attempt, 30)
                IRISLog.Warning("Wikipedia " + str(response.status_code) + " - backing off "
                                + str(wait) + "s (attempt " + str(attempt + 1) + "/" + str(attempts) + ")")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        response.raise_for_status()
        return response

    def onFetch(self, request):
        try:
            lang = request.lang if request.lang in ("en", "pt") else "en"
            apiUrl = "https://" + lang + ".wikipedia.org/w/api.php"

            # 0. distil the question into a clean search title via the LLM
            queryTitle = self.searchTopic(request.topic, lang)

            # 1. resolve the title FOLLOWING REDIRECTS (redirects=1). If it lands on a real article
            #    - e.g. "Bruce Wayne" -> "Batman", "Anakin Skywalker" -> "Darth Vader" - store that
            #    redirected page directly. Only when the title is MISSING or resolves to a
            #    DISAMBIGUATION page do we fall back to the candidate search + rerank in step 2.
            chosenId = None
            try:
                resolveResp = self.wikiGet(apiUrl, {
                    "action": "query", "redirects": 1, "prop": "info|pageprops", "inprop": "url",
                    "ppprop": "disambiguation", "titles": queryTitle, "format": "json"}, timeout=(10, 60))
                resolved = next(iter(resolveResp.json().get("query", {}).get("pages", {}).values()), {})
                isMissing = ("missing" in resolved) or (resolved.get("pageid") is None)
                isDisambig = "disambiguation" in (resolved.get("pageprops") or {})
                if not isMissing and not isDisambig:
                    chosenId = int(resolved["pageid"])
                    IRISLog.Info("resolved '" + queryTitle + "' (redirects followed) to '"
                                 + resolved.get("title", "") + "' pageId " + str(chosenId))
                else:
                    IRISLog.Info("'" + queryTitle + "' is "
                                 + ("missing" if isMissing else "a disambiguation page")
                                 + " - falling back to candidate search")
            except Exception as e:
                IRISLog.Warning("title resolve failed, using search: " + str(e))

            # 2. fall back: gather candidate pages for the title and pick the one whose intro best
            #    matches the question - excluding disambiguation pages from the candidates.
            if chosenId is None:
                searchResponse = self.wikiGet(apiUrl, {
                    "action": "query", "list": "search", "srsearch": queryTitle,
                    "srlimit": max(1, int(self.searchCandidates)), "format": "json"}, timeout=(10, 60))
                results = searchResponse.json().get("query", {}).get("search", [])
                if not results:
                    return Status.OK(), wikiFetchResponse(0, 0, "", "", "")
                candidateIds = [str(r["pageid"]) for r in results]

                introResponse = self.wikiGet(apiUrl, {
                    "action": "query", "prop": "extracts|pageprops", "exintro": 1, "explaintext": 1,
                    "ppprop": "disambiguation", "redirects": 1, "pageids": "|".join(candidateIds),
                    "format": "json"}, timeout=(10, 90))
                pages = introResponse.json().get("query", {}).get("pages", {})
                # exclude disambiguation pages from the candidate set
                candidateIds = [pid for pid in candidateIds
                                if "disambiguation" not in (pages.get(pid, {}).get("pageprops") or {})]
                if not candidateIds:
                    return Status.OK(), wikiFetchResponse(0, 0, "", "", "")

                chosenId = int(candidateIds[0])
                if len(candidateIds) > 1:
                    bestScore = -1.0
                    for pid in candidateIds:
                        page = pages.get(pid, {})
                        intro = (page.get("extract") or "")[:1500]
                        if not intro:
                            continue
                        # embed "title - intro" to mirror how stored chunks are embedded
                        IRISLog.Info(page.get("title", "") + " — " + intro + " — " + request.topic)
                        score = self.cosineToQuestion(page.get("title", "") + " — " + intro, request.topic)
                        if score > bestScore:
                            bestScore, chosenId = score, int(pid)
                    IRISLog.Info("disambiguated '" + request.topic + "' to pageId " + str(chosenId)
                                 + " among " + str(len(candidateIds)) + " candidates (score "
                                 + str(round(bestScore, 3)) + ")")

            # 3. fetch ONLY the chosen page's plain-text extract - linked pages are never followed
            extractResponse = self.wikiGet(apiUrl, {
                "action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
                "inprop": "url", "pageids": chosenId, "format": "json"}, timeout=(10, 120))
            page = extractResponse.json().get("query", {}).get("pages", {}).get(str(chosenId), {})
            extract = (page.get("extract") or "")[:int(self.maxExtractChars)]
            if not extract:
                return Status.OK(), wikiFetchResponse(0, 0, "", "", "")

            title = page.get("title", "")
            url = page.get("fullurl", "https://" + lang + ".wikipedia.org/?curid=" + str(chosenId))
            IRISLog.Info("fetched Wikipedia page '" + title + "' (" + str(len(extract)) + " chars)")
            return Status.OK(), wikiFetchResponse(1, chosenId, title, url, extract)
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in wikipediaOperation onFetch: " + str(e))
            return status, wikiFetchResponse(0, 0, "", "", "")


# --------------------------------------------------------------------------
# Business Operation - Ollama LLM (entity extraction + answer generation)
# --------------------------------------------------------------------------

EXTRACTION_PROMPT = (
    "You are an information extraction engine building a knowledge graph. "
    "Extract the named entities and the relationships between them from the user's text. "
    "For each entity, use its FULL, CANONICAL name as \"name\" (e.g. 'Luke Skywalker' not 'Luke', "
    "'Tesla, Inc.' not 'Tesla'). Resolve pronouns and partial mentions to that canonical entity, and "
    "put any shorter or alternate forms that appear in the text into \"aliases\". "
    "Respond ONLY with JSON in exactly this shape: "
    '{"entities": [{"name": "...", "type": "...", "aliases": ["..."], "description": "..."}], '
    '"relationships": [{"source": "...", "target": "...", "predicate": "...", "description": "..."}]} '
    "Relationships MUST reference entities by their canonical \"name\". "
    "Use short predicates (e.g. bornIn, authorOf, partOf). Keep descriptions short. "
    "Limit to the 6 most important entities so the JSON stays small.")

# appended to EXTRACTION_PROMPT when earlier chunks of the same article already found entities, so
# the model reuses the established canonical name instead of spawning a duplicate under a variant.
KNOWN_ENTITIES_PROMPT = (
    "\n\nEntities already identified in this article. If a mention refers to one of these, reuse its "
    "EXACT canonical name (do not invent a new spelling or a different type):\n{known}")

ANSWER_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the knowledge "
    "context JSON provided (text chunks, entities and relationship triples from a knowledge graph). "
    "If the context is insufficient, say so honestly. Answer in the language of the question ({lang}). "
    "Be concise. Answer the questions as the context "
    "were knowledge that you already know.\n\nKnowledge context:\n{context}")

TOPIC_PROMPT = (
    "Turn the user's question into the single best Wikipedia article title to look up. "
    "Reply with ONLY that title - the main subject - no quotes, no punctuation, no explanation. "
    "Examples: 'Who was the first person to walk on the moon?' -> Neil Armstrong; "
    "'tell me about the electric car company Tesla' -> Tesla, Inc.; "
    "'quem foi Santos Dumont?' -> Santos Dumont.")


class llmOperation(BusinessOperation):

    ollamaUrl = IRISProperty(
        default="http://ollama:11434",
        description="Ollama base URL (set from OLLAMA_URL; editable here in the Portal)",
        settings="Ollama")
    ollamaModel = IRISProperty(
        default="gemma4:e2b",
        description="Ollama model tag (set from OLLAMA_MODEL; editable here in the Portal)",
        settings="Ollama")
    # --- response tuning, all editable live in the Management Portal under "Ollama Options" ---
    maxTokens = IRISProperty(
        default=400, datatype=int,
        description="Max tokens the LLM may generate per response (Ollama num_predict). The main speed lever - lower is faster.",
        settings="Ollama Options")
    temperature = IRISProperty(
        default="0.3",
        description="Sampling temperature (Ollama temperature). Lower = more focused and settles sooner.",
        settings="Ollama Options")
    numCtx = IRISProperty(
        default=8192, datatype=int,
        description="Context window in tokens (Ollama num_ctx). Larger costs more time and memory.",
        settings="Ollama Options")
    topP = IRISProperty(
        default="0.9",
        description="Nucleus sampling cutoff (Ollama top_p). Lower = more focused output.",
        settings="Ollama Options")
    topK = IRISProperty(
        default=40, datatype=int,
        description="Sampling considers only the top K tokens (Ollama top_k). Lower = more focused output.",
        settings="Ollama Options")
    keepAlive = IRISProperty(
        default="30m",
        description="How long Ollama keeps the model resident between calls (avoids per-call reload latency).",
        settings="Ollama Options")
    requestTimeout = IRISProperty(
        default=600, datatype=int,
        description="HTTP read timeout in seconds for the Ollama call.",
        settings="Ollama Options")
    extractMaxTokens = IRISProperty(
        default=1200, datatype=int,
        description="Separate (larger) token budget for entity-extraction JSON. Too low truncates the JSON and the chunk's entities are lost.",
        settings="Ollama Options")
    answerThink = IRISProperty(
        default=1, datatype=int,
        description="1 = let the model 'think' before the FINAL answer (Ollama think mode). The reasoning is discarded and only the final answer is returned. Needs answerMaxTokens large enough to cover reasoning + the full answer.",
        settings="Ollama Options")
    answerMaxTokens = IRISProperty(
        default=1024, datatype=int,
        description="Token budget for the final answer. Must be larger than maxTokens when answerThink=1 so the reasoning AND the complete answer both fit (otherwise the answer is truncated/empty).",
        settings="Ollama Options")

    MessageMap = {
        "WikiGraph.extractTopicMsg": "onExtractTopic",
        "WikiGraph.extractEntitiesMsg": "onExtractEntities",
        "WikiGraph.generateAnswerMsg": "onGenerateAnswer"
    }

    def llmConfig(self):
        return self.ollamaUrl.rstrip("/"), self.ollamaModel

    def chat(self, systemPrompt, userPrompt, jsonMode=False, maxTokens=None, think=False):
        baseUrl, model = self.llmConfig()
        numPredict = int(maxTokens) if maxTokens is not None else int(self.maxTokens)
        payload = {
            "model": model,
            "stream": False,
            # think mode: when on, Ollama returns reasoning in message.thinking and the final answer
            # in message.content. Off for extraction/topic (clean JSON / direct output, no wasted budget).
            "think": bool(think),
            "keep_alive": self.keepAlive,
            "options": {
                "num_predict": numPredict,
                "temperature": float(self.temperature),
                "num_ctx": int(self.numCtx),
                "top_p": float(self.topP),
                "top_k": int(self.topK)
            },
            "messages": [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ]
        }
        IRISLog.Info("Payload: " + str(payload))
        if jsonMode:
            payload["format"] = "json"
        response = requests.post(
            baseUrl + "/api/chat", json=payload, timeout=(10, int(self.requestTimeout)))
        response.raise_for_status()
        message = response.json().get("message", {})
        IRISLog.Info("Response: " + str(message))
        if think:
            # return ONLY the final answer; the reasoning (message.thinking) is intentionally discarded
            return message.get("content") or ""
        # non-thinking: content, falling back to thinking text if a model emitted it there
        return message.get("content") or message.get("thinking") or ""

    def onExtractTopic(self, request):
        try:
            content = self.chat(TOPIC_PROMPT, request.question, maxTokens=40)
            topic = (content or "").strip().strip('"').strip("'")
            topic = topic.splitlines()[0][:200] if topic else ""
            return Status.OK(), llmResponseMsg(topic)
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in llmOperation onExtractTopic: " + str(e))
            return status, llmResponseMsg("")

    def onExtractEntities(self, request):
        try:
            systemPrompt = EXTRACTION_PROMPT
            known = (getattr(request, "knownEntities", "") or "").strip()
            if known and known not in ("[]", "{}"):
                systemPrompt += KNOWN_ENTITIES_PROMPT.format(known=known)
            content = self.chat(systemPrompt, request.chunkText, jsonMode=True,
                                maxTokens=int(self.extractMaxTokens))
            return Status.OK(), llmResponseMsg(content)
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in llmOperation onExtractEntities: " + str(e))
            return status, llmResponseMsg("")

    def onGenerateAnswer(self, request):
        try:
            systemPrompt = ANSWER_PROMPT.format(lang=request.lang, context=request.contextJson)
            think = bool(int(self.answerThink))
            content = self.chat(systemPrompt, request.question, think=think,
                                maxTokens=int(self.answerMaxTokens))
            if think and not (content or "").strip():
                # reasoning consumed the whole budget before answering - retry without thinking so
                # the user still gets a complete answer
                IRISLog.Warning("answer empty after thinking, retrying without think")
                content = self.chat(systemPrompt, request.question, maxTokens=int(self.answerMaxTokens))
            return Status.OK(), llmResponseMsg(content)
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in llmOperation onGenerateAnswer: " + str(e))
            return status, llmResponseMsg("")

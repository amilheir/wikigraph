"""WikiGraph - business processes: the chat orchestrator and the ingestion pipeline."""

import json

from intersystems_pyprod import IRISProperty, BusinessProcess, IRISLog, Status

from common import (statusIsOk, logErrorAndReturnStatus, updateChatStatus, documentExists,
                    insertRow, nowTimestamp, chunkArticleText, parseLlmJson, sqlExec,
                    normalizeAliases, mergeAliasJson, normalizeName, normalizeType)
from messages import (chatRequestMsg, chatResponseMsg, knowledgeSearchRequest, wikiFetchRequest,
                      ingestRequest, ingestResponse, extractEntitiesMsg, generateAnswerMsg)

iris_package_name = "WikiGraph"


# --------------------------------------------------------------------------
# Business Process - chat orchestrator
# --------------------------------------------------------------------------

class chatProcess(BusinessProcess):

    knowledgeTarget = IRISProperty(default="knowledgeOperation", settings="Target:selector?context={Ens.ContextSearch/ProductionItems?targets=1&productionName=@productionId}")
    wikipediaTarget = IRISProperty(default="wikipediaOperation", settings="Target:selector?context={Ens.ContextSearch/ProductionItems?targets=1&productionName=@productionId}")
    ingestTarget = IRISProperty(default="ingestProcess", settings="Target:selector?context={Ens.ContextSearch/ProductionItems?targets=1&productionName=@productionId}")
    llmTarget = IRISProperty(default="llmOperation", settings="Target:selector?context={Ens.ContextSearch/ProductionItems?targets=1&productionName=@productionId}")
    similarityThreshold = IRISProperty(
        default="0.55",
        description="Below this best cosine score the process learns from Wikipedia",
        settings="Search")
    searchTopK = IRISProperty(
        default=5, datatype=int,
        description="Number of chunks the document-scoped vector search returns per knowledge query (the topK sent to knowledgeOperation, before the per-entity gather and the maxChunks merge cap).",
        settings="Search")

    def OnRequest(self, request):
        requestId = request.requestId
        try:
            # "ingest" mode: add a Wikipedia article by title, no answer generated
            if (getattr(request, "mode", "") or "chat") == "ingest":
                return self.addArticle(requestId, request.question, request.lang)

            updateChatStatus(requestId, "searching", "Searching the knowledge base")
            status, searchResp = self.SendRequestSync(
                self.knowledgeTarget, knowledgeSearchRequest(request.question, request.lang, self.searchTopK))
            if not statusIsOk(status):
                raise RuntimeError("knowledge search failed: " + str(status))

            if float(searchResp.bestScore) < float(self.similarityThreshold):
                updateChatStatus(requestId, "learning", "Learning about this topic from Wikipedia")
                status, wikiResp = self.SendRequestSync(
                    self.wikipediaTarget, wikiFetchRequest(request.question, request.lang, 1))
                if statusIsOk(status) and int(wikiResp.found) == 1:
                    if not documentExists(wikiResp.pageId, request.lang):
                        updateChatStatus(requestId, "learning",
                                         "Reading '" + wikiResp.title + "' and building the knowledge graph")
                        status, _ = self.SendRequestSync(
                            self.ingestTarget,
                            ingestRequest(wikiResp.pageId, wikiResp.title, request.lang, wikiResp.url, wikiResp.extract))
                        if not statusIsOk(status):
                            IRISLog.Warning("ingest failed, answering with available knowledge: " + str(status))
                    # re-query the (now enriched) knowledge base
                    status, searchResp = self.SendRequestSync(
                        self.knowledgeTarget, knowledgeSearchRequest(request.question, request.lang, self.searchTopK))
                else:
                    IRISLog.Warning("No Wikipedia page found for: " + request.question)

            updateChatStatus(requestId, "answering", "Generating the answer")
            status, llmResp = self.SendRequestSync(
                self.llmTarget, generateAnswerMsg(request.question, request.lang, searchResp.contextJson))
            if not statusIsOk(status):
                raise RuntimeError("answer generation failed: " + str(status))

            updateChatStatus(requestId, "done", "", llmResp.content)
            return Status.OK(), chatResponseMsg(requestId, llmResp.content)
        except Exception as e:
            IRISLog.Error("ERROR in chatProcess OnRequest: " + str(e))
            updateChatStatus(requestId, "error", str(e))
            return Status.ERROR(str(e)), chatResponseMsg(requestId, "")

    def addArticle(self, requestId, title, lang):
        """Add a Wikipedia article to the knowledge base by title - fetch the page and ingest it,
        without generating an answer. Used by the 'Add an article' control in Manage knowledge."""
        try:
            updateChatStatus(requestId, "learning", "Looking up '" + title + "' on Wikipedia")
            # add-by-title: search the exact title, no LLM distillation (it can swap the subject,
            # e.g. 'han solo' -> 'Darth Vader'); the user already gave the canonical title.
            status, wikiResp = self.SendRequestSync(self.wikipediaTarget, wikiFetchRequest(title, lang, 0))
            if not (statusIsOk(status) and int(wikiResp.found) == 1):
                updateChatStatus(requestId, "error", "No Wikipedia article found for '" + title + "'")
                return Status.OK(), chatResponseMsg(requestId, "")

            if documentExists(wikiResp.pageId, lang):
                updateChatStatus(requestId, "done", "", "'" + wikiResp.title + "' is already in the knowledge base.")
                return Status.OK(), chatResponseMsg(requestId, wikiResp.title)

            updateChatStatus(requestId, "learning", "Reading '" + wikiResp.title + "' and building the knowledge graph")
            status, _ = self.SendRequestSync(
                self.ingestTarget,
                ingestRequest(wikiResp.pageId, wikiResp.title, lang, wikiResp.url, wikiResp.extract))
            if not statusIsOk(status):
                updateChatStatus(requestId, "error", "Could not ingest '" + wikiResp.title + "'")
                return Status.ERROR(str(status)), chatResponseMsg(requestId, "")

            updateChatStatus(requestId, "done", "", "Added '" + wikiResp.title + "' to the knowledge base.")
            return Status.OK(), chatResponseMsg(requestId, wikiResp.title)
        except Exception as e:
            IRISLog.Error("ERROR in chatProcess addArticle: " + str(e))
            updateChatStatus(requestId, "error", str(e))
            return Status.ERROR(str(e)), chatResponseMsg(requestId, "")


# --------------------------------------------------------------------------
# Business Process - ingestion (chunk -> embed -> knowledge graph)
# --------------------------------------------------------------------------

class ingestProcess(BusinessProcess):

    llmTarget = IRISProperty(default="llmOperation", settings="Target:selector?context={Ens.ContextSearch/ProductionItems?targets=1&productionName=@productionId}")
    chunkChars = IRISProperty(
        default=2500, datatype=int,
        description="Target maximum characters per chunk (paragraph-aware split). Larger = more context per chunk and more of the article covered within maxEmbedChunks, but coarser vector matching.",
        settings="Chunks")
    maxEmbedChunks = IRISProperty(
        default=80, datatype=int,
        description="Only the first N chunks of an article are stored/embedded",
        settings="Chunks")
    maxExtractChunks = IRISProperty(
        default=6, datatype=int,
        description="Entity extraction (LLM) runs only on the first N stored chunks",
        settings="Chunks")
    maxEntitiesPerChunk = IRISProperty(
        default=6, datatype=int,
        description="Cap on entities stored per extracted chunk",
        settings="Chunks")

    def OnRequest(self, request):
        try:
            documentId = insertRow("GraphKB.Document", {
                "pageId": int(request.pageId), "title": request.title, "lang": request.lang,
                "url": request.url, "ingestedAt": nowTimestamp()})

            chunks = chunkArticleText(request.text, int(self.chunkChars))[:int(self.maxEmbedChunks)]
            chunkIds = []
            for index, chunkText in enumerate(chunks):
                # embedText (= article title + chunk) is the EMBEDDING source: prefixing the title
                # separates senses in vector space (e.g. "Tesla, Inc." chunks vs "Nikola Tesla" chunks)
                # while chunkText stays clean for display. The EMBEDDING column computes via Ollama on %Save().
                chunkIds.append(insertRow("GraphKB.Chunk", {
                    "documentRef": documentId, "chunkIndex": index,
                    "chunkText": chunkText[:29000],
                    "embedText": (request.title + " — " + chunkText)[:30900]}))

            # LINK-KG registry: canonical entities found so far in THIS article (name -> type).
            # Fed back into each chunk's extraction so the LLM reuses the established canonical name
            # (resolving "Luke" -> "Luke Skywalker") instead of spawning a duplicate under a variant.
            registry = {}
            entityCount = 0
            for index in range(min(len(chunks), int(self.maxExtractChunks))):
                knownEntities = json.dumps(
                    [{"name": n, "type": t} for n, t in list(registry.items())[:50]], ensure_ascii=False)
                status, llmResp = self.SendRequestSync(
                    self.llmTarget, extractEntitiesMsg(request.lang, chunks[index], knownEntities))
                if not statusIsOk(status):
                    IRISLog.Warning("entity extraction failed for chunk " + str(index) + ": " + str(status))
                    continue
                entityCount += self.storeGraph(llmResp.content, request.lang, chunkIds[index], documentId, registry)

            IRISLog.Info("Ingested '" + request.title + "': " + str(len(chunks)) + " chunks, "
                         + str(entityCount) + " entities")
            return Status.OK(), ingestResponse(documentId, len(chunks), entityCount)
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in ingestProcess OnRequest: " + str(e))
            return status, ingestResponse(0, 0, 0)

    def upsertEntity(self, name, entityType, description, lang, documentRef, aliases=None):
        name = (name or "").strip()[:290]
        if not name:
            return 0
        entityType = normalizeType(entityType)[:90]
        aliasList = normalizeAliases(aliases)
        # dedup by name + lang only: identity is the canonical NAME, not the LLM's free-text type. The
        # type was unreliable as a key - the same entity gets typed "Droid" on one chunk and "Droide"
        # on the next, which split it into duplicate nodes. entityType is kept as a plain attribute.
        # Same-name-different-thing senses (Nikola Tesla vs Tesla, Inc.) stay distinct because they are
        # extracted under different canonical names; a short form they share (e.g. "Tesla") is recorded
        # in aliases on each, so both are findable by it while the meaning-based scoping in
        # knowledgeOperation picks the right sense.
        for row in sqlExec(
                "SELECT %ID, aliases FROM GraphKB.Entity WHERE entityName = ? AND lang = ?",
                name, lang):
            entityId = int(row[0])
            mergedAliases = mergeAliasJson(row[1], aliasList)
            if mergedAliases is not None:
                sqlExec("UPDATE GraphKB.Entity SET aliases = ? WHERE %ID = ?", mergedAliases, entityId)
            return entityId
        # descVector EMBEDDING column computes via Ollama on %Save()
        return insertRow("GraphKB.Entity", {
            "entityName": name, "entityType": entityType, "lang": lang,
            "documentRef": documentRef, "description": (description or name)[:3900],
            "aliases": json.dumps(aliasList, ensure_ascii=False) if aliasList else ""})

    def storeGraph(self, llmContent, lang, chunkId, documentId, registry):
        """Parse the LLM extraction JSON and persist entities, mentions and relationships.
        Updates `registry` (canonical name -> type) so later chunks reuse the same canonical names."""
        try:
            graph = parseLlmJson(llmContent)
        except Exception as e:
            IRISLog.Warning("could not parse extraction JSON: " + str(e))
            return 0
        if not isinstance(graph, dict):
            IRISLog.Warning("extraction JSON was not a JSON object; skipping chunk")
            return 0

        # the LLM sometimes returns entities/relationships as bare strings instead of objects, so
        # keep only the dict items - otherwise a malformed chunk crashes the whole ingest with
        # 'str' object has no attribute 'get'.
        chunkEntities = [e for e in graph.get("entities", []) if isinstance(e, dict)][:int(self.maxEntitiesPerChunk)]
        # every canonical name in play (this chunk + the running registry) - an alias that matches
        # one of these is really a DIFFERENT entity mislabelled as an alias, so drop it.
        otherNames = set(normalizeName(e.get("name") or "") for e in chunkEntities)
        otherNames.update(normalizeName(n) for n in registry.keys())
        otherNames.discard("")

        nameToId = {}     # canonical name AND aliases -> id, so relationships resolve either way
        storedIds = set()
        for entity in chunkEntities:
            name = (entity.get("name") or "").strip()
            selfNorm = normalizeName(name)
            aliases = [a for a in (entity.get("aliases") or [])
                       if isinstance(a, str)
                       and not (normalizeName(a) in otherNames and normalizeName(a) != selfNorm)]
            entityId = self.upsertEntity(name, entity.get("type"), entity.get("description"),
                                         lang, documentId, aliases)
            if not entityId:
                continue
            isNew = entityId not in storedIds
            storedIds.add(entityId)
            registry[name] = (entity.get("type") or "Unknown")
            nameToId[name.lower()] = entityId
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    nameToId.setdefault(alias.strip().lower(), entityId)
            if isNew:
                sqlExec("INSERT INTO GraphKB.EntityMention (entityRef, chunkRef) VALUES (?, ?)", entityId, chunkId)

        for relationship in graph.get("relationships", []):
            if not isinstance(relationship, dict):
                continue
            sourceId = nameToId.get((relationship.get("source") or "").strip().lower())
            targetId = nameToId.get((relationship.get("target") or "").strip().lower())
            if sourceId and targetId and sourceId != targetId:
                sqlExec(
                    "INSERT INTO GraphKB.Relationship (sourceEntityRef, targetEntityRef, predicate, description, chunkRef) VALUES (?, ?, ?, ?, ?)",
                    sourceId, targetId, (relationship.get("predicate") or "relatedTo")[:190],
                    (relationship.get("description") or "")[:3900], chunkId)
        return len(storedIds)

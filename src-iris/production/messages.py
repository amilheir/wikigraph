"""WikiGraph - interoperability message classes (compiled into the WikiGraph package)."""

from intersystems_pyprod import JsonSerialize, Column

iris_package_name = "WikiGraph"


class chatRequestMsg(JsonSerialize):
    requestId = Column(datatype=int, index=True)
    question = Column()
    lang = Column()
    mode = Column()   # "chat" (ask a question) or "ingest" (add a Wikipedia article by title)


class chatResponseMsg(JsonSerialize):
    requestId = Column(datatype=int)
    answer = ""


class knowledgeSearchRequest(JsonSerialize):
    query = Column()
    lang = Column()
    topK = Column(datatype=int)


class knowledgeSearchResponse(JsonSerialize):
    bestScore = 0.0
    matched = Column(datatype=int)
    contextJson = ""


class wikiFetchRequest(JsonSerialize):
    topic = Column()
    lang = Column()


class wikiFetchResponse(JsonSerialize):
    found = Column(datatype=int)
    pageId = Column(datatype=int)
    title = Column()
    url = Column()
    extract = ""


class ingestRequest(JsonSerialize):
    pageId = Column(datatype=int)
    title = Column()
    lang = Column()
    url = Column()
    text = ""


class ingestResponse(JsonSerialize):
    documentId = Column(datatype=int)
    chunkCount = Column(datatype=int)
    entityCount = Column(datatype=int)


class extractTopicMsg(JsonSerialize):
    question = Column()
    lang = Column()


class extractEntitiesMsg(JsonSerialize):
    lang = Column()
    chunkText = ""
    knownEntities = ""   # JSON list of canonical entities found earlier in this article (LINK-KG registry)


class generateAnswerMsg(JsonSerialize):
    question = Column()
    lang = Column()
    contextJson = ""


class llmResponseMsg(JsonSerialize):
    content = ""

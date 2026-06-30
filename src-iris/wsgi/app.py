"""WikiGraph WSGI backend - Flask app hosted inside IRIS (%SYS.Python.WSGI).

The frontend talks to this app over the WSGI protocol. POST /api/chat just inserts a 'pending'
GraphKB.ChatRequest row; the chatService polling adapter inside the interoperability production
picks it up and drives the pipeline. The frontend polls GET /api/chat/<id> for the result. The
app never calls into the production directly (that would register the web worker as an Ens job
and race across workers with <Ens>ErrJobRegistryNotClean).
"""

import datetime
import json
import os

from flask import Flask, jsonify, request, send_from_directory

import iris

SUPPORTED_LANGS = ("en", "pt")

appDir = os.path.dirname(os.path.abspath(__file__))
staticDir = os.path.join(appDir, "static")
docsDir = os.path.join(appDir, "docs")
app = Flask(__name__)


def sqlExec(query, *args):
    return iris.sql.exec(query, *args)


def sqlExecSafe(query, *args):
    """Run a DELETE/UPDATE that may affect zero rows (iris.sql.exec raises SQLCODE 100 otherwise)."""
    try:
        iris.sql.exec(query, *args)
    except Exception:
        pass


def langFilter():
    """The ?lang= query param if it is a supported language, else None (no filter).

    The knowledge graph is stored per-language via the lang column on Document/Entity, so the
    read endpoints scope to one language at a time: an EN session never sees the PT graph and
    vice-versa. Missing/invalid lang returns None so the endpoint stays backward compatible."""
    lang = (request.args.get("lang") or "").lower()
    return lang if lang in SUPPORTED_LANGS else None


@app.route("/")
def index():
    # Serve index.html as a plain byte body rather than via send_from_directory: the IRIS WSGI
    # gateway truncates the file-wrapper response for the root path at ~16KB, which left the page
    # blank below the header once index.html grew past that size. A plain Response is drained fully.
    with open(os.path.join(staticDir, "index.html"), "rb") as fh:
        return app.response_class(fh.read(), mimetype="text/html")


@app.route("/docs/<path:filename>")
def docs(filename):
    """Serve the standalone project documentation (e.g. /docs/overview.html), shown in the chat's
    Documentation side panel via an iframe."""
    return send_from_directory(docsDir, filename)


@app.route("/api/documents", methods=["GET"])
def listDocuments():
    lang = langFilter()
    query = ("SELECT d.%ID, d.title, d.lang, "
             "(SELECT COUNT(*) FROM GraphKB.Chunk c WHERE c.documentRef = d.%ID) AS chunks, "
             "(SELECT COUNT(*) FROM GraphKB.Entity e WHERE e.documentRef = d.%ID) AS entities "
             "FROM GraphKB.Document d")
    rows = sqlExec(query + " WHERE d.lang = ? ORDER BY d.title", lang) if lang \
        else sqlExec(query + " ORDER BY d.title")
    docs = []
    for row in rows:
        docs.append({"id": int(row[0]), "title": row[1], "lang": row[2],
                     "chunks": int(row[3]), "entities": int(row[4])})
    return jsonify({"documents": docs})


@app.route("/api/documents", methods=["POST"])
def addDocument():
    """Queue a Wikipedia article to be fetched and ingested by title. Inserts an 'ingest'
    ChatRequest row that the chatService poller turns into a fetch + ingest (no answer)."""
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    lang = (payload.get("lang") or os.environ.get("DEFAULT_LANG", "en")).lower()
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    if not title:
        return jsonify({"error": "title is required"}), 400

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = iris.cls("GraphKB.ChatRequest")._New()
    row.question = title[:1900]
    row.lang = lang
    row.status = "ingest"          # the chatService poller treats this as an add-article request
    row.statusDetail = ""
    row.createdAt = now
    row.updatedAt = now
    if str(row._Save()) != "1":
        return jsonify({"error": "could not queue the article"}), 500
    return jsonify({"requestId": int(row._Id()), "status": "ingest"})


@app.route("/api/documents/<int:documentId>", methods=["DELETE"])
def deleteDocument(documentId):
    title = None
    for row in sqlExec("SELECT title FROM GraphKB.Document WHERE %ID = ?", documentId):
        title = row[0]
    if title is None:
        return jsonify({"error": "article not found"}), 404

    # cascade: remove every chunk, entity, mention and relationship tied to this article.
    chunkSub = "(SELECT %ID FROM GraphKB.Chunk WHERE documentRef = " + str(documentId) + ")"
    entitySub = "(SELECT %ID FROM GraphKB.Entity WHERE documentRef = " + str(documentId) + ")"
    sqlExecSafe("DELETE FROM GraphKB.EntityMention WHERE chunkRef IN " + chunkSub + " OR entityRef IN " + entitySub)
    sqlExecSafe("DELETE FROM GraphKB.Relationship WHERE chunkRef IN " + chunkSub
                + " OR sourceEntityRef IN " + entitySub + " OR targetEntityRef IN " + entitySub)
    sqlExecSafe("DELETE FROM GraphKB.Chunk WHERE documentRef = ?", documentId)
    sqlExecSafe("DELETE FROM GraphKB.Entity WHERE documentRef = ?", documentId)
    sqlExecSafe("DELETE FROM GraphKB.Document WHERE %ID = ?", documentId)
    return jsonify({"deleted": documentId, "title": title})


@app.route("/api/resolve", methods=["POST"])
def resolveDuplicates():
    """Queue an entity-resolution sweep for one language. Inserts a 'resolve' ChatRequest row that
    the chatService poller routes to resolutionOperation; the frontend polls GET /api/chat/<id> and
    reads the JSON report from the 'answer' field. dryRun (default true) previews without merging."""
    payload = request.get_json(silent=True) or {}
    lang = (payload.get("lang") or os.environ.get("DEFAULT_LANG", "en")).lower()
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    dry = payload.get("dryRun") is not False   # preview unless explicitly told to apply
    exclude = payload.get("exclude") or []     # survivor names the user deselected (skip on apply)
    # the resolve params travel in the question field of the ChatRequest row (the poller reads it)
    params = json.dumps({"dryRun": dry, "exclude": exclude})[:1900]

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = iris.cls("GraphKB.ChatRequest")._New()
    row.question = params
    row.lang = lang
    row.status = "resolve"
    row.statusDetail = "Resolving duplicate entities"
    row.createdAt = now
    row.updatedAt = now
    if str(row._Save()) != "1":
        return jsonify({"error": "could not queue the resolution"}), 500
    return jsonify({"requestId": int(row._Id()), "status": "resolve", "dryRun": dry})


@app.route("/api/chat", methods=["POST"])
def createChat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    lang = (payload.get("lang") or os.environ.get("DEFAULT_LANG", "en")).lower()
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    if not question:
        return jsonify({"error": "question is required"}), 400

    # object API gives a reliable %ID (LAST_IDENTITY() returns empty via iris.sql.exec)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chatRow = iris.cls("GraphKB.ChatRequest")._New()
    chatRow.question = question[:1900]
    chatRow.lang = lang
    chatRow.status = "pending"
    chatRow.statusDetail = ""
    chatRow.createdAt = now
    chatRow.updatedAt = now
    if str(chatRow._Save()) != "1":
        return jsonify({"error": "could not create chat request"}), 500
    requestId = int(chatRow._Id())

    # the chatService polling adapter inside the production picks up this 'pending' row
    return jsonify({"requestId": requestId, "status": "pending"})


@app.route("/api/chat/<int:requestId>", methods=["GET"])
def getChat(requestId):
    for row in sqlExec(
            "SELECT question, lang, status, statusDetail, answer FROM GraphKB.ChatRequest WHERE %ID = ?",
            requestId):
        return jsonify({
            "requestId": requestId,
            "question": row[0],
            "lang": row[1],
            "status": row[2],
            "statusDetail": row[3] or "",
            "answer": row[4] or ""
        })
    return jsonify({"error": "not found"}), 404


@app.route("/api/stats", methods=["GET"])
def getStats():
    lang = langFilter()
    stats = {}
    if lang:
        # Chunk/Relationship have no lang column, so scope them through their parent Entity/Document.
        scoped = (
            ("SELECT COUNT(*) FROM GraphKB.Document WHERE lang = ?", "documents"),
            ("SELECT COUNT(*) FROM GraphKB.Chunk c JOIN GraphKB.Document d ON c.documentRef = d.%ID WHERE d.lang = ?", "chunks"),
            ("SELECT COUNT(*) FROM GraphKB.Entity WHERE lang = ?", "entities"),
            ("SELECT COUNT(*) FROM GraphKB.Relationship r JOIN GraphKB.Entity s ON r.sourceEntityRef = s.%ID WHERE s.lang = ?", "relationships"))
        for query, key in scoped:
            for row in sqlExec(query, lang):
                stats[key] = int(row[0])
    else:
        for table, key in (("GraphKB.Document", "documents"), ("GraphKB.Chunk", "chunks"),
                           ("GraphKB.Entity", "entities"), ("GraphKB.Relationship", "relationships")):
            for row in sqlExec("SELECT COUNT(*) FROM " + table):
                stats[key] = int(row[0])
    return jsonify(stats)


@app.route("/api/graph", methods=["GET"])
def getGraph():
    """The knowledge graph for the constellation view: entities as nodes (grouped by their source
    document), relationships as edges. Scoped to ?lang= so each language renders as its own graph."""
    lang = langFilter()

    nodeQuery = ("SELECT e.%ID, e.entityName, e.entityType, e.lang, e.documentRef, d.title "
                 "FROM GraphKB.Entity e LEFT JOIN GraphKB.Document d ON e.documentRef = d.%ID")
    nodeRows = sqlExec(nodeQuery + " WHERE e.lang = ?", lang) if lang else sqlExec(nodeQuery)
    nodes = []
    for row in nodeRows:
        doc = row[4]
        nodes.append({
            "id": int(row[0]), "name": row[1] or "", "type": row[2] or "Unknown", "lang": row[3] or "",
            "doc": int(doc) if doc not in (None, "") else None, "docTitle": row[5] or "—"})

    # Relationships have no lang column; scope them by their source entity's language (both ends of
    # a relationship are always same-language, since each is extracted from one chunk of one article).
    linkQuery = ("SELECT r.sourceEntityRef, r.targetEntityRef, r.predicate FROM GraphKB.Relationship r "
                 "JOIN GraphKB.Entity s ON r.sourceEntityRef = s.%ID "
                 "WHERE r.sourceEntityRef IS NOT NULL AND r.targetEntityRef IS NOT NULL")
    linkRows = sqlExec(linkQuery + " AND s.lang = ?", lang) if lang else sqlExec(linkQuery)
    links = []
    for row in linkRows:
        links.append({"source": int(row[0]), "target": int(row[1]), "predicate": row[2] or "relatedTo"})

    docQuery = "SELECT %ID, title, lang FROM GraphKB.Document"
    docRows = sqlExec(docQuery + " WHERE lang = ? ORDER BY title", lang) if lang \
        else sqlExec(docQuery + " ORDER BY title")
    documents = []
    for row in docRows:
        documents.append({"id": int(row[0]), "title": row[1], "lang": row[2]})

    return jsonify({"nodes": nodes, "links": links, "documents": documents})

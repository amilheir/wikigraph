"""WikiGraph - shared helpers and constants used by the component modules.

This module defines NO interoperability classes, so it is not compiled by pyprod; it is a plain
Python module imported at runtime by services / processes / operations (which sit in the same
directory, added to sys.path by each generated class's ScriptPath).
"""

import datetime
import json
import re

from intersystems_pyprod import IRISLog, Status

EMBEDDING_CONFIG = "wikiembed"


def statusIsOk(status):
    return str(status) == "1"


def logErrorAndReturnStatus(errorMessage):
    IRISLog.Error(errorMessage)
    return Status.ERROR(errorMessage), None


def sqlExec(query, *args):
    import iris
    return iris.sql.exec(query, *args)


def nowTimestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def insertRow(className, fields):
    """Insert a row via the object API and return the new %ID.

    Reliable and race-free, unlike SELECT LAST_IDENTITY() which returns empty through
    iris.sql.exec. EMBEDDING columns auto-populate on %Save() (an HTTP call to Ollama).
    """
    import iris
    obj = iris.cls(className)._New()
    for key, value in fields.items():
        setattr(obj, key, value)
    status = obj._Save()
    if str(status) != "1":
        raise RuntimeError("save failed for " + className + " (status " + str(status) + ")")
    return int(obj._Id())


def parseLlmJson(content):
    """Parse LLM JSON output, tolerating code fences and token-truncated (unterminated) output."""
    text = content.strip()
    fenceMatch = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenceMatch:
        text = fenceMatch.group(1).strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    try:
        return json.loads(text)
    except ValueError:
        # the LLM response was cut off by the token limit - salvage what completed
        return json.loads(repairTruncatedJson(text))


def repairTruncatedJson(text):
    """Best-effort close of a JSON object truncated mid-output (close open string, balance brackets)."""
    s = text.rstrip()
    if len(re.findall(r'(?<!\\)"', s)) % 2 == 1:
        s += '"'                       # close an unterminated string
    s = s.rstrip().rstrip(",")          # drop a dangling comma / partial element
    s += "]" * max(0, s.count("[") - s.count("]"))
    s += "}" * max(0, s.count("{") - s.count("}"))
    return s


def normalizeAliases(aliases, cap=8):
    """Clean a list of alias surface forms: lowercase, trim, dedup, drop too-short, cap the count."""
    out, seen = [], set()
    for alias in (aliases or []):
        if not isinstance(alias, str):
            continue
        cleaned = alias.strip().lower()[:190]
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= cap:
            break
    return out


def mergeAliasJson(existingJson, newAliases, cap=12):
    """Union newAliases into the existing JSON alias list. Returns the new JSON string, or None
    if nothing changed (so the caller can skip a no-op UPDATE)."""
    try:
        existing = json.loads(existingJson) if existingJson else []
        if not isinstance(existing, list):
            existing = []
    except ValueError:
        existing = []
    merged = [a for a in existing if isinstance(a, str)]
    seen = set(merged)
    changed = False
    for alias in newAliases:
        if alias not in seen:
            merged.append(alias)
            seen.add(alias)
            changed = True
    if not changed:
        return None
    return json.dumps(merged[:cap], ensure_ascii=False)


def aliasListFromJson(aliasJson):
    """Parse a stored alias JSON column into a list of strings (tolerant of NULL/garbage)."""
    try:
        parsed = json.loads(aliasJson) if aliasJson else []
        return [a for a in parsed if isinstance(a, str)] if isinstance(parsed, list) else []
    except ValueError:
        return []


def chunkArticleText(text, maxChars):
    """Split a Wikipedia plain-text extract into paragraph-aware chunks."""
    chunks = []
    current = ""
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 1 > maxChars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = (current + "\n" + paragraph) if current else paragraph
        while len(current) > maxChars:
            chunks.append(current[:maxChars])
            current = current[maxChars:]
    if current:
        chunks.append(current)
    return chunks


def updateChatStatus(requestId, status, statusDetail="", answer=None):
    if answer is None:
        sqlExec(
            "UPDATE GraphKB.ChatRequest SET status = ?, statusDetail = ?, updatedAt = CURRENT_TIMESTAMP WHERE %ID = ?",
            status, statusDetail[:480], int(requestId))
    else:
        sqlExec(
            "UPDATE GraphKB.ChatRequest SET status = ?, statusDetail = ?, answer = ?, updatedAt = CURRENT_TIMESTAMP WHERE %ID = ?",
            status, statusDetail[:480], answer[:29000], int(requestId))


def documentExists(pageId, lang):
    rows = sqlExec("SELECT COUNT(*) FROM GraphKB.Document WHERE pageId = ? AND lang = ?", int(pageId), lang)
    for row in rows:
        return int(row[0]) > 0
    return False

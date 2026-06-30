"""WikiGraph - shared helpers and constants used by the component modules.

This module defines NO interoperability classes, so it is not compiled by pyprod; it is a plain
Python module imported at runtime by services / processes / operations (which sit in the same
directory, added to sys.path by each generated class's ScriptPath).
"""

import datetime
import json
import re
import unicodedata

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


def sqlExecSafe(query, *args):
    """Run a DELETE/UPDATE that may affect zero rows without raising (iris.sql.exec raises SQLCODE 100
    when nothing matched). Use for statements whose row count is not guaranteed (e.g. repointing a
    loser entity that has no relationships during a merge)."""
    import iris
    try:
        iris.sql.exec(query, *args)
    except Exception:
        pass


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


def normalizeName(name):
    """Canonical comparison key for an entity name: lowercase, strip accents, drop punctuation,
    collapse whitespace. Keeps single spaces so callers can compare token sets for containment."""
    s = (name or "").strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalizeType(entityType):
    """Canonicalize a free-text entity type to a uniform form: take the first segment (before any
    '/', ',', ';' or '|'), trim, and Title-Case it - so 'LOCATION', 'location' and 'Location/Object'
    all become 'Location'. Type is descriptive metadata only (not identity), so this just keeps the
    'by type' grouping tidy. Empty -> 'Unknown'."""
    segment = re.split(r"[\/,;|]", (entityType or "").strip())[0].strip()
    if not segment:
        return "Unknown"
    return " ".join(word.capitalize() for word in segment.split())


def jaroWinkler(a, b):
    """Jaro-Winkler string similarity (0..1), pure Python so no extra dependency. Used by the
    entity-resolution pass to catch spelling variants the normalized-exact key misses."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    window = max(la, lb) // 2 - 1
    if window < 0:
        window = 0
    aMatched, bMatched = [False] * la, [False] * lb
    matches = 0
    for i in range(la):
        lo, hi = max(0, i - window), min(i + window + 1, lb)
        for j in range(lo, hi):
            if not bMatched[j] and a[i] == b[j]:
                aMatched[i] = bMatched[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    transpositions, k = 0, 0
    for i in range(la):
        if aMatched[i]:
            while not bMatched[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2
    jaro = (matches / la + matches / lb + (matches - transpositions) / matches) / 3
    prefix = 0
    for i in range(min(4, la, lb)):
        if a[i] == b[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def unionFind(pairs, items):
    """Group items into connected components given a list of (a, b) "same" pairs. Returns only the
    clusters with more than one member (the duplicate groups to merge)."""
    parent = {i: i for i in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    clusters = {}
    for i in items:
        clusters.setdefault(find(i), []).append(i)
    return [members for members in clusters.values() if len(members) > 1]


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

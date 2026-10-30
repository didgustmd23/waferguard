"""실험적 Graph RAG 지식 묶음(pack·지식·그래프·출처) 계약과 검증. 오류는 고정 코드로만 알린다.

지식 묶음은 추적되는 로컬 원본이며 모델에 보낼 참고자료다. 실행 지시·권한이 아니다.
- 파일 4개(pack·knowledge·graph·sources)를 고정 이름으로 한 폴더에서 읽는다. 링크·하드 링크·크기 초과·중복 키를 거부한다.
- 경로가 있는 출처는 검토 시점의 sha256을 기록해야 한다. 강제 출처(enforce)의 현재 파일이 다르면 묶음 전체를 쓰지 않는다.
  색인 구축이나 로딩이 이 값을 새로 계산해 덮어쓰지 않는다(지식 검토를 대신하지 않음).
- 지원 유형(task_kinds)·적용 단서(cues)는 프로필이 정한 값만 허용한다.
"""

import hashlib
import json
from pathlib import Path
import re

from . import strict_json
from .worker_paths import _path


PACK_SCHEMA = "rag_pack_v1"
KNOWLEDGE_SCHEMA = "rag_knowledge_v1"
GRAPH_SCHEMA = "rag_graph_v1"
SOURCES_SCHEMA = "rag_sources_v1"
FILES = {"pack": "pack.json", "knowledge": "knowledge.json", "graph": "graph.json", "sources": "sources.json"}
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_ITEMS, MAX_NODES, MAX_EDGES, MAX_SOURCES = 500, 2000, 4000, 200
MAX_TITLE, MAX_BODY, MAX_ENTRY, MAX_LIST = 120, 700, 300, 8
MAX_EMBED_CHARS = 1500
EMBED_TEXT_VERSION = "rag_embed_text_v1"
KINDS = {"product_constraint", "general", "case"}
PRODUCT_SUPPORT = {"supported", "unsupported", "constraint"}
SOURCE_KINDS = {"product_code", "product_doc", "official_doc", "computation"}
RELATION_TYPES = ("applies_to", "requires", "conflicts_with", "may_cause", "ask_question", "alternative_to")
NODE_TYPES = {"option", "split_strategy", "role", "condition", "effect", "question", "capability"}
CONTRADICTIONS = ({"requires", "conflicts_with"},)
REVIEW_METHODS = {"code", "doc", "docstring", "computation"}
REVIEW_STATUS = {"author_verified", "unreviewed"}
HUMAN_STATUS = {"pending", "approved"}
AI_STATUS = {"not_reviewed", "consistent", "fixed_after_review", "open_issue"}
ITEM_FIELDS = {"id", "title", "kind", "support", "body", "applies_when", "not_applies_when", "effects", "info_loss",
               "cautions", "options", "alternatives", "sources", "review", "task_kinds", "cues"}
EDGE_FIELDS = {"id", "source", "target", "type", "scope", "condition", "caution", "knowledge"}
NODE_FIELDS = {"id", "type", "label"}
NODE_OPTIONAL = {"paths", "support"}
SOURCE_FIELDS = {"id", "kind", "title", "path", "version", "sha256", "enforce"}
# 저장소 안 상대 경로 중 이 폴더만 출처로 허용한다. 개인 작업 폴더(_workspace 등)·실행 기록·데이터는 제품 출처가 될 수 없다.
SOURCE_ROOTS = ("src/", "docs/", "configs/", "scripts/")
# 지식 문장에 들어가면 안 되는 형태: 로컬 절대 경로·UNC·실행 기록 폴더·개인 작업 폴더·승인 토큰 형태의 16진수.
DRIVE_PATH = re.compile(r"[A-Za-z]:[\\/]")
FORBIDDEN_TEXT = (DRIVE_PATH, re.compile(r"(^|\s)/(Users|home|tmp)/"), re.compile(r"\\\\"),
                  re.compile(r"logs/agents"), re.compile(r"_workspace"), re.compile(r"\b[a-fA-F0-9]{32,}\b"))
# 공식 문서(official_doc) 출처의 title에만 허용하는 주소 형식: 소문자 https, 점이 있는 호스트, 경로·쿼리·절(#)의 제한 문자.
# 예외는 이 형식으로 확인한 주소의 "s:/" 한 곳(드라이브 경로 패턴 오인)뿐이다. 주소 안팎의 나머지 문자열은 모든 금지 패턴을 그대로 검사한다.
# 주소는 표시용 문자열이다. 접속·다운로드·HTML 렌더링을 하지 않고, 모델 요청에는 출처 ID만 보낸다.
# 주소는 문자열 처음·공백·여는 괄호 뒤에서만 시작한다(git+https:·view-source:https: 같은 다른 형식을 받지 않게).
OFFICIAL_URL = re.compile(r"(?<![^\s(])https://[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[A-Za-z0-9._~/-]*)?(?:\?[A-Za-z0-9._~=&+-]*)?"
                          r"(?:#[A-Za-z0-9._~-]*)?")


class KnowledgeError(ValueError):
    """고정 코드와 대상 ID만 담는다. 파일 원문·경로를 담지 않는다."""

    def __init__(self, code, targets=()):
        super().__init__(code)
        self.code, self.targets = code, list(targets)[:20]


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _read(path, limit):
    """링크·하드 링크가 아닌 일반 파일만 상한까지 읽는다."""
    try:
        path = _path(Path(path).absolute(), directory=False)
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
    except (OSError, ValueError, TypeError) as error:
        raise KnowledgeError("RAG_KNOWLEDGE_UNAVAILABLE") from error

    if len(raw) > limit:
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", ["size"])

    return raw


def _load_json(path):
    raw = _read(path, MAX_FILE_BYTES)

    try:
        return strict_json.loads(raw.decode("utf-8"), MAX_FILE_BYTES)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", ["json"]) from error


def _text(value, limit):
    return (type(value) is str and value.strip() != "" and len(value) <= limit
            and not any(ord(char) < 32 and char not in "\n\t" for char in value))


def _texts(value, *, allow_empty=True):
    return (type(value) is list and (allow_empty or value) and len(value) <= MAX_LIST
            and all(_text(item, MAX_ENTRY) for item in value))


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_strings(item)


def _forbidden_text(text, *, urls=False):
    schemes = {match.start() + len("http") for match in OFFICIAL_URL.finditer(text)} if urls else set()
    return any(not (pattern is DRIVE_PATH and match.start() in schemes)
               for pattern in FORBIDDEN_TEXT for match in pattern.finditer(text))


def _forbidden(value, *, urls=False):
    return any(_forbidden_text(text, urls=urls) for text in _all_strings(value))


def embed_text(item):
    """임베딩 입력. 본문만이 아니라 적용·비적용 조건·주의를 함께 넣는다(조건 없이 본문만 검색되지 않게)."""
    parts = [item["title"], item["body"], "적용 조건: " + " / ".join(item["applies_when"])]

    if item["not_applies_when"]:
        parts.append("비적용: " + " / ".join(item["not_applies_when"]))

    parts.append("주의: " + " / ".join(item["cautions"]))
    return "\n".join(parts)


# ==========================================
# 파일별 검사
# ==========================================

def validate_pack(pack, profile):
    if (type(pack) is not dict or set(pack) != {"schema_version", "pack_id", "version", "profile", "embedding"}
            or pack["schema_version"] != PACK_SCHEMA or pack["profile"] != profile.PROFILE_ID
            or type(pack["pack_id"]) is not str or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", pack["pack_id"]) is None
            or type(pack["version"]) is not str or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]{0,31}", pack["version"]) is None):
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", ["pack"])

    embedding = pack["embedding"]

    if (type(embedding) is not dict or set(embedding) != {"model", "digest", "dimensions"}
            or embedding["model"] != profile.EMBED_MODEL or embedding["dimensions"] != profile.DIMENSIONS
            or type(embedding["digest"]) is not str or re.fullmatch(r"[a-f0-9]{64}", embedding["digest"]) is None):
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", ["pack.embedding"])


def _source_path(value):
    """저장소 기준 상대 경로. 허용 폴더 아래의 슬래시 경로만 받고 상위 이동·드라이브·특수 이름을 거부한다."""
    return (type(value) is str and len(value) <= 200 and value.startswith(SOURCE_ROOTS)
            and re.fullmatch(r"[A-Za-z0-9_./-]+", value) is not None
            and all(part not in {"", ".", ".."} for part in value.split("/")))


def validate_sources(sources):
    if (type(sources) is not dict or set(sources) != {"schema_version", "sources"} or sources["schema_version"] != SOURCES_SCHEMA
            or type(sources["sources"]) is not list or not 0 < len(sources["sources"]) <= MAX_SOURCES):
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", ["sources"])

    errors, seen = [], set()

    for source in sources["sources"]:
        sid = source.get("id") if type(source) is dict else None

        if type(sid) is not str or re.fullmatch(r"S_[A-Z0-9_]{2,40}", sid) is None or sid in seen:
            errors.append(str(sid)[:48])
            continue

        seen.add(sid)

        if (set(source) != SOURCE_FIELDS or source["kind"] not in SOURCE_KINDS or type(source["enforce"]) is not bool
                or not _text(source["title"], MAX_ENTRY)):
            errors.append(sid)
        elif source["path"] is None:
            # 경로 없는 1차 자료(공식 문서 등)는 버전 문자열로 식별한다. 강제 대조 대상이 될 수 없다.
            if not _text(source["version"], 64) or source["sha256"] is not None or source["enforce"]:
                errors.append(sid)
        elif (not _source_path(source["path"]) or type(source["sha256"]) is not str
              or re.fullmatch(r"[a-f0-9]{64}", source["sha256"]) is None
              or (source["version"] is not None and not _text(source["version"], 64))):
            errors.append(sid)

    # path·sha256은 위 형식 검사로 대신한다. title의 https 주소는 공식 문서 출처에서만 허용한다(OFFICIAL_URL).
    if errors or any(_forbidden({key: value for key, value in item.items() if key not in ("path", "sha256", "title")})
                     or _forbidden(item["title"], urls=item["kind"] == "official_doc") for item in sources["sources"]):
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", errors or ["sources.text"])


def validate_knowledge(knowledge, sources, profile):
    if (type(knowledge) is not dict or set(knowledge) != {"schema_version", "items"} or knowledge["schema_version"] != KNOWLEDGE_SCHEMA
            or type(knowledge["items"]) is not list or not 0 < len(knowledge["items"]) <= MAX_ITEMS):
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", ["knowledge"])

    source_ids = {source["id"] for source in sources["sources"]}
    errors, seen = [], set()

    for item in knowledge["items"]:
        iid = item.get("id") if type(item) is dict else None

        if type(iid) is not str or re.fullmatch(r"K\d{2,4}", iid) is None or iid in seen or set(item) != ITEM_FIELDS:
            errors.append(str(iid)[:48])
            continue

        seen.add(iid)
        support_ok = (item["support"] in PRODUCT_SUPPORT if item["kind"] == "product_constraint" else item["support"] is None)
        review = item["review"]

        if (item["kind"] not in KINDS or not support_ok or not _text(item["title"], MAX_TITLE) or not _text(item["body"], MAX_BODY)
                or not _texts(item["applies_when"], allow_empty=False) or not _texts(item["cautions"], allow_empty=False)
                or not all(_texts(item[name]) for name in ("not_applies_when", "effects", "info_loss", "options", "alternatives"))
                or type(item["sources"]) is not list or not 0 < len(item["sources"]) <= MAX_LIST
                or any(type(ref) is not dict or set(ref) != {"id", "section"} or ref["id"] not in source_ids
                       or not _text(ref["section"], MAX_ENTRY) for ref in item["sources"])
                or type(item["task_kinds"]) is not list or not item["task_kinds"] or len(set(item["task_kinds"])) != len(item["task_kinds"])
                or not set(item["task_kinds"]) <= set(profile.TASK_KINDS)
                or type(item["cues"]) is not list or len(set(item["cues"])) != len(item["cues"]) or not set(item["cues"]) <= set(profile.CUES)
                or type(review) is not dict or not {"method", "status", "human"} <= set(review) <= {"method", "status", "human", "ai"}
                or review["method"] not in REVIEW_METHODS or review["status"] not in REVIEW_STATUS or review["human"] not in HUMAN_STATUS
                or review.get("ai", "not_reviewed") not in AI_STATUS
                or _forbidden(item) or len(embed_text(item)) > MAX_EMBED_CHARS):
            errors.append(iid)

    for item in knowledge["items"]:
        if type(item) is dict and type(item.get("alternatives")) is list and set(item["alternatives"]) - seen:
            errors.append(str(item.get("id"))[:48])

    if errors:
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", errors)


def validate_graph(graph, knowledge):
    if (type(graph) is not dict or set(graph) != {"schema_version", "relation_types", "nodes", "edges"}
            or graph["schema_version"] != GRAPH_SCHEMA or graph["relation_types"] != list(RELATION_TYPES)
            or type(graph["nodes"]) is not list or len(graph["nodes"]) > MAX_NODES
            or type(graph["edges"]) is not list or len(graph["edges"]) > MAX_EDGES):
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", ["graph"])

    items = {item["id"]: item for item in knowledge["items"]}
    errors, nodes, edge_ids, pairs = [], set(), set(), {}

    for node in graph["nodes"]:
        nid = node.get("id") if type(node) is dict else None

        if (type(nid) is not str or re.fullmatch(r"N_[A-Z0-9_]{2,60}", nid) is None or nid in nodes
                or not NODE_FIELDS <= set(node) <= NODE_FIELDS | NODE_OPTIONAL or node["type"] not in NODE_TYPES
                or not _text(node["label"], MAX_TITLE) or node.get("support", "supported") not in {"supported", "unsupported"}
                or ("paths" in node and not _texts(node["paths"])) or _forbidden(node)):
            errors.append(str(nid)[:48])
            continue

        nodes.add(nid)

    for edge in graph["edges"]:
        eid = edge.get("id") if type(edge) is dict else None

        if type(eid) is not str or re.fullmatch(r"E\d{2,5}", eid) is None or eid in edge_ids or set(edge) != EDGE_FIELDS:
            errors.append(str(eid)[:48])
            continue

        edge_ids.add(eid)
        refs = edge["knowledge"]
        # 제품 조합 제한은 제품 제약 근거가, 일반 원칙은 일반 근거가 하나 이상 있어야 한다(범위 혼동 방지).
        wanted = "product_constraint" if edge["scope"] == "product" else "general"

        if (edge["type"] not in RELATION_TYPES or edge["source"] not in nodes or edge["target"] not in nodes
                or edge["source"] == edge["target"] or edge["scope"] not in {"general", "product"}
                or not _text(edge["condition"], MAX_ENTRY) or type(edge["caution"]) is not str or len(edge["caution"]) > MAX_ENTRY
                or not _texts(refs, allow_empty=False) or any(ref not in items for ref in refs)
                or not any(items[ref]["kind"] == wanted for ref in refs) or _forbidden(edge)):
            errors.append(eid)
            continue

        pairs.setdefault(frozenset((edge["source"], edge["target"])), []).append((edge["type"], eid))

    for found in pairs.values():
        if any(conflict <= {kind for kind, _ in found} for conflict in CONTRADICTIONS):
            errors.extend(eid for _, eid in found)

    if errors:
        raise KnowledgeError("RAG_KNOWLEDGE_INVALID", errors)


def check_sources(sources, repo_root):
    """경로 출처의 현재 해시를 검토 시점 값과 대조한다. 강제 출처가 다르면 거부, 비강제 출처는 변경 ID만 돌려준다."""
    changed = []

    for source in sources["sources"]:
        if source["path"] is None:
            continue

        try:
            current = hashlib.sha256(_read(Path(repo_root) / source["path"], MAX_SOURCE_BYTES)).hexdigest()
        except KnowledgeError:
            current = None

        if current != source["sha256"]:
            if source["enforce"]:
                raise KnowledgeError("RAG_SOURCE_CHANGED", [source["id"]])
            changed.append(source["id"])

    return changed


def load_pack(directory, profile, repo_root):
    """지식 묶음을 읽고 모두 검증한다. 하나라도 실패하면 고정 코드로 거부한다(부분 사용 없음)."""
    directory = Path(directory)
    values = {name: _load_json(directory / filename) for name, filename in FILES.items()}
    validate_pack(values["pack"], profile)
    validate_sources(values["sources"])
    validate_knowledge(values["knowledge"], values["sources"], profile)
    validate_graph(values["graph"], values["knowledge"])
    changed = check_sources(values["sources"], repo_root)
    return dict(values, sha256=canonical_sha256({name: values[name] for name in FILES}), changed_sources=changed)

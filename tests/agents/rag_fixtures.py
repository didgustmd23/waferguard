"""실험적 Graph RAG 테스트용 합성 지식 묶음·모의 임베더. 자료 담당의 검토 자료(_workspace)를 읽지 않는다."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from src.agents import rag_index, rag_knowledge, rag_tabular


DIGEST = "a" * 64
DIMENSIONS = rag_tabular.DIMENSIONS
SOURCE_PATH = "src/agents/synthetic_rule.py"
SOURCE_TEXT = b"# synthetic product rule\n"
DOC_PATH = "docs/agents/synthetic.md"
DOC_TEXT = b"synthetic doc\n"


def unit(*weights):
    """앞쪽 축에 가중치를 둔 정규화 벡터."""
    vector = np.zeros(DIMENSIONS)
    vector[:len(weights)] = weights
    return list(vector / np.linalg.norm(vector))


def item(iid, *, kind="general", support=None, kinds=("tabular_csv", "tabular_v2"), cues=(), title=None, body=None):
    return {"id": iid, "title": title or f"합성 지식 {iid}", "kind": kind, "support": support,
            "body": body or f"{iid} 합성 본문입니다. 실제 제품 지식이 아닙니다.", "applies_when": [f"{iid} 적용 조건"],
            "not_applies_when": [f"{iid} 비적용 조건"], "effects": [], "info_loss": [], "cautions": [f"{iid} 주의 문장"],
            "options": [], "alternatives": [], "sources": [{"id": "S_RULE", "section": "합성 절"}],
            "review": {"method": "code", "status": "author_verified", "human": "pending", "ai": "not_reviewed"},
            "task_kinds": list(kinds), "cues": list(cues)}


def pack_values():
    items = [item("K01", kind="product_constraint", support="constraint"),
             item("K02", cues=["outliers"]),
             item("K03", kind="product_constraint", support="unsupported", kinds=("tabular_v2",)),
             item("K04", kind="case", cues=["few_values"])]
    nodes = [{"id": "N_OPT_A", "type": "option", "label": "옵션 A"}, {"id": "N_ROLE_B", "type": "role", "label": "역할 B"},
             {"id": "N_COND_C", "type": "condition", "label": "조건 C"},
             {"id": "N_EFFECT_D", "type": "effect", "label": "효과 D", "support": "unsupported"}]

    def edge(eid, source, target, kind, scope, refs):
        return {"id": eid, "source": source, "target": target, "type": kind, "scope": scope, "condition": f"{eid} 관계 조건",
                "caution": f"{eid} 관계 주의", "knowledge": refs}

    # A→B→C→A 순환과 v2 전용 근거를 둔 관계(E04)를 포함한다.
    edges = [edge("E01", "N_OPT_A", "N_ROLE_B", "applies_to", "product", ["K01"]),
             edge("E02", "N_ROLE_B", "N_COND_C", "may_cause", "general", ["K02"]),
             edge("E03", "N_COND_C", "N_OPT_A", "alternative_to", "general", ["K02"]),
             edge("E04", "N_OPT_A", "N_EFFECT_D", "requires", "product", ["K03"])]
    return {
        "pack": {"schema_version": rag_knowledge.PACK_SCHEMA, "pack_id": "synthetic", "version": "0.1",
                 "profile": rag_tabular.PROFILE_ID,
                 "embedding": {"model": rag_tabular.EMBED_MODEL, "digest": DIGEST, "dimensions": DIMENSIONS}},
        "knowledge": {"schema_version": rag_knowledge.KNOWLEDGE_SCHEMA, "items": items},
        "graph": {"schema_version": rag_knowledge.GRAPH_SCHEMA, "relation_types": list(rag_knowledge.RELATION_TYPES),
                  "nodes": nodes, "edges": edges},
        "sources": {"schema_version": rag_knowledge.SOURCES_SCHEMA, "sources": [
            {"id": "S_RULE", "kind": "product_code", "title": "합성 규칙", "path": SOURCE_PATH, "version": None,
             "sha256": hashlib.sha256(SOURCE_TEXT).hexdigest(), "enforce": True},
            {"id": "S_DOC", "kind": "product_doc", "title": "합성 문서", "path": DOC_PATH, "version": None,
             "sha256": hashlib.sha256(DOC_TEXT).hexdigest(), "enforce": False}]},
    }


def write_repo(root):
    root = Path(root)
    (root / SOURCE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / SOURCE_PATH).write_bytes(SOURCE_TEXT)
    (root / DOC_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / DOC_PATH).write_bytes(DOC_TEXT)


def write_pack(directory, values=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for name, filename in rag_knowledge.FILES.items():
        (directory / filename).write_text(json.dumps((values or pack_values())[name], ensure_ascii=False), encoding="utf-8")


ITEM_AXES = {"K01": 0, "K02": 1, "K03": 2, "K04": 3}


class FakeEmbedder:
    """문서 임베딩은 항목별 축 벡터, 질의는 지정한 벡터를 돌려준다. 호출 입력·횟수를 기록한다."""

    def __init__(self, query=None, error=None):
        self.query, self.error, self.texts = query, error, []

    def embed(self, text, *, model, digest, dimensions):
        self.texts.append(text)

        if self.error is not None:
            raise self.error

        for iid, axis in ITEM_AXES.items():
            if text.startswith(f"합성 지식 {iid}\n"):
                return unit(*([0.0] * axis + [1.0]))

        return list(self.query)


class Profile:
    """임시 폴더를 가리키는 표형 프로필. 질의 구성·단서는 제품 rag_tabular를 그대로 쓴다."""

    def __init__(self, root):
        root = Path(root)
        self.REPO_ROOT, self.PACK_DIR, self.INDEX_ROOT = root, root / "configs/agents/rag/tabular", root / "models/rag/tabular"

    def __getattr__(self, name):
        return getattr(rag_tabular, name)


def prepared(root, values=None):
    """합성 저장소·지식 묶음·색인을 만들고 (프로필, 지식 묶음)을 돌려준다."""
    profile = Profile(root)
    write_repo(root)
    write_pack(profile.PACK_DIR, values)
    pack = rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
    rag_index.build_index(profile.INDEX_ROOT, pack, FakeEmbedder())
    return profile, pack


def changed(values, path, value):
    """pack_values의 사본에서 path(키·인덱스 목록)의 값을 바꾼다."""
    values = copy.deepcopy(values)
    target = values

    for key in path[:-1]:
        target = target[key]

    target[path[-1]] = value
    return values

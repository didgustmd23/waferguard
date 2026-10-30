"""실험적 Graph RAG 벡터 색인 검증·구축. 색인은 지식 묶음에서 다시 만들 수 있는 생성물이다.

- 색인 폴더 이름과 manifest는 지식 묶음 해시·임베딩 모델 식별값에 결합한다. 하나라도 다르면 열지 않는다(조용한 재사용 없음).
- vectors.npy는 allow_pickle=False로 읽고 형태·dtype·유한값·정규화·파일 해시·항목 순서를 검사한다.
- 구축은 별도 관리 명령(rag_index_cli)에서만 한다. 작업 시작·앱 시작·검색 실패 때 자동으로 만들지 않는다.
  임시 폴더에 다 쓴 뒤 이름 변경으로 공개하며, 기존 색인을 덮어쓰지 않는다. 실패한 임시 폴더는 정상 색인 이름을 갖지 않는다.
- 이전 색인 벡터는 같은 모델·같은 임베딩 입력(해시)일 때만 재사용한다.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import numpy as np

from . import strict_json
from .rag_knowledge import EMBED_TEXT_VERSION, canonical_sha256, embed_text
from .artifacts import _is_link
from .worker_paths import _path


INDEX_SCHEMA = "rag_index_v1"
VECTORS, MANIFEST = "vectors.npy", "manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_VECTOR_BYTES = 4 * 1024 * 1024
MANIFEST_FIELDS = {"schema_version", "pack_id", "pack_version", "pack_sha256", "embed_text_version", "model", "count",
                   "dimensions", "dtype", "normalized", "vectors_sha256", "items"}


class RagIndexError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def index_name(pack):
    """지식 묶음·모델 식별값으로 정해지는 색인 폴더 이름. 활성 색인 포인터 파일을 두지 않는다."""
    return f"{pack['pack']['pack_id']}-{pack['sha256'][:12]}-{pack['pack']['embedding']['digest'][:12]}"


def _model(pack):
    embedding = pack["pack"]["embedding"]
    return {"name": embedding["model"], "digest": embedding["digest"], "dimensions": embedding["dimensions"]}


def _read(path, limit, code):
    try:
        path = _path(Path(path).absolute(), directory=False)
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
    except (OSError, ValueError, TypeError) as error:
        raise RagIndexError(code) from error

    if len(raw) > limit:
        raise RagIndexError("RAG_INDEX_INVALID")

    return raw


def _matrix(raw, count, dimensions):
    """npy 바이트를 pickle 없이 읽고 형태·dtype·유한값·정규화를 검사한다.

    np.load는 헤더의 shape만큼 먼저 할당하므로, 읽기 전에 헤더(형태·dtype·저장 순서)와 전체 길이를 기대값과 대조한다.
    파일 크기 상한만으로는 메모리 상한이 되지 않는다.
    """
    try:
        stream = io.BytesIO(raw)
        version = np.lib.format.read_magic(stream)

        if version not in ((1, 0), (2, 0)):
            raise ValueError("npy 버전")

        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream, max_header_size=1024)

        if (shape != (count, dimensions) or fortran or dtype != np.dtype("<f4")
                or len(raw) != stream.tell() + count * dimensions * 4):
            raise ValueError("npy 헤더")

        matrix = np.load(io.BytesIO(raw), allow_pickle=False)
    except (ValueError, OSError, EOFError, TypeError) as error:
        raise RagIndexError("RAG_INDEX_INVALID") from error

    if (type(matrix) is not np.ndarray or matrix.dtype != np.float32 or matrix.shape != (count, dimensions)
            or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)):
        raise RagIndexError("RAG_INDEX_INVALID")

    return matrix


def _manifest(directory):
    raw = _read(Path(directory) / MANIFEST, MAX_MANIFEST_BYTES, "RAG_INDEX_UNAVAILABLE")

    try:
        manifest = strict_json.loads(raw.decode("utf-8"), MAX_MANIFEST_BYTES)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise RagIndexError("RAG_INDEX_INVALID") from error

    if (set(manifest) != MANIFEST_FIELDS or manifest["schema_version"] != INDEX_SCHEMA or type(manifest["items"]) is not list
            or type(manifest["count"]) is not int or manifest["count"] != len(manifest["items"])
            or manifest["dtype"] != "float32" or manifest["normalized"] is not True
            or any(type(entry) is not dict or set(entry) != {"id", "text_sha256"} for entry in manifest["items"])):
        raise RagIndexError("RAG_INDEX_INVALID")

    return manifest


def _vectors(directory, manifest):
    raw = _read(Path(directory) / VECTORS, MAX_VECTOR_BYTES, "RAG_INDEX_INVALID")

    if hashlib.sha256(raw).hexdigest() != manifest["vectors_sha256"]:
        raise RagIndexError("RAG_INDEX_INVALID")

    return _matrix(raw, manifest["count"], manifest["dimensions"])


def load_index(root, pack):
    """현재 지식 묶음·모델과 맞는 색인만 연다. 맞지 않으면 고정 코드로 거부한다."""
    directory = Path(root) / index_name(pack)
    manifest = _manifest(directory)

    if manifest["model"] != _model(pack):
        raise RagIndexError("RAG_INDEX_MODEL_MISMATCH")

    items = pack["knowledge"]["items"]

    if (manifest["pack_sha256"] != pack["sha256"] or manifest["pack_id"] != pack["pack"]["pack_id"]
            or manifest["pack_version"] != pack["pack"]["version"] or manifest["embed_text_version"] != EMBED_TEXT_VERSION
            or manifest["items"] != [{"id": item["id"], "text_sha256": canonical_sha256(embed_text(item))} for item in items]):
        raise RagIndexError("RAG_INDEX_MISMATCH")

    matrix = _vectors(directory, manifest)
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return {"name": directory.name, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "ids": [item["id"] for item in items],
            "matrix": matrix}


def _previous(directory, model):
    """재사용할 이전 색인. 무결성(해시·형태·정규화)과 같은 모델을 확인한 경우에만 text_sha256 → 벡터를 돌려준다."""
    manifest = _manifest(directory)

    if manifest["model"] != model or manifest["embed_text_version"] != EMBED_TEXT_VERSION:
        return {}

    matrix = _vectors(directory, manifest)
    return {entry["text_sha256"]: matrix[index] for index, entry in enumerate(manifest["items"])}


def build_index(root, pack, embedder, *, previous=None):
    """명시적 관리 작업. 새 색인을 임시 폴더에 만든 뒤 고정 이름으로 공개한다. 이미 있으면 덮어쓰지 않는다.

    embedder.embed(text, model=, digest=, dimensions=)는 문서 1개마다 1회 호출하며 실패하면 즉시 중단한다(재시도 없음).
    """
    root = Path(root)
    name = index_name(pack)
    final = root / name
    model = _model(pack)

    if final.exists():
        raise RagIndexError("RAG_INDEX_EXISTS")

    # 쓰기 전에 색인 루트와 상위 경로의 링크·재분석 지점을 거부한다(링크 대상에 색인을 쓰지 않게).
    for component in (root.absolute(), *root.absolute().parents):
        if component.exists() and _is_link(component):
            raise RagIndexError("RAG_INDEX_PATH_INVALID")

    reuse = {} if previous is None else _previous(root / previous, model)
    rows, entries, embedded = [], [], 0

    for item in pack["knowledge"]["items"]:
        text = embed_text(item)
        digest = canonical_sha256(text)

        if digest in reuse:
            vector = np.asarray(reuse[digest], dtype=np.float64)
        else:
            vector = np.asarray(embedder.embed(text, model=model["name"], digest=model["digest"],
                                               dimensions=model["dimensions"]), dtype=np.float64)
            embedded += 1

        norm = float(np.linalg.norm(vector))

        if vector.shape != (model["dimensions"],) or not np.isfinite(vector).all() or norm < 1e-6:
            raise RagIndexError("RAG_INDEX_INVALID")

        rows.append((vector / norm).astype(np.float32))
        entries.append({"id": item["id"], "text_sha256": digest})

    buffer = io.BytesIO()
    np.save(buffer, np.vstack(rows), allow_pickle=False)
    raw = buffer.getvalue()
    manifest = {"schema_version": INDEX_SCHEMA, "pack_id": pack["pack"]["pack_id"], "pack_version": pack["pack"]["version"],
                "pack_sha256": pack["sha256"], "embed_text_version": EMBED_TEXT_VERSION, "model": model, "count": len(entries),
                "dimensions": model["dimensions"], "dtype": "float32", "normalized": True,
                "vectors_sha256": hashlib.sha256(raw).hexdigest(), "items": entries}
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ("." + name + ".partial-" + uuid4().hex)
    temporary.mkdir()

    for filename, payload in ((VECTORS, raw), (MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))):
        with (temporary / filename).open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    # 다 쓴 폴더만 고정 이름으로 바꾼다. 대상이 이미 있으면 os.rename이 실패해 덮어쓰지 않는다.
    os.rename(temporary, final)
    return {"name": name, "count": len(entries), "embedded": embedded, "reused": len(entries) - embedded}


def is_index_name(value):
    return type(value) is str and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}-[a-f0-9]{12}-[a-f0-9]{12}", value) is not None

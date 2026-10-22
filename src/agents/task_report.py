"""통합 결과 보고서 조립. 유형 어댑터가 검증된 결과에서 추출한 사실만 공통 절 구성으로 배치한다.

LLM을 쓰지 않는다. 사실에 없는 수치는 추측하지 않고 '미집계'로 둔다. 경로·해시·내부 작업명은 기술 상세에만 둔다.
"""

import json

from . import task_adapters, task_recommend


MISSING = "미집계"


def code_label(record, code):
    """권장안 거부·모델 오류 코드의 고정 설명. 등록되지 않은 코드는 설명하지 않는다."""
    labels = {**task_recommend.CODE_LABELS, **task_adapters.adapter(record["kind"]).CODE_LABELS}
    return labels.get(code)


def build_facts(record):
    """어댑터 결과 요약에 공통 항목(목표·권장안 사용 여부·실행 연결)을 더한다."""
    adapter = task_adapters.adapter(record["kind"])
    # 저장 후 다시 읽은 기록은 키가 정렬된다. 같은 결과면 같은 보고서가 되도록 단계 정의 순서로 맞춘다.
    order = [step for step, _, _ in adapter.STEPS if step in record["steps"]]
    record = dict(record, steps={step: record["steps"][step] for step in order + sorted(set(record["steps"]) - set(order))})
    facts = adapter.report_facts(record)
    draft = record["draft"]
    cautions = list(facts["cautions"])

    if draft["source"] != "model":
        cautions.append("로컬 LLM 권장안을 쓰지 못해 기본 계획에서 시작했습니다: "
                        + (code_label(record, draft.get("code")) or "사유 코드 확인 불가") + ".")

    coverage = draft.get("coverage") or {}

    if coverage.get("summarized"):
        cautions.append(f"모델 검토 범위: 판단 대상 {coverage['total']}개 중 {coverage['summarized']}개는 요약만 제공해 모델이 "
                        "역할을 권하지 않았습니다(기본 계획 또는 사용자 편집 값 사용).")

    for item in (record["confirmed"] or {}).get("answers", []):
        cautions.append("질문 답변으로 정한 항목: " + str(item["path"] or item["id"]) + " = " + str(item["value"]))

    technical = dict(facts["technical"], runs={step: item["run_name"] for step, item in record["steps"].items()},
                     delegations={step: len(item["delegations"]) for step, item in record["steps"].items()},
                     consents={key: value["consent_id"] for key, value in record["consents"].items()},
                     draft={"source": draft["source"], "code": draft.get("code"),
                            "coverage": {key: coverage[key] for key in ("total", "detailed", "named", "summarized") if key in coverage}})
    # 유형 어댑터가 새 절(핵심 결과·경고·활용)을 주지 않아도 공통 보고서 틀은 같은 절을 비워 둔다.
    return dict(facts, goal=record["goal"], cautions=cautions, technical=technical,
                highlights=facts.get("highlights", []), warnings=facts.get("warnings", []), usage=facts.get("usage", []),
                recommendation={"source": draft["source"], "code": draft.get("code")})


def _plain(value):
    """입력 유래 문자열(컬럼명·모델 문장)을 Markdown·HTML로 해석되지 않는 한 줄 글자로 바꾼다."""
    import html

    # 백슬래시를 먼저 바꿔야 뒤에 붙이는 이스케이프가 원래 글자의 백슬래시와 합쳐지지 않는다.
    text = html.escape(str(value).replace("\\", "\\\\")).replace("\r", " ").replace("\n", " ").replace("|", "/")

    for character in "`*_[]#!":
        text = text.replace(character, "\\" + character)

    return text


def _table(rows):
    if not rows:
        return ["(없음)"]

    keys = list(rows[0])
    return ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |",
            *["| " + " | ".join(_plain(row[key]) for key in keys) + " |" for row in rows]]


TECHNICAL_HEAD = "## 기술 상세\n<details><summary>경로·해시·내부 작업 (펼치기)</summary>\n\n```json\n"
TECHNICAL_TAIL = "\n```\n\n</details>"
WARNING_KINDS = ("정보 손실", "미처리", "미검증")


def render_markdown(facts):
    """공통 절 구성: 핵심 결과·주요 문제 → 컬럼별 처리 → 실제 전후 비교 → 경고 → 활용 → 접힌 기술 상세.

    유형 어댑터가 준 사실만 배치한다. 이전 형식 사실(highlights·warnings·usage 없음)은 해당 줄을 비워 둔다.
    입력 유래 문자열은 _plain으로 한 줄 글자로 바꾼다. 절 제목만 고정 Markdown이다.
    """
    lines = ["# " + facts["title"], "", "목표: " + _plain(facts["goal"]), "",
             "## 핵심 결과와 주요 문제", *["- " + _plain(text) for text in facts.get("highlights", [])], "",
             _plain(" · ".join(f"{item['label']}: {item['value']}" for item in facts["metrics"])), "",
             *[_plain(text) for text in facts.get("lines", [])], "",
             "### 주요 문제", *_table(facts["problems"]), "",
             "## 컬럼별 처리와 제외 사유", *_table(facts["processing"]), "",
             "## 실제 전후 비교", *([_plain(facts["comparison_note"]), ""] if facts.get("comparison_note") else []), *_table(facts["comparison"]), "",
             "## 그래프", *(["- " + _plain(item["label"]) for item in facts["images"]] or ["(검증된 그래프 없음)"]), ""]
    warnings = facts.get("warnings", [])
    lines += ["## 경고: 정보 손실·미처리·미검증"]

    for kind in WARNING_KINDS:
        items = [item["내용"] for item in warnings if item["구분"] == kind]
        lines += ["### " + kind, *(["- " + _plain(text) for text in items] or ["- (없음)"])]

    lines += ["", "## 참고 사항", *["- " + _plain(text) for text in facts["cautions"]], "",
              "## 적용 구간과 결과 파일 활용", *["- " + _plain(text) for text in facts.get("usage", [])], "",
              "### " + facts.get("files_title", "결과 파일"), *_table(facts["files"]), ""]
    return "\n".join(lines) + "\n" + TECHNICAL_HEAD + json.dumps(facts["technical"], ensure_ascii=False, indent=2) + TECHNICAL_TAIL + "\n"


def _same_report(existing, text):
    """다시 조립한 보고서와 같은 내용인지. 바이트가 같거나, 기술 상세 JSON 구역 밖은 바이트가 같고 그 JSON이 목록·키
    순서만 다를 때 참이다(이전 버전의 단계 순서 차이). 다른 구역으로 옮겨진 줄은 같은 것으로 보지 않는다."""
    if existing == text:
        return True

    if existing.count(TECHNICAL_HEAD) != 1 or text.count(TECHNICAL_HEAD) != 1:
        return False

    head, body = existing.split(TECHNICAL_HEAD)
    new_head, new_body = text.split(TECHNICAL_HEAD)

    if head != new_head or body != body.rstrip("\n") + "\n" or new_body != new_body.rstrip("\n") + "\n":
        return False

    body, new_body = body.rstrip("\n"), new_body.rstrip("\n")

    if not body.endswith(TECHNICAL_TAIL) or not new_body.endswith(TECHNICAL_TAIL):
        return False

    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}

        if isinstance(value, list):
            return sorted((canonical(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))

        return value

    try:
        old, new = (json.loads(part[:-len(TECHNICAL_TAIL)]) for part in (body, new_body))
    except ValueError:
        return False

    return canonical(old) == canonical(new)


def write_report(record, directory, *, reuse_identical=False):
    """검증된 결과로 보고서를 새 파일에만 쓴다. 기존 보고서는 덮어쓰지 않는다.

    reuse_identical이면 같은 시도의 기존 보고서 중 다시 조립한 내용과 같은 파일(_same_report)을 그대로 참조한다
    (완료 기록 복구용). 기록의 본문·해시는 그 파일 기준이다. 같은 파일이 없으면 새 파일을 만든다(덮어쓰기 없음).
    """
    import hashlib

    import os

    facts = build_facts(record)
    text = render_markdown(facts)
    # 기존 보고서와 같은 줄 끝(텍스트 모드의 플랫폼 줄 끝)으로 쓴다. 해시는 실제 파일 바이트 기준이다.
    payload = text.replace("\n", os.linesep).encode("utf-8")
    folder = directory / "report"
    folder.mkdir(exist_ok=True)
    index = 1
    path = None

    while (folder / f"report_{record['attempt']}_{index}.md").exists():
        existing = folder / f"report_{record['attempt']}_{index}.md"

        if reuse_identical and path is None and not existing.is_symlink():
            data = existing.read_bytes()
            # 텍스트 모드로 읽은 것과 같게 줄 끝만 맞춰 비교한다.
            content = data.decode("utf-8").replace("\r\n", "\n")

            if data == payload or _same_report(content, text):
                path, payload, text = existing, data, content

        index += 1

    if path is None:
        path = folder / f"report_{record['attempt']}_{index}.md"

        with path.open("xb") as stream:
            stream.write(payload)

    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "text": text,
            "bundle": facts.get("outputs", {}).get("bundle"), "facts": facts}


def verified_images(record):
    """화면 표시용 그래프. 저장된 경로를 쓰지 않고 매번 기록 검사와 결과 검증을 거친 PNG만 돌려준다."""
    from .task_orchestrator import _check_names

    _check_names(record)
    adapter = task_adapters.adapter(record["kind"])
    adapter.check_lineage(record)
    return adapter.images(record)


def rebuild_report(task_id, *, root=None):
    """완료 작업의 보고서를 검증된 결과로 다시 조립한다. 계산·worker·승인·모델 호출은 없다."""
    from . import task_store
    from .task_orchestrator import _check_names

    with task_store.task_lock(task_id, root=root) as directory:
        record = task_store.load_task(task_id, root=root)

        if record["phase"] != "completed":
            raise ValueError("완료된 작업만 보고서를 다시 조립할 수 있습니다.")

        _check_names(record)
        record["report"] = write_report(record, directory)
        return task_store.save_task(record, expected_revision=record["revision"], root=root)

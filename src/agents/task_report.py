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


SECTION_NAMES = {"split": "분할"}


def recommendation_notice(record):
    """기본 계획을 쓴 사유 안내(화면·보고서 공용). 모델 권장안을 쓴 기록은 빈 목록이다.

    계획 검증으로 권장안을 거부한 기록은 전체 거부·변경 미적용과 확인된 하위 원인·검사한 계약 값만 고정 문구로 보인다.
    하위 진단이 생기는 코드인데 진단이 없는 이전 기록은 상세 원인 미확인으로 둔다. 역할 등 버려진 변경을 추정하지 않는다.
    저장된 진단은 표시 직전에 유형 어댑터의 허용 명세(rejection_valid)로 다시 확인한다. 명세가 없는 유형은 해석하지 않는다.
    """
    draft = record["draft"]

    if draft["source"] == "model":
        return []

    code = draft.get("code")
    label = code_label(record, code) or "사유 코드 확인 불가"
    rejection = draft.get("rejection")
    adapter = task_adapters.adapter(record["kind"])
    detailed = getattr(adapter, "REJECTION_DETAIL_CODES", ())
    valid = getattr(adapter, "rejection_valid", None)

    if rejection is None and code not in detailed:
        return ["로컬 LLM 권장안을 쓰지 못해 기본 계획에서 시작했습니다: " + label + "."]

    lines = ["로컬 LLM 권장안 전체를 거부해 기본 계획에서 시작했습니다(권장안의 변경은 적용하지 않았습니다): " + label + "."]

    if rejection is None:
        return lines + ["세부 원인: 확인되지 않음(이 기록에는 하위 진단이 없습니다)"]

    if valid is None or not valid(code, rejection):
        # 손상·변조된 진단은 해석하지 않는다(값을 지우거나 바꿔 정상처럼 보이지 않음). 화면·보고서 조립은 멈추지 않는다.
        return lines + ["세부 원인: 확인되지 않음(하위 진단을 읽지 못했습니다)"]

    lines.append("세부 원인: " + (code_label(record, rejection["reason"]) or "분류되지 않음"))
    section = SECTION_NAMES.get(rejection["section"], rejection["section"])
    values = " · ".join(f"{key}={value}" for key, value in rejection["values"].items())
    changed = ", ".join(rejection["from_model"]) or "없음"
    lines.append(f"거부한 {section} 값(병합 결과): {values} (모델이 바꾼 항목: {changed})")

    if rejection["omitted"]:
        lines.append("검사를 통과하지 못해 기록하지 않은 항목: " + ", ".join(rejection["omitted"]))

    return lines


def build_facts(record):
    """어댑터 결과 요약에 공통 항목(목표·권장안 사용 여부·실행 연결)을 더한다."""
    adapter = task_adapters.adapter(record["kind"])
    # 저장 후 다시 읽은 기록은 키가 정렬된다. 같은 결과면 같은 보고서가 되도록 단계 정의 순서로 맞춘다.
    order = [step for step, _, _ in adapter.STEPS if step in record["steps"]]
    record = dict(record, steps={step: record["steps"][step] for step in order + sorted(set(record["steps"]) - set(order))})
    facts = adapter.report_facts(record)
    draft = record["draft"]
    cautions = list(facts["cautions"])

    cautions += recommendation_notice(record)

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
    # 따옴표는 태그 밖 본문에서 해석되지 않으므로 그대로 둔다(quote=True면 &#x27;의 #가 아래에서 다시 escape되어 표시가 깨진다).
    # <, >, &는 계속 바꿔 HTML로 해석되지 않게 한다.
    text = html.escape(str(value).replace("\\", "\\\\"), quote=False).replace("\r", " ").replace("\n", " ").replace("|", "/")

    for character in "`*_[]#!":
        text = text.replace(character, "\\" + character)

    return text


def _table(rows):
    if not rows:
        return ["(없음)"]

    keys = list(rows[0])
    return ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |",
            *["| " + " | ".join(_plain(row[key]) for key in keys) + " |" for row in rows]]


BAR_WIDTH = 20


def _bars(chart):
    """그래프 값 행에 계열별 텍스트 막대 열을 더한다. 막대 길이는 그래프 안 최댓값 기준이며 값 자체는 바꾸지 않는다.
    0보다 큰 값은 최소 한 칸이라 작은 클래스·구간도 0처럼 보이지 않는다."""
    peak = max((row[key] for row in chart["rows"] for key in chart["values"]), default=0)
    return [dict(row, **{"막대 " + key: "█" * max(1, round(BAR_WIDTH * row[key] / peak)) if row[key] > 0 else ""
                         for key in chart["values"]}) for row in chart["rows"]]


def chart_rows(chart):
    """그래프 값 표의 표시용 행. 사실이 자릿수(digits)를 정한 열만 고정 소수로 쓰고, 정하지 않은 이전 사실은 그대로다.
    화면 표와 Markdown 표가 같이 쓴다. 그래프에 그리는 값은 바꾸지 않는다."""
    digits = chart.get("digits") or {}
    return [{key: f"{value:.{digits[key]}f}" if key in digits else value for key, value in row.items()} for row in chart["rows"]]


TECHNICAL_HEAD ="## 기술 상세\n<details><summary>경로·해시·내부 작업 (펼치기)</summary>\n\n```json\n"
TECHNICAL_TAIL = "\n```\n\n</details>"
WARNING_KINDS = ("정보 손실", "미처리", "미검증")


def render_markdown(facts):
    """공통 절 구성: 핵심 결과·주요 문제 → (처리 전 EDA·계획과 결정·분할, 유형이 줄 때) → 컬럼별 처리 → 실제 전후 비교 →
    그래프(보고서 그래프 값 표, 유형이 줄 때 → 검증된 PNG 이름) → 경고 → 활용 → 접힌 기술 상세.

    유형 어댑터가 준 사실만 배치한다. 이전 형식 사실(highlights·warnings·usage 없음)은 해당 줄을 비워 둔다.
    입력 유래 문자열은 _plain으로 한 줄 글자로 바꾼다. 절 제목만 고정 Markdown이다.
    """
    lines = ["# " + facts["title"], "", "목표: " + _plain(facts["goal"]), "",
             "## 핵심 결과와 주요 문제", *["- " + _plain(text) for text in facts.get("highlights", [])], "",
             _plain(" · ".join(f"{item['label']}: {item['value']}" for item in facts["metrics"])), "",
             *[_plain(text) for text in facts.get("lines", [])], "",
             "### 주요 문제", *_table(facts["problems"]), ""]
    # 처리 전 EDA·계획과 결정 절은 이를 주는 유형 어댑터만 채운다. 표마다 제목·대상·선정 안내(notes)·행을 둔다.
    for key, heading, note in (("eda_tables", "## 처리 전 데이터와 기본 EDA", facts.get("eda_note")),
                               ("decision_tables", "## 계획과 결정", None)):
        if facts.get(key):
            lines += [heading, *([_plain(note), ""] if note else [])]

            for table in facts[key]:
                lines += ["### " + _plain(table["title"]), *["- " + _plain(text) for text in table["notes"]], "", *_table(table["rows"]), ""]

    # 분할 표·그래프 설명은 이를 주는 유형 어댑터만 채운다. 없는 이전 형식·다른 유형 사실은 기존 줄 그대로 둔다.
    split = facts.get("split_table")

    if split:
        lines += ["## 분할 결과", _plain(split["title"]), "", *_table(split["rows"]), ""]

    notes = facts.get("image_notes") or {}
    lines += ["## 컬럼별 처리와 제외 사유", *_table(facts["processing"]), "",
              "## 실제 전후 비교", *([_plain(facts["comparison_note"]), ""] if facts.get("comparison_note") else []), *_table(facts["comparison"]), "",
              "## 그래프"]
    # 보고서 그래프(charts)는 이를 주는 유형만 채운다. 화면 그래프와 같은 값을 표와 텍스트 막대로 남긴다.
    for chart in facts.get("charts") or []:
        lines += ["### " + _plain(chart["title"]), *["- " + _plain(text) for text in chart["notes"]], ""]

        if chart["rows"]:
            lines += ["- 막대: 이 그래프의 최댓값을 " + str(BAR_WIDTH) + "칸으로 한 텍스트 막대입니다(화면 그래프와 같은 값).", "",
                      *_table(chart_rows(dict(chart, rows=_bars(chart)))), ""]

    if facts.get("charts"):
        lines += ["### 분할 EDA 그래프 (검증된 PNG, 화면에서 표시)"]

    lines += [*(["- " + _plain(item["label"]) + (" — " + _plain(notes[item["label"]]) if item["label"] in notes else "")
                 for item in facts["images"]] or ["(검증된 그래프 없음)"]), ""]
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

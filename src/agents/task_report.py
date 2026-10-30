"""통합 결과 보고서 조립. 유형 어댑터가 검증된 결과에서 추출한 사실만 공통 절 구성으로 배치한다.

LLM을 쓰지 않는다. 사실에 없는 수치는 추측하지 않고 '미집계'로 둔다. 경로·해시·내부 작업명은 기술 상세에만 둔다.
"""

import json

from . import task_adapters, task_rag, task_recommend
from .task_plan import PlanError, get_path


MISSING = "미집계"
MODEL_LIMITATION_NOTICE = (
    "현재 로컬 모델의 한계: 확인한 사례에서 진단 수치 해석, 라벨·그룹 구분, 전처리 권장 이유에 오류가 관찰됐습니다. "
    "응답 형식 검사를 통과해도 내용의 타당성을 보장하지 않습니다. 계산 결과와 모델 의견을 구분해 확인하세요. "
    "이 한계를 모든 모델에 일반화하지 않으며, 다른 로컬·API 모델의 품질은 별도 평가가 필요합니다."
)


# 화면·보고서 표시 문구. task_recommend는 RAG 고정 출처이자 코드 지문 대상이라 그 라벨을 바꾸지 않고 표시만 정확히 한다.
DISPLAY_LABELS = {
    "RECOMMENDATION_CHANGE_INVALID": "모델 변경 항목이 권장안 계약에 맞지 않았습니다"
                                     "(항목 형식 오류, 허용되지 않은 경로·값, 같은 경로 중복 중 하나이며 하위 원인은 기록되지 않았습니다)",
}
# 모델 응답을 받았지만 권장안 전체를 쓰지 않은 경우의 안내. 모델 응답 원문·값은 담지 않는다.
REJECTED_GUIDANCE = "모델 변경 0건을 적용했고 자동으로 다시 요청하지 않습니다. 아래 질문과 계획 양식에서 직접 정할 수 있습니다."


def code_label(record, code):
    """권장안 거부·모델 오류 코드의 고정 설명. 등록되지 않은 코드는 설명하지 않는다."""
    labels = {**task_recommend.CODE_LABELS, **task_rag.REQUEST_CODE_LABELS, **task_adapters.adapter(record["kind"]).CODE_LABELS,
              **DISPLAY_LABELS}
    return labels.get(code)


def code_text(record, code):
    """고정 설명과 코드를 함께 쓴다. 등록되지 않은 값은 기록에서 그대로 옮기지 않는다."""
    label = code_label(record, code)
    return "사유 코드 확인 불가" if label is None else f"{label} ({code})"


def recommendation_guidance(record):
    """권장안 거부 뒤 처리 결과와 다음 행동. 응답 형식·내용 거부에만 있고 연결 실패 등에는 없다."""
    return [REJECTED_GUIDANCE] if record["draft"].get("code") in task_recommend.REJECTION_CODES else []


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
    label = code_text(record, code)
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

    cautions += answer_notes(record)

    technical = dict(facts["technical"], runs={step: item["run_name"] for step, item in record["steps"].items()},
                     delegations={step: len(item["delegations"]) for step, item in record["steps"].items()},
                     consents={key: value["consent_id"] for key, value in record["consents"].items()},
                     draft={"source": draft["source"], "code": draft.get("code"),
                            "coverage": {key: coverage[key] for key in ("total", "detailed", "named", "summarized") if key in coverage}})
    # 검색 근거(실험적 Graph RAG) 사용 기록은 이를 남긴 작업에만 절을 둔다. 저장된 기록만 쓰고 검색을 다시 하지 않는다.
    rag = task_rag.view(draft)
    extra = {} if rag is None else {"rag": rag}

    if rag is not None:
        usage = draft["rag"] if type(draft.get("rag")) is dict else {}
        technical["draft"]["rag"] = {key: usage.get(key) for key in ("mode", "status", "code", "pack", "index", "embedding",
                                                                    "included_items", "included_relations")}

    # 유형 어댑터가 새 절(핵심 결과·경고·활용)을 주지 않아도 공통 보고서 틀은 같은 절을 비워 둔다.
    return dict(facts, goal=record["goal"], cautions=cautions, technical=technical,
                highlights=facts.get("highlights", []), warnings=facts.get("warnings", []), usage=facts.get("usage", []),
                recommendation={"source": draft["source"], "code": draft.get("code")}, **extra)


def answer_notes(record):
    """질문 답변 한 줄씩. 응답값(고른 선택지)·계획 경로·확정 계획 값을 구분한다.

    컬럼 선택 질문은 응답값이 컬럼 이름이고 선택지가 정한 역할은 확정 계획에 있다(확정 계획은 답변을 마지막에 반영한 결과).
    응답값이 곧 그 항목 값이면 기존 표기를 쓴다. 확정 계획에 경로가 없는 이전 기록은 값을 추정하지 않는다.
    """
    confirmed = record.get("confirmed") or {}
    lines = []

    for item in confirmed.get("answers", []):
        path, value = item.get("path"), item["value"]

        if path is None:
            lines.append("질문 답변(질문 " + str(item["id"]) + "): 고른 선택지 " + str(value) + " · 바꾼 계획 항목 없음")
            continue

        try:
            final = get_path(confirmed["plan"], path)
        except (KeyError, TypeError, PlanError):
            lines.append("질문 답변 항목: " + str(path) + " · 확정 계획 값 확인되지 않음 (고른 선택지: " + str(value) + ")")
            continue

        if final == value:
            lines.append("질문 답변으로 정한 항목: " + str(path) + " = " + str(value))
        else:
            lines.append("질문 답변으로 정한 항목: " + str(path) + " = " + str(final) + " (확정 계획 값, 고른 선택지: " + str(value) + ")")

    return lines


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


# 보고서 표의 열 순서. 저장한 사실을 다시 읽으면 행 키가 가나다순으로 정렬되므로 화면·Markdown 모두 이 순서로 열을 놓는다.
# 기존 표 머리글(유형 어댑터의 조립 순서)을 모두 그대로 재현하는 하나의 순서다. 목록에 없는 열은 처음 나온 순서로 뒤에 둔다.
COLUMN_ORDER = (
    "구분", "순위", "항목", "설정 이름", "최종 값", "파일", "용도", "컬럼", "특징", "특징 A", "특징 B", "구간", "대상 구간", "대상",
    "컬럼 범위", "단위", "역할", "자료형", "결측(건)", "결측 비율", "관측 고유값", "상수·전체 결측", "공백 문자열", "+무한대", "-무한대",
    "유한 관측 수", "하한", "상한", "후보 수", "후보 비율", "관측 수", "평균", "표준편차", "최솟값", "25%", "중앙값", "75%", "최댓값",
    "범주 수", "빈 표기(건)", "최빈 범주 점유율", "희소 범주 수", "희소 범주 행 비율", "Pearson", "Spearman", "쌍별 관측 수",
    # 결정 D 화면의 예상 처리 영향·출력 폭 표(저장된 후보 사실을 다시 읽어 그린다).
    "변환 전 고유값", "clipping 변경 값", "구간 수", "입력 특징", "출력 열", "변환 후 상수", "정보 손실",
    "요청 비율", "확정 비율", "행 수", "그룹 수", "클래스", "건수", "비율", "변환", "값", "전", "후", "후 (실측)", "예상 (후보)",
    "처리 전", "처리 후", "결정 출처", "모델 의견", "실행 기록 대조", "실제 변경(근거)", "내용", "처리", "제외 사유", "대상·분모",
    "근거", "비고", "크기(byte)",
)


def table_columns(rows, columns=None):
    """표의 열 이름. 화면 표와 Markdown 표가 같이 쓴다.

    columns(표를 만드는 쪽이 정한 순서)가 있으면 그 순서를 먼저 쓰고, 목록에 없는 데이터 열은 사라지지 않게 뒤에 둔다.
    없으면(이전 facts 등) 행마다 키가 달라도 모든 키를 모아 호환용 COLUMN_ORDER 순으로 둔다.
    """
    seen = list(dict.fromkeys(key for row in rows for key in row))
    rank = {name: index for index, name in enumerate(COLUMN_ORDER)}

    if columns:
        return list(dict.fromkeys(columns)) + sorted((key for key in seen if key not in columns),
                                                     key=lambda key: (rank.get(key, len(rank)), seen.index(key)))

    return sorted(seen, key=lambda key: (rank.get(key, len(rank)), seen.index(key)))


def _table(rows, columns=None):
    if not rows:
        return ["(없음)"]

    keys = table_columns(rows, columns)
    # 일부 행에 없는 열은 빈 칸으로 둔다(값을 만들지 않는다).
    return ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |",
            *["| " + " | ".join(_plain(row.get(key, "")) for key in keys) + " |" for row in rows]]


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


def _figure_lines(facts, label):
    """저장 그림의 상대 경로 링크. 저장하지 못했으면 숨기지 않고 사유를 적는다. 그림 결과가 없는 facts는 줄이 없다."""
    figures = facts.get("figures") or {}
    item = next((item for item in figures.get("items", []) if item["label"] == label), None)

    if item is None:
        return []

    if item["status"] == "saved":
        return [f"![{_plain(label)}]({figures['directory']}/{item['file']})", ""]

    return ["- 그림 저장 실패: " + _plain(label) + " (" + _plain(item["reason"]) + ")", ""]


def _chart_lines(chart, facts):
    lines = ["### " + _plain(chart["title"]), *["- " + _plain(text) for text in chart["notes"]], "", *_figure_lines(facts, chart["title"])]

    if chart["rows"]:
        lines += ["- 막대: 이 그래프의 최댓값을 " + str(BAR_WIDTH) + "칸으로 한 텍스트 막대입니다(화면 그래프와 같은 값).", "",
                  *_table(chart_rows(dict(chart, rows=_bars(chart)))), ""]

    return lines


def _section_charts(facts, section):
    return [chart for chart in facts.get("charts") or [] if chart.get("section") == section]


def _image_lines(facts):
    notes = facts.get("image_notes") or {}
    lines = []

    for item in facts["images"]:
        figure_lines = _figure_lines(facts, item["label"])
        lines += ["- " + _plain(item["label"]) + (" — " + _plain(notes[item["label"]]) if item["label"] in notes else ""),
                  *([""] if figure_lines else []), *figure_lines]

    return (lines or ["(검증된 그래프 없음)"]) + [""]


def render_markdown(facts):
    """공통 절 구성: 핵심 결과·주요 문제 → (처리 전 EDA·계획과 결정·분할, 유형이 줄 때) → 컬럼별 처리 → 실제 전후 비교 →
    그래프 → 경고 → 참고 → 활용 → 접힌 기술 상세. 그래프에 절(section)을 준 유형은 그래프 절 없이 해당 절에 두고,
    모델 의견(model_opinions)을 준 유형은 계산 사실과 분리한 절을 참고 사항 뒤에 둔다.

    유형 어댑터가 준 사실만 배치한다. 이전 형식 사실(highlights·warnings·usage 없음)은 해당 줄을 비워 둔다.
    입력 유래 문자열은 _plain으로 한 줄 글자로 바꾼다. 절 제목만 고정 Markdown이다.
    """
    lines = ["# " + facts["title"], "", "목표: " + _plain(facts["goal"]), "",
             "## 핵심 결과와 주요 문제", *["- " + _plain(text) for text in facts.get("highlights", [])], "",
             _plain(" · ".join(f"{item['label']}: {item['value']}" for item in facts["metrics"])), "",
             *[_plain(text) for text in facts.get("lines", [])], "",
             "### 주요 문제", *_table(facts["problems"]), ""]
    placed = any("section" in chart for chart in facts.get("charts") or [])
    # 처리 전 EDA·계획과 결정 절은 이를 주는 유형 어댑터만 채운다. 표마다 제목·대상·선정 안내(notes)·행을 둔다.
    for key, heading, note in (("eda_tables", "## 처리 전 데이터와 기본 EDA", facts.get("eda_note")),
                               ("decision_tables", "## 계획과 결정", None)):
        if facts.get(key):
            lines += [heading, *([_plain(note), ""] if note else [])]

            for table in facts[key]:
                lines += ["### " + _plain(table["title"]), *["- " + _plain(text) for text in table["notes"]], "", *_table(table["rows"], table.get("columns")), ""]

            if placed and key == "eda_tables":
                for chart in _section_charts(facts, "eda"):
                    lines += _chart_lines(chart, facts)

                lines += ["### 분할 EDA 그래프 (검증된 PNG)", *_image_lines(facts)]

    # 분할 표·그래프 설명은 이를 주는 유형 어댑터만 채운다. 없는 이전 형식·다른 유형 사실은 기존 줄 그대로 둔다.
    split = facts.get("split_table")

    if split:
        lines += ["## 분할 결과", _plain(split["title"]), "", *_table(split["rows"]), ""]

        for chart in _section_charts(facts, "split"):
            lines += _chart_lines(chart, facts)

    lines += ["## 컬럼별 처리와 제외 사유", *_table(facts["processing"]), "",
              "## 실제 전후 비교", *([_plain(facts["comparison_note"]), ""] if facts.get("comparison_note") else []), *_table(facts["comparison"]), ""]

    if not placed:
        lines += ["## 그래프"]
        # 보고서 그래프(charts)는 이를 주는 유형만 채운다. 화면 그래프와 같은 값을 표와 텍스트 막대로 남긴다.
        for chart in facts.get("charts") or []:
            lines += _chart_lines(chart, facts)

        if facts.get("charts"):
            lines += ["### 분할 EDA 그래프 (검증된 PNG, 화면에서 표시)"]

        lines += _image_lines(facts)

    warnings = facts.get("warnings", [])
    lines += ["## 경고: 정보 손실·미처리·미검증"]

    for kind in WARNING_KINDS:
        items = [item["내용"] for item in warnings if item["구분"] == kind]
        lines += ["### " + kind, *(["- " + _plain(text) for text in items] or ["- (없음)"])]

    rag = facts.get("rag")

    if rag:
        lines += ["", "## 검색 근거 (실험적 Graph RAG)", *["- " + _plain(text) for text in rag["lines"]]]

        if rag["items"] or rag["relations"]:
            lines += ["", "### 생성 요청에 포함한 지식", *_table(rag["items"], task_rag.ITEM_COLUMNS), "",
                      "### 생성 요청에 포함한 관계", *_table(rag["relations"], task_rag.RELATION_COLUMNS)]

    lines += ["", "## 참고 사항", *["- " + _plain(text) for text in facts["cautions"]], ""]
    opinions = facts.get("model_opinions")

    if opinions:
        lines += ["## 모델 의견과 한계 (계산 사실 아님)", "- " + _plain(MODEL_LIMITATION_NOTICE),
                  *["- " + _plain(text) for text in opinions["notes"]], "", *_table(opinions["rows"], opinions["columns"]), ""]

    lines += ["## 적용 구간과 결과 파일 활용", *["- " + _plain(text) for text in facts.get("usage", [])], "",
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
    folder = directory / "report"
    folder.mkdir(exist_ok=True)

    if any("section" in chart for chart in facts.get("charts") or []):
        from .report_figures import save_figures

        try:
            images = task_adapters.adapter(record["kind"]).images(record)
        except Exception:
            # 못 읽은 그림은 저장 결과에 실패로 남아 화면·보고서가 알린다. 전처리 완료 자체는 막지 않는다.
            images = []

        facts = save_figures(facts, images, folder, record["attempt"])

    text = render_markdown(facts)
    # 기존 보고서와 같은 줄 끝(텍스트 모드의 플랫폼 줄 끝)으로 쓴다. 해시는 실제 파일 바이트 기준이다.
    payload = text.replace("\n", os.linesep).encode("utf-8")
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

    # 승인을 발급하지 않는 재조립이라 보호 이전 작업도 기존처럼 허용한다(작업 폴더 보유는 유지).
    with task_store.task_lock(task_id, root=root, protected=False) as directory:
        record = task_store.load_task(task_id, root=root)

        if record["phase"] != "completed":
            raise ValueError("완료된 작업만 보고서를 다시 조립할 수 있습니다.")

        _check_names(record)
        record["report"] = write_report(record, directory)
        return task_store.save_task(record, expected_revision=record["revision"], root=root)

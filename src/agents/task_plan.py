"""작업 계획(결정 항목 값)의 경로 단위 변경·질문 답변 반영·필수 질문 판정. 데이터 형태·도메인을 모른다.

계획은 {구획: {항목: 값}} 형태이고 항목은 "구획.항목" 경로로 가리킨다. 허용 경로·값은 유형 어댑터의 항목 명세가 정한다.
이 모듈은 값 형식만 확인하고, 의미 검증(역할 조합·분할 조건 등)은 어댑터의 validate_plan이 한다.
"""

import copy
import hashlib
import json
import math
import re


MAX_TEXT = 300
MAX_QUESTIONS = 12
LEVELS = ("required", "advisory")
QUESTION_ID = r"[a-z][a-z0-9_.:-]{0,79}"


class PlanError(ValueError):
    """계획 검증 실패. 메시지는 고정 문구이고 입력 자료 값은 subjects로만 분리한다."""

    def __init__(self, code, message, *, subjects=()):
        super().__init__(code)
        self.code, self.message, self.subjects = code, message, tuple(subjects)


class TaskProblem(Exception):
    """사용자에게 보여 줄 고정 코드와 다음 행동이 있는 정지. 원문 예외·경로는 담지 않는다.

    diagnostic은 서비스가 검증한 worker 분류(확인된 사실), hint는 어댑터가 계획에서 고른 추정 원인 코드다.
    actions는 사용자가 차단 화면에서 고를 수 있는 어댑터의 고정 해결 동작 ID, detail은 정수 개수만 담는다.
    """

    def __init__(self, code, message, *, retryable=False, diagnostic=None, hint=None, actions=(), detail=None, error=None,
                 findings=()):
        super().__init__(code)
        self.error = error
        # 검증된 결과에서 확인한 지적(고정 코드·대상 이름). 대상 이름은 화면에서 일반 텍스트·표로만 보인다.
        self.findings = [{"code": str(item["code"]), "subject": None if item.get("subject") is None else str(item["subject"])}
                         for item in findings]
        self.code, self.message, self.retryable = code, message, retryable
        self.diagnostic, self.hint, self.actions = diagnostic, hint, list(actions)
        self.detail = {key: value for key, value in (detail or {}).items() if type(value) is int}


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _split(path):
    if type(path) is not str or path.count(".") < 1:
        raise PlanError("PLAN_PATH_INVALID", "계획 항목 경로 형식이 올바르지 않습니다.")

    section, key = path.split(".", 1)
    return section, key


def get_path(plan, path):
    section, key = _split(path)
    return plan[section][key]


def set_path(plan, path, value):
    """한 경로만 바꾼 새 계획을 돌려준다. 다른 항목을 연쇄로 바꾸지 않는다."""
    section, key = _split(path)

    if section not in plan or key not in plan[section]:
        raise PlanError("PLAN_PATH_INVALID", "계획에 없는 항목입니다.")

    updated = copy.deepcopy(plan)
    updated[section][key] = value
    return updated


def value_allowed(field, value):
    """항목 명세(type·options·범위)에 맞는 값인지 확인한다."""
    kind = field["type"]

    if value is None:
        return field.get("nullable", False)

    if kind == "enum":
        return type(value) in (str, int, float, bool) and value in field["options"]

    if kind == "boolean":
        return type(value) is bool

    if kind in ("integer", "number"):
        if type(value) is bool or type(value) not in ((int,) if kind == "integer" else (int, float)):
            return False

        if not math.isfinite(value):
            return False

        return field.get("minimum", -math.inf) <= value <= field.get("maximum", math.inf)

    return False


def fields_by_path(fields):
    return {field["path"]: field for field in fields}


def flatten(plan):
    """전체 유효 설정 표시용 (경로, 값) 목록."""
    return [(section + "." + key, value) for section in sorted(plan) for key, value in sorted(plan[section].items())]


def changes_between(base, plan):
    """기본 계획과 다른 항목. 표시·기록용이다."""
    return [{"path": path, "before": value, "after": get_path(plan, path)}
            for path, value in flatten(base) if get_path(plan, path) != value]


def _text(value):
    return (type(value) is str and value.strip() and len(value) <= MAX_TEXT
            and not any(ord(char) < 32 for char in value))


# ==========================================
# 질문 계약 (설계 7.5절)
# - 필수 질문은 어댑터만 만든다. 모델 질문은 항상 advisory이며 스키마 밖 경로는 버린다.
# - 답변 하나는 해당 path 하나만 바꾼다. 권장 선택지도 사용자가 고르기 전에는 반영하지 않는다.
# - 필수 여부는 답변을 반영한 최종 계획에서 어댑터 질문을 다시 계산해 판정한다.
# ==========================================

def question(qid, level, text, *, path=None, options=(), recommended=None, subjects=(), source="adapter"):
    """어댑터가 쓰는 질문 생성 함수. 형식이 맞지 않으면 즉시 거부한다.

    선택지는 {value, label}이고, 질문에 path가 없을 때만 선택지별로 {path, set}을 더해 그 선택지가 바꿀 항목 하나를 정할 수 있다.
    """
    item = {"id": qid, "source": source, "level": level, "path": path, "text": text,
            "options": [dict(option) for option in options], "recommended": recommended,
            "subjects": [str(subject) for subject in subjects]}

    if (type(qid) is not str or re.fullmatch(QUESTION_ID, qid) is None or level not in LEVELS
            or source not in ("adapter", "model") or not _text(text)
            or (path is not None and type(path) is not str)
            or any(set(option) not in ({"value", "label"}, {"value", "label", "path", "set"}) or not _text(option["label"])
                   or ("path" in option and (path is not None or type(option["path"]) is not str))
                   for option in item["options"])
            or len({str(option["value"]) for option in item["options"]}) != len(item["options"])):
        raise ValueError("질문 형식이 올바르지 않습니다.")

    if recommended is not None and recommended not in [option["value"] for option in item["options"]]:
        raise ValueError("권장 선택지는 선택지 중 하나여야 합니다.")

    return item


def model_questions(items, fields):
    """모델 질문을 안내용으로만 받아들인다. 형식·경로·선택지가 맞지 않는 질문은 버리고 개수만 돌려준다."""
    by_path = fields_by_path(fields)
    accepted, dropped = [], 0

    for index, item in enumerate(items[:5]):
        try:
            path = item.get("path")
            field = None if path is None else by_path[path]
            values = item.get("options") or []

            if type(values) is not list or len(values) > 10 or (field is None and values):
                raise ValueError

            if field is not None and any(not value_allowed(field, value) for value in values):
                raise ValueError

            accepted.append(question("model:" + str(index), "advisory", item["text"], path=path, source="model",
                                     options=[{"value": value, "label": str(value)} for value in values]))
        except (KeyError, TypeError, ValueError, AttributeError):
            dropped += 1

    return accepted, dropped


def apply_answers(plan, questions, answers, fields):
    """질문 ID별 답변을 해당 경로에만 반영한다. 질문에 없는 ID·선택지 밖 값·경로 없는 질문의 값은 거부한다."""
    if type(answers) is not dict:
        raise PlanError("ANSWERS_INVALID", "질문 답변 형식이 올바르지 않습니다.")

    known = {item["id"]: item for item in questions}
    by_path = fields_by_path(fields)
    applied, chosen = [], {}

    for qid, value in sorted(answers.items()):
        item = known.get(qid)
        option = None if item is None else next((entry for entry in item["options"] if entry["value"] == value), None)

        if option is None:
            raise PlanError("ANSWERS_INVALID", "표시하지 않은 질문이나 선택지 밖의 답변입니다.")

        # 답변 하나는 질문의 path(값=답) 또는 고른 선택지의 path(값=set) 한 항목만 바꾼다.
        path, target = (item["path"], value) if item["path"] is not None else (option.get("path"), option.get("set"))

        if path is not None:
            if path not in by_path or not value_allowed(by_path[path], target):
                raise PlanError("ANSWERS_INVALID", "답변 값이 계획 항목 조건에 맞지 않습니다.")

            if chosen.setdefault(path, target) != target:
                raise PlanError("ANSWERS_CONFLICT", "같은 항목에 서로 다른 답변을 골랐습니다.")

            plan = set_path(plan, path, target)

        applied.append({"id": qid, "path": path, "value": value})

    return plan, applied


def answer_targets(questions, answers):
    """답변이 바꿀 (경로, 값). 질문에 없는 ID·선택지 밖 값은 apply_answers가 거부하므로 여기서는 건너뛴다."""
    known = {item["id"]: item for item in questions}
    targets = []

    for qid, value in sorted((answers or {}).items()):
        item = known.get(qid)
        option = None if item is None else next((entry for entry in item["options"] if entry["value"] == value), None)

        if option is None:
            continue

        path, target = (item["path"], value) if item["path"] is not None else (option.get("path"), option.get("set"))

        if path is not None:
            targets.append({"id": qid, "path": path, "value": target})

    return targets


def answer_conflicts(shown, edited, questions, answers):
    """사용자가 양식에서 직접 바꾼 항목과 질문 답변이 같은 항목에 다른 값을 정한 경우. 어느 쪽도 자동으로 고르지 않는다."""
    conflicts = []

    for item in answer_targets(questions, answers):
        try:
            before, form = get_path(shown, item["path"]), get_path(edited, item["path"])
        except (KeyError, PlanError):
            continue

        if form != before and form != item["value"]:
            conflicts.append({"id": item["id"], "path": item["path"], "form": form, "answer": item["value"]})

    return conflicts


def origins(base, recommended, shown, edited, applied):
    """최종 계획 항목별 출처: 질문 답변 > 양식에서 사용자 변경 > 모델 권장 > 표시 계획(재시도면 직전 확정) > 기본 계획.

    recommended는 모델 권장 변경 목록({path, value, reason})이다. 이유는 모델 의견이며 계산 사실과 구분해 표시한다.
    """
    reasons = {item["path"]: item.get("reason") for item in recommended}
    answered = {item["path"] for item in applied if item["path"] is not None}
    result = {}

    for path, value in flatten(edited):
        if path in answered:
            result[path] = {"source": "answer"}
        elif value != get_path(shown, path):
            result[path] = {"source": "user"}
        elif path in reasons and value == get_path(shown, path) and value != get_path(base, path):
            result[path] = {"source": "model", "reason": reasons[path]}
        elif value != get_path(base, path):
            result[path] = {"source": "previous"}
        else:
            result[path] = {"source": "base"}

    return result


def unanswered_required(questions, applied=()):
    """최종 계획에서 다시 계산한 필수 질문 중 해결되지 않은 것.

    계획 항목을 바꾼 답변은 최종 계획에서 조건이 사라져야 해결된다. 항목을 바꾸지 않는 확인형 답변(질문 path도
    고른 선택지 path도 없음, 예: 예측 대상 없음)만 답한 것으로 해결한다.
    """
    confirmed = {item["id"] for item in applied if item["path"] is None}
    return [item for item in questions if item["level"] == "required" and item["id"] not in confirmed]

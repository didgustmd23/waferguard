"""배열 표본 묶음(.npz) 작업 어댑터. 공통 조정기가 부르는 진단·계획·질문·단계 생성·결과 요약을 배열 서비스에 연결한다.

계산·격리·결과 검증은 array_task_service와 array_task_results가 한다. 이 모듈은 검증된 결과에서 사실만 읽고
없는 값은 '미집계'로 둔다. 웨이퍼 전용 규칙과 시간 처리는 다루지 않는다.
설계 기준: docs/agents/DESIGN.md "배열·이미지 통합 작업 연결"
"""

import json
import math
from pathlib import Path

from src.preprocessing.array_dataset import ArrayInputError, CHANNEL_AXES, check_mapping, meta_candidates, sample_candidates
from src.preprocessing.split import SplitRules
from . import array_options
from .array_task_service import code_id, service
from .array_task_results import _preview_png
from .isolated_eda import _read_file
from .task_plan import PlanError, TaskProblem, fields_by_path, question, signature, value_allowed


KIND = "array_npz"
INPUT = {"label": "배열 표본 파일(.npz)", "picker": "npz_file",
         "help": "원본은 변경하지 않고 작업용 복사본만 사용합니다. pickle 없이 숫자·문자열 배열만 읽습니다. 경로는 모델에 보내지 않습니다. "
                 "파일·압축 해제 크기 상한 안이어도 계산 중 메모리는 더 필요할 수 있으며, 작업 메모리 상한을 넘으면 중단됩니다."}
MISSING = "미집계"
TERMINAL = {"completed", "blocked_input", "failed_runtime", "security_locked", "cancelled"}
STEPS = (("prepare", "array_prepare", "C"), ("apply", "array_apply", "D"))
RUN_ALIAS = {}
REVIEW_STEP = "prepare"
STEP_LABELS = {"prepare": "분할·Train 기준 후보", "apply": "적용·결과 묶음"}
TRANSFER_SCOPE = "goal_array_names_shapes_and_aggregates_without_values_or_label_names"
ACTIONS = {}
FINDING_LABELS = {}
LABEL_LIMIT = "모델에는 라벨 클래스 이름 대신 순번(class_1…)과 개수만 보냈습니다. 모델은 클래스 의미를 해석하지 못하며 권장안은 분포만 근거로 합니다."
CODE_LABELS = {
    "PLAN_FIELDS_INVALID": "매핑·분할·전처리 항목 구성이 계약과 다릅니다",
    "MAPPING_INVALID": "매핑이 배열 계약에 맞지 않습니다",
    "MAPPING_SAMPLES_INVALID": "표본 배열은 2~5차원 숫자 배열이어야 합니다",
    "MAPPING_CHANNEL_INVALID": "채널 축은 3차원 이상 배열에서 길이 1~16인 축만 고를 수 있습니다",
    "MAPPING_META_INVALID": "라벨·그룹·표본 ID는 표본 수와 길이가 같은 1차원 배열이어야 하고 서로 달라야 합니다",
    "SPLIT_INVALID": "분할 비율·시드가 분할 계약에 맞지 않습니다",
    "GROUP_REQUIRED": "group 분할에는 그룹 배열이 필요합니다. 그룹 배열을 고르거나 random을 고르세요",
    "STRATIFY_INVALID": "계층 분할은 라벨 배열이 있고 random 분할이며 라벨 없는 표본을 유지하지 않고 같은 표본 묶기를 끈 경우에만 쓸 수 있습니다",
    "CLIP_RANGE_INVALID": "clipping 하한 분위수는 상한보다 작아야 합니다",
}
HINTS = {
    "NONFINITE_PRESENT": {"confirmed": True, "cause": "Train에 NaN·무한대 값이 있어 '있으면 중단' 설정으로 멈췄습니다(worker 세부 코드로 확인).",
                          "next": "계획 수정에서 비유한 값 처리를 'Train 채널 중앙값으로 채움'으로 바꾼 뒤 다시 확인하세요."},
    "GROUP_MISSING": {"confirmed": True, "cause": "그룹 배열에 비어 있는 값이 있습니다(worker 세부 코드로 확인).",
                      "next": "그룹 배열을 다른 배열로 바꾸거나 없음으로 두고 random 분할을 고르세요."},
    "SAMPLE_ID_INVALID": {"confirmed": True, "cause": "표본 ID 배열에 빈 값이나 중복이 있습니다(worker 세부 코드로 확인).",
                          "next": "표본 ID 배열을 없음으로 두면 위치 번호를 ID로 씁니다."},
    "SPLIT_EMPTY": {"confirmed": True, "cause": "분할 뒤 Train 또는 Validation 구간이 비었습니다(worker 세부 코드로 확인).",
                    "next": "비율을 바꾸거나, 그룹 수가 너무 적으면 random 분할을 고르세요."},
    "CHANNEL_ALL_NONFINITE": {"confirmed": True, "cause": "Train에서 유한값이 하나도 없는 채널이 있습니다(worker 세부 코드로 확인).",
                              "next": "채널 축 선택을 확인하세요. 입력 파일 확인이 필요할 수 있습니다."},
}
SPLIT_DEFAULTS = {"strategy": "random", "train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15, "random_state": 42,
                  "stratify": False}
PREPARATION_DEFAULTS = {"nonfinite": "block", "clip": "none", "clip_low": 1.0, "clip_high": 99.0, "scaling": "none",
                        "fixed_divisor": 255.0, "missing_labels": None, "duplicate_grouping": False}


def accepts(path):
    return Path(path).suffix.lower() == ".npz"


def _short(record):
    return record["task_id"][:10]


def _summary(record):
    return record["diagnostic"]["summary"]


def _plan(record):
    return record["confirmed"]["plan"]


# ==========================================
# 결정 A와 입력 진단
# ==========================================

def consent_a_content(record, action=None):
    if action is not None:
        raise ValueError("이 유형은 해결 동작이 없습니다.")

    return {"transfer": TRANSFER_SCOPE, "auto_steps": ["input_snapshot", "array_profile"], "allowed": [["array_profile", "audit"]],
            "array_code_id": code_id()}


def consent_a_text(model):
    return ["읽는 것: 선택한 .npz의 복사본만 사용하며 원본은 변경하지 않습니다. pickle은 읽지 않습니다.",
            f"모델 전달: 목표 문장과 배열 이름·형상·자료형, 값 집계(결측·평균·표준편차 등), 라벨 순번별 개수를 로컬 모델 {model}에만 "
            "보냅니다. 표본 값·이미지·라벨 이름·최솟값·최댓값은 보내지 않습니다.",
            "자동 진행: 입력 복사와 격리 진단, 확인용 표본 미리보기(로컬 전용). 진단 단계의 1회 승인은 이 동의 범위에서 위임 발급합니다.",
            "메모리: 입력 파일·압축 해제 크기 상한과 실제 계산 메모리 요구량은 다릅니다. 작업 메모리 상한을 넘으면 자동으로 늘리지 않고 중단합니다."]


def _drive(record, context, module_id, name, decision, expected, bound):
    """한 실행을 끝까지 진행한다. 승인은 해당 결정의 동의 범위에서만 위임 발급한다. (최종 상태, 저장한 기록)을 돌려준다."""
    item = service(module_id)

    for _ in range(8):
        state = item.status(name)

        if state["status"] in TERMINAL:
            return state, record

        if state["actions"] and state["actions"][0]["id"] == "approve":
            authorize = context.authorizer(record, decision, module_id, state["stage"], expected=expected, bound=bound)
            result = item.record_delegated_approval(name, item.preview_approval(name), authorize=authorize)
            record["diagnostic"]["delegation"] = {"consent_id": record["consents"][decision]["consent_id"],
                                                  "scope_sha256": result["scope_sha256"]}
            record = context.save(record)
            current = item.status(name)
            item.next(name, expected_stage=current["stage"], expected_revision=current["revision"], approval_id=result["approval_id"])
        elif state["actions"]:
            item.next(name, expected_stage=state["stage"], expected_revision=state["revision"])
        else:
            break

    raise TaskProblem("STATE_UNCLEAR", "진단 작업 상태가 진행되지 않았습니다. 결과 확인이 필요합니다.")


def diagnose(record, directory, context, *, name=None):
    """격리 진단을 진행하고 검증된 진단 요약만 기록한다. 승인은 결정 A로만 위임 발급한다."""
    name = name or "t" + _short(record) + "-diag"
    snapshot = record["snapshot"]
    record["diagnostic"] = {"run_name": name, "status": "reserved"}
    record = context.save(record)
    service("array_profile").create(name, snapshot_directory=snapshot["directory"], snapshot=snapshot["snapshot"],
                                    packet={"max_total_bytes": 2 * 1024 ** 3}, limits=record["limits"])
    state, record = _drive(record, context, "array_profile", name, "A", expected={"array_code_id": code_id(), "limits": record["limits"]},
                   bound={"snapshot_sha256": snapshot["snapshot"]["sha256"], "array_code_id": code_id()})

    if state["status"] == "failed_runtime":
        # 실행기는 중단 상태만 기록한다. 원인이 메모리라고 단정하지 않고 가능한 원인과 다음 행동을 함께 안내한다.
        raise TaskProblem("DIAGNOSTIC_RUNTIME_FAILED", "배열 진단의 격리 실행이 끝나지 않았습니다(시간·메모리 상한 초과 등 실행 환경 문제일 수 "
                          "있음). 파일 크기 상한 안이어도 계산 메모리는 더 필요할 수 있습니다. 상한을 자동으로 늘리지 않았습니다. 설정의 자원 상한을 "
                          "확인하거나 더 작은 파일로 나눠 새 작업으로 시작하세요.", diagnostic=service("array_profile").worker_diagnostic(name))

    if state["status"] != "completed":
        diagnostic = service("array_profile").worker_diagnostic(name)
        detail = (diagnostic or {}).get("detail")
        raise TaskProblem("DIAGNOSTIC_BLOCKED", "배열 진단이 입력 확인에서 멈췄습니다. 지원 형식(.npz, 숫자·bool·문자열 배열, pickle 없음)과 "
                          "파일 상태를 확인한 뒤 새 작업으로 시작하세요." + (" (확인된 사유 코드: " + detail + ")" if detail else ""),
                          diagnostic=diagnostic)

    _, summary = service("array_profile").verified(name)

    if not sample_candidates(summary["headers"]):
        raise TaskProblem("NO_SAMPLE_ARRAY", "표본 배열(첫 축이 표본인 2~5차원 숫자 배열)이 없습니다. 파일 구성을 확인한 뒤 새 작업으로 시작하세요.")

    record["diagnostic"].update(status="completed", summary=summary)
    return record


def input_notices(record):
    return [LABEL_LIMIT]


def diagnosis_view(record):
    """결정 화면의 진단 요약. 배열 이름은 표 값으로만 쓴다."""
    summary = _summary(record)
    rows = []

    for key, header in sorted(summary["headers"].items()):
        sample = summary["samples"].get(key)
        meta = summary["meta"].get(key)
        row = {"배열": key, "형상": "×".join(str(value) for value in header["shape"]), "자료형": header["dtype"],
               "구분": "표본 배열 후보" if sample else "1차원(라벨·그룹·ID 후보)" if meta else "사용 안 함"}

        if sample:
            overall = sample["overall"]
            row.update({"비유한 값": overall["nonfinite"], "값 종류": overall["distinct"] if overall["distinct"] <= 256 else ">256",
                        "평균": _round(overall["mean"]), "최솟값(로컬)": _round(overall["min"]), "최댓값(로컬)": _round(overall["max"]),
                        "상수 표본": sample["constant_samples"], "동일 표본": sample["duplicate_samples"]})
        elif meta:
            row.update({"비유한 값": meta["missing"], "값 종류": meta["unique"]})

        rows.append(row)

    return {"lines": [f"배열 {len(summary['headers'])}개 · 표본 배열 후보 {len(summary['samples'])}개. 최솟값·최댓값은 로컬 표시용이며 모델에 보내지 않습니다."],
            "rows": rows}


def preview_images(record):
    """입력 확인용 표본 미리보기. 매번 결과 검증을 거친 PNG만 돌려준다. 모델·전달 묶음에는 넣지 않는다."""
    folder, summary = service("array_profile").verified(record["diagnostic"]["run_name"])
    found = []

    for entry in summary["previews"]:
        data = _read_file(folder / "previews" / entry["file"], 512 * 1024)

        # 표시 직전에도 검증기와 같은 고정 형식 검사를 한다(검증 뒤 파일이 바뀌는 경우 대비).
        if _preview_png(data):
            found.append({"label": f"입력 미리보기 · 표본 위치 {entry['position']} (처리 결과 아님)", "data": data})

    return found


def _round(value):
    return MISSING if value is None else float(f"{value:.4g}")


# ==========================================
# 권장안·계획·질문
# ==========================================

def plan_fields(record):
    headers = _summary(record)["headers"]
    one_dim = sorted(key for key, item in headers.items() if len(item["shape"]) == 1)
    number = lambda path, low, high: {"path": path, "type": "number", "nullable": False, "minimum": low, "maximum": high}
    enum = lambda path, options, nullable=False: {"path": path, "type": "enum", "options": list(options), "nullable": nullable}
    return [enum("mapping.samples", sample_candidates(headers), True), enum("mapping.channel_axis", CHANNEL_AXES),
            *[enum("mapping." + role, one_dim, True) for role in ("label", "group", "sample_id")],
            enum("split.strategy", ("random", "group")), number("split.train_ratio", 0.01, 0.98),
            number("split.validation_ratio", 0.01, 0.98), number("split.test_ratio", 0.01, 0.98),
            {"path": "split.random_state", "type": "integer", "nullable": False, "minimum": 0, "maximum": 2 ** 32 - 1},
            {"path": "split.stratify", "type": "boolean", "nullable": False},
            enum("preparation.nonfinite", ("block", "median")), enum("preparation.clip", ("none", "percentile")),
            number("preparation.clip_low", 0.0, 49.0), number("preparation.clip_high", 51.0, 100.0),
            enum("preparation.scaling", ("none", "standard", "minmax", "fixed")), number("preparation.fixed_divisor", 1e-6, 1e9),
            enum("preparation.missing_labels", ("exclude", "keep"), True),
            {"path": "preparation.duplicate_grouping", "type": "boolean", "nullable": False}]


def base_plan(record):
    """형상만 보고 정한 기본 계획. 확정은 사용자가 한다."""
    return {"mapping": dict(_summary(record)["base_mapping"]), "split": dict(SPLIT_DEFAULTS), "preparation": dict(PREPARATION_DEFAULTS)}


def _classes(meta):
    """라벨 후보의 순번별 개수. 클래스 이름은 보내지 않는다."""
    classes = meta.get("classes")
    return None if classes is None or len(classes) > 20 else [entry["count"] for entry in classes]


def recommendation_input(record):
    """권장안 입력. 배열 이름·형상·자료형과 값 집계만 준다. 표본 값·라벨 이름·최솟값·최댓값·경로는 넣지 않는다.

    집계 이름에 건수·분모·비율 기준을 적는다(예: nonfinite_ratio_of_values = nonfinite_count / values_total).
    """
    summary = _summary(record)
    items = []

    for key, header in sorted(summary["headers"].items()):
        sample, meta = summary["samples"].get(key), summary["meta"].get(key)
        overall = dict((sample or {}).get("overall") or {})

        # 값이 한 종류이거나 유한값이 2개 이하이면 평균·표준편차로 개별 값을 복원할 수 있어 보내지 않는다(표형과 같은 규칙).
        # 값이 정확히 두 종류이고 하나가 0이면 0 비율과 평균으로 나머지 값을 복원할 수 있어 함께 뺀다. 로컬 진단·계산에는 그대로 쓴다.
        if overall and (overall["distinct"] <= 1 or overall["values"] - overall["nonfinite"] <= 2
                        or (overall["distinct"] == 2 and (overall["zero_ratio"] or 0) > 0)):
            overall.update(mean=None, std=None)

        total = math.prod(header["shape"])
        finite = overall["values"] - overall["nonfinite"] if overall else None
        missing = overall.get("nonfinite", (meta or {}).get("missing"))
        detail = {
            "role_candidate": "samples" if sample else "metadata_1d" if meta else "unused",
            "shape": header["shape"], "dtype_kind": header["kind"], "samples_n": header["shape"][0] if header["shape"] else None,
            "values_total": total,
            "nonfinite_or_missing_count": missing,
            "nonfinite_or_missing_ratio_of_values": None if missing is None or not total else round(missing / total, 4),
            "zero_count_of_finite_values": None if overall.get("zero_ratio") is None else round(overall["zero_ratio"] * finite),
            "zero_ratio_of_finite_values": None if overall.get("zero_ratio") is None else round(overall["zero_ratio"], 4),
            "mean_of_finite_values": None if overall.get("mean") is None else float(f"{overall['mean']:.4g}"),
            "std_of_finite_values": None if overall.get("std") is None else float(f"{overall['std']:.4g}"),
            "distinct_values_capped_257": overall.get("distinct", (meta or {}).get("unique")),
            "duplicate_sample_count_of_samples_n": (sample or {}).get("duplicate_samples"),
            "class_counts_by_index": _classes(meta) if meta else None}
        items.append({"id": key, "priority": 2 if sample else 1, "reasons": ["sample_candidate"] if sample else [],
                      "bucket": "samples" if sample else "meta", "paths": [], "detail": detail})

    instructions = (
        "이 작업은 배열 표본 묶음(.npz)입니다. subjects는 배열 이름별 집계입니다. 경로는 mapping.<역할>, split.<항목>, preparation.<항목> 형식입니다. "
        "집계 이름의 _of_ 뒤는 분모입니다: nonfinite_or_missing_ratio_of_values = 개수/values_total, zero_ratio_of_finite_values = 0인 값/유한값. "
        "0의 비율을 비유한 값(NaN·무한대) 비율로 읽지 마세요. 숫자를 인용할 때는 필드 이름 그대로 쓰세요. "
        "수치 전처리(preparation.nonfinite·clip·clip_low·clip_high·scaling·fixed_divisor)는 mapping.samples로 고른 표본 배열에만 적용되며 "
        "라벨·그룹·ID 배열에는 적용되지 않습니다. 라벨·그룹·ID 배열의 자료형(문자열 등)을 이유로 수치 전처리를 바꾸거나 묻지 마세요. "
        "mapping.group은 같은 묶음을 한 구간에 모으는 분할 기준이며 라벨이 아닙니다. mapping.label·mapping.group을 바꿀 때는 reason에 근거(배열 이름·분포)를 쓰세요. "
        "배열의 의미는 확인할 수 없으므로 사용자가 확인합니다. class_counts_by_index는 클래스 순번별 개수이며 클래스 이름은 제공하지 않습니다. "
        "클래스 의미를 추측하지 마세요. 기본 계획과 같은 값은 changes에 넣지 마세요. 시간 기준 분할은 지원하지 않습니다.")
    return {"data": {"arrays": len(summary["headers"])}, "instructions": instructions, "items": items}


def change_check(record, change):
    """권장 변경의 확인 안내. 라벨·그룹 매핑 변경은 의미를 확인하지 못한 모델 의견이므로 사용자 확인이 필요하다."""
    if change["path"] in ("mapping.label", "mapping.group"):
        return "확인 필요: 모델은 배열 이름·분포만 보고 권했고 배열의 의미는 확인하지 못했습니다. 확정 전 필수 확인 질문에 답해야 합니다."

    if change["path"].startswith("mapping."):
        return "확인 권장: 모델은 배열 이름·형상만 보고 권했습니다."

    return None


def validate_plan(record, plan):
    """계획 전체 검증. 화면 선검사·권장안 병합·서버 확정이 같은 함수를 쓴다."""
    fields = fields_by_path(plan_fields(record))

    if (type(plan) is not dict or set(plan) != {"mapping", "split", "preparation"}
            or any(type(plan[section]) is not dict for section in plan)
            or {section + "." + key for section in plan for key in plan[section]} != set(fields)
            or any(not value_allowed(fields[section + "." + key], value) for section in plan for key, value in plan[section].items())):
        raise PlanError("PLAN_FIELDS_INVALID", CODE_LABELS["PLAN_FIELDS_INVALID"])

    mapping, split, options = plan["mapping"], plan["split"], plan["preparation"]

    try:
        check_mapping(_summary(record)["headers"], mapping)
    except ArrayInputError as error:
        raise PlanError(error.code if error.code in CODE_LABELS else "MAPPING_INVALID",
                        CODE_LABELS.get(error.code, CODE_LABELS["MAPPING_INVALID"]), subjects=["mapping.samples"]) from error

    try:
        SplitRules(**split)
    except ValueError as error:
        raise PlanError("SPLIT_INVALID", CODE_LABELS["SPLIT_INVALID"], subjects=["split.train_ratio"]) from error

    if split["strategy"] == "group" and mapping["group"] is None:
        raise PlanError("GROUP_REQUIRED", CODE_LABELS["GROUP_REQUIRED"], subjects=["split.strategy", "mapping.group"])

    if split["stratify"] and (mapping["label"] is None or split["strategy"] != "random" or options["missing_labels"] == "keep"
                              or options["duplicate_grouping"]):
        raise PlanError("STRATIFY_INVALID", CODE_LABELS["STRATIFY_INVALID"], subjects=["split.stratify"])

    if options["clip_low"] >= options["clip_high"]:
        raise PlanError("CLIP_RANGE_INVALID", CODE_LABELS["CLIP_RANGE_INVALID"], subjects=["preparation.clip_low", "preparation.clip_high"])


def questions(record, plan):
    """사용자가 정하거나 확인해야 할 항목. 필수 여부는 최종 계획에서 다시 계산한다. 자동으로 값을 바꾸지 않는다."""
    summary, mapping, options = _summary(record), plan["mapping"], plan["preparation"]
    headers, result = summary["headers"], []
    candidates = sample_candidates(headers)

    if mapping["samples"] is None:
        result.append(question("mapping.samples", "required", "표본 배열(첫 축이 표본인 숫자 배열)을 고르세요.", path="mapping.samples",
                               options=[{"value": key, "label": key} for key in candidates]))
        return result

    meta = meta_candidates(headers, mapping["samples"])
    stats = summary["samples"][mapping["samples"]]

    # 모델이 라벨·그룹 매핑을 권했고 화면에 표시한 계획과 최종 계획이 모두 그 값을 쓰면 사용자가 의미를 확인해야 확정할 수 있다.
    # 값을 바꾸면 질문이 사라진다. 표시한 계획에 없던 값을 사용자가 양식에서 직접 고른 경우는 사용자 결정이므로 묻지 않는다
    # (화면에 보이지 않는 필수 질문으로 막히지 않게).
    shown = (record.get("confirmed") or {}).get("plan") or record["draft"]["plan"]

    if record["draft"]["source"] == "model":
        for role, name, subject, using in (("label", "라벨", "라벨이", "라벨로"), ("group", "그룹", "그룹이", "그룹으로")):
            change = next((item for item in record["draft"].get("changes", []) if item["path"] == "mapping." + role), None)

            if change is not None and change["value"] is not None and mapping[role] == change["value"] == shown["mapping"][role]:
                result.append(question("mapping." + role + ".confirm", "required",
                                       f"모델이 {name} 배열을 권했습니다(관련 항목). 모델은 배열 이름·분포만 보고 판단했고 의미는 확인하지 못했습니다. "
                                       f"이 배열이 {subject} 맞으면 확인을 고르고, 아니면 양식에서 바꾸세요.",
                                       options=[{"value": "confirmed", "label": f"맞음: 이 배열을 {using} 사용"}], subjects=[change["value"]]))

    if mapping["label"] is None and meta:
        result.append(question("mapping.label", "advisory", "라벨(정답) 배열이 있으면 고르세요. 없으면 답하지 않아도 됩니다.",
                               path="mapping.label", options=[{"value": key, "label": key} for key in meta]))

    if mapping["label"] is not None and summary["meta"].get(mapping["label"], {}).get("missing") and options["missing_labels"] is None:
        result.append(question("preparation.missing_labels", "required", "라벨이 비어 있는 표본이 있습니다. 분할 전에 제외할지, 라벨 없이 유지할지 고르세요. "
                                                                          "원본은 지우지 않습니다.",
                               path="preparation.missing_labels",
                               options=[{"value": "exclude", "label": "분할 전 제외"}, {"value": "keep", "label": "라벨 없이 유지"}]))

    if stats["overall"]["nonfinite"] and options["nonfinite"] == "block":
        result.append(question("preparation.nonfinite", "required", "표본 배열에 NaN·무한대 값이 있습니다. 그대로는 진행할 수 없습니다. "
                                                                     "Train 채널 중앙값으로 채우려면 아래를 고르세요.",
                               path="preparation.nonfinite", options=[{"value": "median", "label": "Train 채널 중앙값으로 채움"}]))

    if stats["duplicate_samples"] and not options["duplicate_grouping"]:
        result.append(question("preparation.duplicate_grouping", "advisory", "값이 완전히 같은 표본이 있습니다. 서로 다른 구간에 들어가면 누수가 생길 수 "
                                                                            "있습니다. 같은 표본을 한 구간에 모을까요? 삭제는 하지 않습니다.",
                               path="preparation.duplicate_grouping", recommended=True,
                               options=[{"value": True, "label": "같은 표본을 한 구간에 모으기"}, {"value": False, "label": "모으지 않음"}]))

    shape = headers[mapping["samples"]]["shape"]

    if mapping["channel_axis"] == "none" and len(shape) >= 3 and (shape[-1] <= 4 or shape[1] <= 4):
        result.append(question("mapping.channel_axis", "advisory", "축 길이가 작은 차원이 있습니다. 채널 축이면 고르세요(채널별로 기준을 계산합니다).",
                               path="mapping.channel_axis",
                               options=[{"value": value, "label": array_options.value_text("mapping.channel_axis", value)} for value in CHANNEL_AXES]))

    return result


def on_rejected(record, draft):
    return draft


def check_resolution(record, action):
    raise ValueError("이 유형은 해결 동작이 없습니다.")


def resolution_facts(record, action):
    return {}


def option_guide(record, plan, origins):
    return array_options.option_guide(record, plan, origins)


def approval_stages(module_id):
    from .array_task_service import MODULES

    return [MODULES[module_id]]


def prepare(record, directory, plan, label):
    """확정 계획을 새 파일로 남긴다. 기존 파일은 덮어쓰지 않는다."""
    validate_plan(record, plan)
    folder = directory / "plans"
    folder.mkdir(exist_ok=True)

    with (folder / (label + ".json")).open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, sort_keys=True)

    return {"plan_name": label, "plan_signature": signature(plan)}


# ==========================================
# 동의 내용·실행 대조값
# ==========================================

def consent_c_content(record):
    plan = _plan(record)
    return {"array_code_id": code_id(), "mapping": plan["mapping"], "split": plan["split"], "preparation": plan["preparation"]}


def consent_c_text():
    return ("동의 범위(결정 C): 확인한 매핑·분할·전처리로 표본 메타데이터 기준 분할 → Train 채널 기준 산출 → Train 예상 영향 집계까지 "
            "자동 진행합니다. 단계의 1회 승인은 이 범위에서 위임 발급합니다. 변환 파일은 아직 만들지 않으며 실제 적용은 후보를 본 뒤 따로 확인합니다.")


def consent_d_content(record):
    return {"candidate_signature": review_signature(record), "array_code_id": code_id()}


def consent_d_text():
    return ("Train 구간에서 산출한 채널 기준입니다. 적용하면 같은 기준으로 Train·Validation 표본을 float32 배열로 바꾸고 표본 메타데이터와 "
            "기준 파일을 결과 묶음으로 만듭니다. Test에는 적용하지 않고 기준을 다시 산출하지 않습니다. 원본·작업 복사본은 바꾸지 않습니다.")


def expected(record, decision):
    values = {"array_code_id": code_id()}

    if decision == "D":
        values["candidate_signature"] = review_signature(record)

    return values


# ==========================================
# 단계 생성·입력 계보
# ==========================================

def create_step(record, directory, step_id, name):
    plan, snapshot = _plan(record), record["snapshot"]
    packet = {"headers": _summary(record)["headers"], "mapping": plan["mapping"], "split": plan["split"],
              "preparation": plan["preparation"], "plan_digest": signature(plan)}
    parent = record["steps"]["prepare"]["run_name"] if step_id == "apply" else None

    if parent is not None:
        # 적용 단계는 결정 D에서 확인한 후보(실행 이름·서명)에 묶는다. 서명은 승인 인자(config_id)에 들어간다.
        packet["candidate"] = {"run": parent, "signature": review_signature(record)}

    service(dict((step, module) for step, module, _ in STEPS)[step_id]).create(
        name, snapshot_directory=snapshot["directory"], snapshot=snapshot["snapshot"], packet=packet, limits=record["limits"], parent=parent)


def check_lineage(record):
    """단계 실행이 진단한 작업 복사본과 확정 계획으로 만들어졌는지 확인한다."""
    for step_id, item in record["steps"].items():
        config = service(item["module_id"]).config(item["run_name"])

        parent = record["steps"]["prepare"]["run_name"] if step_id == "apply" else None

        if (config["snapshot"]["sha256"] != record["snapshot"]["snapshot"]["sha256"] or config["packet"]["plan_digest"] != signature(_plan(record))
                or config["parent"] != parent or (parent is not None and config["packet"].get("candidate", {}).get("run") != parent)):
            raise TaskProblem("SNAPSHOT_CHANGED", "단계 실행의 입력·계획·선행 후보가 확인한 작업과 다릅니다. 새 작업으로 시작하세요.")


def blocked_problem(record, step_id):
    return None


def failure_hint(record, step_id, diagnostic):
    """worker 세부 코드로 확인된 원인. 근거가 없으면 None(추정하지 않음)."""
    detail = (diagnostic or {}).get("detail")
    return {"code": detail, "subjects": []} if detail in HINTS else None


def step_completed(record, step_id):
    """완료 기록 복구용 단계 확인. 결과 검증을 통과하면 검증 방식을 돌려준다."""
    item = record["steps"][step_id]
    service(item["module_id"]).verified(item["run_name"])
    return "current_code"


# ==========================================
# 결정 D 대상과 결과 요약 (검증된 결과만 읽는다)
# ==========================================

def _candidate(record):
    return service("array_prepare").verified(record["steps"][REVIEW_STEP]["run_name"])[1]


def review_signature(record):
    """결정 D 대상 서명. 완료 표식 해시와 검증된 후보 내용을 함께 묶는다."""
    return service("array_prepare").signature(record["steps"][REVIEW_STEP]["run_name"])


def _channels(state, impact, train):
    rows = []

    for record, item in zip(state["channels"], impact):
        rows.append({"채널": record["channel"], "대치값(Train 중앙값)": _round(record["impute"]) if record["impute"] is not None else "-",
                     "clip 경계": "-" if record["clip"] is None else f"{_round(record['clip'][0])} ~ {_round(record['clip'][1])}",
                     "center": _round(record["center"]), "scale": _round(record["scale"]),
                     "clipping 변경 값": "-" if item["clipped"] is None else f"{item['clipped']} ({item['clipped'] / max(item['values'], 1):.1%})",
                     "대치한 값": item["imputed"], "변환 후 상수": "예" if item["constant_after"] else "아니오"})

    return rows


def _split_rows(assignment, use_index):
    """구간별 표본·클래스·그룹 수. 클래스는 로컬 화면에 값으로 보인다(모델에는 보내지 않음)."""
    return [{"구간": item["split"], "표본": item["samples"],
             "클래스": ", ".join(f"{entry['value']}: {entry['count']}" for entry in item["classes"]) or "-",
             "그룹": "-" if item["groups"] is None else item["groups"]} for item in assignment["summary"]]


def _warnings(record, verified):
    assignment, candidate = verified["assignment"], verified["candidate"]
    options = _plan(record)["preparation"]
    lost = [item["channel"] for item in candidate["impact"] if item["constant_after"]]
    warnings = []

    if assignment["cross_split_duplicates"]:
        warnings.append(f"완전히 같은 표본 {assignment['cross_split_duplicates']}개가 다른 구간에 나뉘었습니다(누수 위험). "
                        "'같은 표본을 한 구간에 모으기'를 켜고 계획을 수정하면 막을 수 있습니다.")

    if assignment["group_overlap"]:
        warnings.append(f"그룹 배열을 골랐지만 무작위 분할이라 같은 그룹이 여러 구간에 있습니다(겹친 그룹 {assignment['group_overlap']}개, 누수 위험). "
                        "그룹 단위로 나누려면 분할 방식을 group으로 바꾸세요.")

    if lost:
        warnings.append(f"Train 기준 적용 후 모든 값이 같아지는 채널 {len(lost)}개(채널 {', '.join(map(str, lost))}, Train 후보 계산 집계). "
                        "이 채널은 학습에 정보를 주지 못합니다(정보 손실).")

    clipped = sum(item["clipped"] or 0 for item in candidate["impact"])

    if clipped:
        warnings.append(f"clipping으로 Train 값 {clipped}개가 경계값으로 바뀝니다(Train 후보 계산 집계).")

    if "CLASS_MISSING_IN_SPLIT" in assignment["warnings"]:
        warnings.append("일부 구간에 없는 클래스가 있습니다. 비율·분할 방식을 확인하세요.")

    if assignment["excluded"]:
        warnings.append(f"라벨 없는 표본 {len(assignment['excluded'])}개를 분할 전에 제외했습니다(원본은 그대로).")

    if warnings:
        warnings.append("경고 때문에 설정을 자동으로 바꾸지 않았습니다. 바꾸려면 적용하지 말고 '계획 수정'에서 해당 항목을 고친 뒤 다시 확인하세요.")

    return warnings


def review(record):
    """결정 D에서 보여 줄 후보. 결과 검증을 통과한 후보에서 구조화된 표와 Train 예상 영향을 만든다."""
    name = record["steps"][REVIEW_STEP]["run_name"]
    verified = _candidate(record)
    assignment, state, candidate = verified["assignment"], verified["state"], verified["candidate"]
    plan = _plan(record)
    train = len(assignment["partitions"]["train"])
    metrics = [{"label": name_ + " 표본", "value": len(assignment["partitions"][key])}
               for key, name_ in (("train", "Train"), ("validation", "Validation"), ("test", "Test"))]
    metrics += [{"label": "제외 표본(라벨 없음)", "value": len(assignment["excluded"])},
                {"label": "구간을 넘는 동일 표본", "value": assignment["cross_split_duplicates"]},
                {"label": "clipping으로 바뀌는 값 (예상)", "value": sum(item["clipped"] or 0 for item in candidate["impact"])},
                {"label": "변환 후 상수 채널 (예상)", "value": sum(1 for item in candidate["impact"] if item["constant_after"])}]
    settings = [{"항목": array_options.GUIDE[section + "." + key]["label"] if section + "." + key in array_options.GUIDE else key,
                 "설정 이름": section + "." + key, "값": array_options.value_text(section + "." + key, value)}
                for section in ("mapping", "split", "preparation") for key, value in plan[section].items()]
    facts = {"metrics": metrics,
             "tables": [{"title": "구간별 표본", "rows": _split_rows(assignment, True)},
                        {"title": "채널별 Train 기준과 예상 영향 (Train 후보 계산)", "rows": _channels(state, candidate["impact"], train)},
                        {"title": "실제 실행 설정 (후보 계산에 쓴 값)", "rows": settings}],
             "warnings": _warnings(record, verified),
             "notes": [array_options.FLOAT32_NOTE,
                       "표의 영향 수치는 Train 후보 계산에서 격리 worker가 센 예상치입니다. 부모 검증은 형식·범위와 저장 기준으로 정해지는 값만 확인했습니다. "
                       "적용하면 Train 합계를 적용 결과와 다시 대조하고, Validation 실측치는 결과 보고서에 따로 표시합니다.", LABEL_LIMIT],
             "text": None, "text_truncated": False}
    return {"run_name": name, "signature": review_signature(record), "facts": facts}


def images(record):
    """결과 화면의 그림. 입력 확인용 미리보기를 처리 결과와 구분한 이름으로 돌려준다."""
    try:
        return preview_images(record)
    except (OSError, ValueError, KeyError, TypeError):
        return []


FILE_USES = {"train_samples.npy": "학습용 표본(float32, 입력과 같은 축 순서)",
             "validation_samples.npy": "검증용 표본(float32)",
             "train_meta.json": "Train 표본 위치·ID·라벨·그룹. 표본 배열과 같은 순서",
             "validation_meta.json": "Validation 표본 위치·ID·라벨·그룹. 표본 배열과 같은 순서",
             "preprocessor.json": "Train 채널 기준(대치값·clip 경계·center·scale). Test·새 데이터에 같은 기준을 적용할 때 사용",
             "application.json": "적용 기록(구간별 실측 영향 합계·설정)"}


def report_facts(record):
    """검증된 결과에서 보고서 사실만 모은다. 원본을 다시 분석하거나 수치를 추정하지 않는다."""
    if set(record["steps"]) != set(STEP_LABELS):
        raise ValueError("작업 기록의 단계 구성이 완료 작업과 다릅니다.")

    check_lineage(record)
    verified = _candidate(record)
    folder, application = service("array_apply").verified(record["steps"]["apply"]["run_name"])
    assignment, candidate = verified["assignment"], verified["candidate"]
    plan, summary = _plan(record), _summary(record)
    mapping = plan["mapping"]
    shape = summary["headers"][mapping["samples"]]["shape"]
    expected_totals = {"clipped": sum(item["clipped"] or 0 for item in candidate["impact"]),
                       "imputed": sum(item["imputed"] for item in candidate["impact"]),
                       "constant_channels": sum(1 for item in candidate["impact"] if item["constant_after"])}
    measured = application["impact"]
    basis_after = "적용 결과(적용 검증기가 출력 배열로 대조, clipping은 범위만)"
    basis_expected = "예상: Train 후보 계산(worker 집계), 적용 Train 합계와 일치(교차 대조)"
    comparison = [
        {"항목": "표본 수", "전": shape[0],
         "후 (실측)": f"train {len(assignment['partitions']['train'])} · validation {len(assignment['partitions']['validation'])} 변환",
         "예상 (후보)": " · ".join(f"{item['split']} {item['samples']}" for item in assignment["summary"]), "근거": "분할 확정 결과 / " + basis_after},
        {"항목": "clipping으로 바뀐 값", "전": "-", "후 (실측)": f"train {measured['train']['clipped']} · validation {measured['validation']['clipped']}",
         "예상 (후보)": f"train {expected_totals['clipped']}", "근거": "후: 적용 worker 집계(범위만 검사) / " + basis_expected},
        {"항목": "대치한 비유한 값", "전": summary["samples"][mapping["samples"]]["overall"]["nonfinite"],
         "후 (실측)": f"train {measured['train']['imputed']} · validation {measured['validation']['imputed']}",
         "예상 (후보)": f"train {expected_totals['imputed']}",
         "근거": "전: 진단(전체 입력) / 후: 적용 worker 집계(범위만 검사) / " + basis_expected},
        {"항목": "변환 후 상수 채널", "전": "-",
         "후 (실측)": f"train {measured['train']['constant_channels']} · validation {measured['validation']['constant_channels']}",
         "예상 (후보)": f"train {expected_totals['constant_channels']}", "근거": "후: " + basis_after + " / " + basis_expected},
        {"항목": "구간을 넘는 동일 표본", "전": summary["samples"][mapping["samples"]]["duplicate_samples"],
         "후 (실측)": assignment["cross_split_duplicates"], "예상 (후보)": "-", "근거": "분할 확정 결과(표본 해시 대조)"},
        {"항목": "Test 구간", "전": f"{len(assignment['partitions']['test'])}개", "후 (실측)": "변환 적용 안 함", "예상 (후보)": "-", "근거": "적용 계약"},
    ]
    processing = [{"항목": array_options.GUIDE[path]["label"], "값": array_options.value_text(path, plan[path.split(".")[0]][path.split(".")[1]]),
                   "결정 출처": _origin(record, path)} for path in array_options.GUIDE]
    problems = []
    stats = summary["samples"][mapping["samples"]]

    for label, value in (("비유한 값", stats["overall"]["nonfinite"]), ("상수 표본", stats["constant_samples"]),
                         ("완전히 같은 표본", stats["duplicate_samples"])):
        if value:
            problems.append({"구분": label, "대상": mapping["samples"], "내용": f"{value}개"})

    if mapping["label"] is not None and summary["meta"][mapping["label"]]["missing"]:
        problems.append({"구분": "라벨 결측", "대상": mapping["label"], "내용": f"{summary['meta'][mapping['label']]['missing']}개"})

    warnings = [{"구분": "정보 손실", "내용": text} for text in _warnings(record, verified)[:-1] if "누수" not in text]
    warnings += [{"구분": "미처리", "내용": text} for text in _warnings(record, verified) if "누수" in text]
    warnings += [{"구분": "정보 손실", "내용": array_options.FLOAT32_NOTE},
                 {"구분": "미처리", "내용": "Test 구간에는 변환을 적용하지 않았습니다."},
                 {"구분": "미검증", "내용": "모델 학습·성능 평가는 하지 않았습니다."},
                 {"구분": "미검증", "내용": "clipping 변경 값 수와 채널별 영향은 격리 worker가 센 값입니다. 부모 검증은 범위·일관성과 "
                                          "후보·적용 Train 합계의 일치만 확인했습니다."},
                 {"구분": "미검증", "내용": "근사 중복(값이 조금 다른 같은 대상)은 찾지 않았습니다."},
                 {"구분": "미검증", "내용": LABEL_LIMIT}]
    files = [{"파일": path.name, "용도": FILE_USES.get(path.name, "설명 없음"), "크기(byte)": path.stat().st_size}
             for path in sorted(Path(folder).iterdir())]
    split = plan["split"]
    return {
        "title": "배열 전처리 결과 보고서",
        "highlights": [f"표본 {shape[0]}개(표본 형상 {'×'.join(map(str, shape[1:]))})를 Train 채널 기준으로 전처리해 Train "
                       f"{len(assignment['partitions']['train'])}개 · Validation {len(assignment['partitions']['validation'])}개를 float32 배열로 만들었습니다.",
                       "Test는 변환하지 않았습니다. 그림의 표본 미리보기는 입력 확인용이며 처리 결과가 아닙니다."],
        "lines": [f"분할: {split['strategy']}{' + 같은 표본 묶기(그룹 단위)' if plan['preparation']['duplicate_grouping'] else ''} "
                  f"{split['train_ratio']}/{split['validation_ratio']}/{split['test_ratio']} (시드 {split['random_state']})."],
        "metrics": [{"label": "입력 표본", "value": shape[0]}, {"label": "채널", "value": len(candidate["impact"])},
                    {"label": "Train/Validation/Test", "value": " / ".join(str(len(assignment["partitions"][key])) for key in ("train", "validation", "test"))}],
        "problems": problems, "processing": processing, "comparison": comparison,
        "comparison_note": "후 (실측)은 적용 결과, 예상 (후보)은 Train 후보 계산(적용 전, worker 집계)입니다.",
        "warnings": warnings,
        "usage": ["train_samples.npy로 학습하고 validation_samples.npy로 설정을 비교하세요. 표본 순서는 같은 이름의 meta.json과 같습니다.",
                  "라벨·그룹·표본 ID는 meta.json에 있습니다. 표본 배열에는 들어 있지 않습니다.",
                  "Test·새 데이터에는 preprocessor.json의 Train 기준을 그대로 적용해야 합니다. 이 작업은 Test에 적용하지 않았습니다.",
                  "결과 파일 경로는 기술 상세에서 확인하세요."],
        "images": [{"label": item["label"]} for item in images(record)],
        "cautions": ["진단 집계(비유한 값·동일 표본 등)는 전체 입력 기준입니다. 전처리 기준은 Train 구간에서만 산출했습니다."],
        "files": files, "files_title": "결과 파일 (적용 결과 묶음)",
        "technical": {"source_path": record["source"]["path"], "snapshot_sha256": record["snapshot"]["snapshot"]["sha256"],
                      "result_folder": str(folder), "array_code_id": candidate["binding"]["code_id"],
                      "assignment_counts": {key: len(value) for key, value in assignment["partitions"].items()}},
        "outputs": {"bundle": str(folder)},
    }


def _origin(record, path):
    section, key = path.split(".", 1)
    answered = {item["path"] for item in record["confirmed"].get("answers", []) if item.get("path")}
    drafted = record["draft"]["plan"][section].get(key)

    if path in answered:
        return "질문 답변"

    if _plan(record)[section].get(key) != drafted:
        return "사용자 변경"

    if drafted != base_plan(record)[section].get(key):
        return "모델 권장"

    return "기본 계획"

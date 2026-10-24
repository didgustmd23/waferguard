"""후보 JSON 상태·출력 계약을 코드로 검사하는 부모 검증기."""

from dataclasses import asdict
import json

from src.preprocessing.transform import _validate_state
from .artifacts import _root, _fingerprint
from .isolated_eda import _read_file
from .split_evidence import read_metadata
from .preparation_candidate_report import candidate_markdown


def read_candidate(directory, contract, code_id, *, committed=False):
    """원본 재파싱·통계 재계산·fit 없이 필드·범위·순서·출처·상한을 확인한다."""
    try:
        # Train을 한 번 검사하고 같은 결과로 기대 binding을 만든다.
        binding = json.loads(json.dumps(contract.packet(code_id)))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        return None
    return read_checked_candidate(directory, contract, code_id, binding, committed=committed)


def read_checked_candidate(directory, contract, code_id, binding, *, committed=False):
    """같은 호출 안에서 contract.packet(code_id)로 검사·생성한 binding으로 후보 파일을 확인한다.

    binding을 다른 호출에서 보관했다가 넘기지 않는다. 검사 없이 binding을 만드는 진입점이 아니다.
    """
    try:
        root = _root(directory)
        expected = {"preprocessor.json", "candidate.json", "candidate.md"}

        if committed:
            expected.add("AGENT_COMPLETE.json")

        if {path.name for path in root.iterdir()} != expected:
            return None

        for name in expected - {"AGENT_COMPLETE.json"}:
            _read_file(root / name, contract.max_candidate_bytes)

        payload = read_metadata(root / "preprocessor.json")
        report = read_metadata(root / "candidate.json")
        roles = json.loads(json.dumps(asdict(contract.train.roles)))
        rules = asdict(contract.rules)

        if set(payload) != {"schema_version", "roles", "rules", "state", "output_columns"} or payload["schema_version"] != "fitted_tabular_prepare_v1" or payload["roles"] != roles or payload["rules"] != rules:
            return None

        state = _validate_state(payload["state"], contract.train.roles, contract.rules)
        columns = [f"feature_{index}__{record['column']}__{offset}"
                   for index, record in enumerate(state["records"]) for offset in range(record["width"])]

        if state["train_rows"] != len(contract.train.row_positions) or payload["output_columns"] != columns:
            return None

        if state["train_rows"] * len(columns) > contract.rules.max_output_cells:
            return None

        removed_columns = {item["column"] for item in state["removed"]}
        retained_order = [name for name in contract.train.roles.feature_columns if name not in removed_columns]

        if retained_order != [record["column"] for record in state["records"]]:
            return None

        for record in state["records"]:
            if record["kind"] == "numeric" and contract.rules.scaling == "none" and (record["center"] != 0 or record["scale"] != 1):
                return None

            if record["kind"] == "categorical" and contract.rules.categorical_imputation == "missing_category" and record["fill"] != "missing":
                return None

        # 제거 사유는 활성 설정과 대응해야 한다. 통계의 정답을 부모가 재계산하지 않는다.
        for item in state["removed"]:
            if item["reason"] == "HIGH_MISSING_RATIO" and contract.rules.missing_ratio_threshold is None:
                return None

            if item["reason"] == "CONSTANT_OR_ALL_MISSING" and not contract.rules.drop_constant:
                return None

        expected_report = {
            "schema_version": "preparation_candidate_report_v1", "status": "candidate_ready",
            "binding": binding,
            "preprocessor_sha256": _fingerprint(root / "preprocessor.json")["sha256"],
            "removed": state["removed"], "retained": [record["column"] for record in state["records"]],
            "output_feature_count": len(columns), "transformed": False, "model_trained": False,
        }

        # v2는 처리 영향을 더한다. v1(영향 집계 이전 후보)은 이전 코드로 실행한 기록일 때만 받는다.
        # 현재 코드의 worker는 항상 v2를 쓰므로 현재 지문의 v1은 영향 검증을 피하려는 기록으로 보고 거부한다.
        typed = contract.schema_version == "preparation_candidate_v2"
        impact_version = "preparation_candidate_report_v3" if typed else "preparation_candidate_report_v2"
        if type(report) is dict and report.get("schema_version") == impact_version:
            if not _valid_impact(report.get("impact"), state, contract.rules):
                return None

            expected_report.update(schema_version=impact_version, impact=report["impact"])
        elif typed or code_id == _current_code_id():
            return None

        if report != expected_report or _read_file(root / "candidate.md", contract.max_candidate_bytes).decode("utf-8") != candidate_markdown(expected_report):
            return None

        return report
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        return None


def _current_code_id():
    from .preparation_candidate_transport import code_manifest
    from .snapshot_eda import _digest

    return _digest(code_manifest())


def _valid_impact(impact, state, rules):
    """worker가 Train에서 센 처리 영향의 형식·범위·순서와 저장 기준에서 결정되는 결과를 확인한다.

    건수의 정답은 원본 없이 다시 셀 수 없으므로 재계산하지 않는다. 다만 clip 상하한이 같거나 구간 경계가 없으면
    출력이 반드시 상수이므로 그 값은 대조한다. 형식이 하나라도 다르면 후보 전체를 받지 않는다.
    """
    rows = state["train_rows"]
    fields = {"column", "kind", "width", "missing", "imputed", "clipped", "distinct_before", "constant_after", "unknown", "missing_after"}

    if type(impact) is not list or len(impact) != len(state["records"]):
        return False

    for item, record in zip(impact, state["records"]):
        if type(item) is not dict or set(item) != fields or item["column"] != record["column"] or item["kind"] != record["kind"]:
            return False

        # bool·실수는 정수와 같게 비교되므로 형식을 먼저 엄격히 본다.
        if any(type(item[key]) is not int for key in ("width", "missing", "imputed", "distinct_before", "unknown", "missing_after")):
            return False

        if item["width"] != record["width"] or any(not 0 <= item[key] <= rows for key in ("missing", "imputed", "distinct_before", "unknown")):
            return False

        if type(item["constant_after"]) is not bool or item["imputed"] > item["missing"] or item["unknown"]:
            return False

        # 대치 여부로 대치 건수와 출력 결측 칸 수가 정해진다(범주는 항상 채우고, 대치 없는 수치 결측은 그대로 남는다).
        filled = record["kind"] == "categorical" or record.get("impute") is not None

        if item["imputed"] != (item["missing"] if filled else 0) or item["missing_after"] != (0 if filled else item["missing"]):
            return False

        clip = record.get("clip") if record["kind"] == "numeric" else None

        if (clip is None) != (item["clipped"] is None) or (item["clipped"] is not None and (type(item["clipped"]) is not int or not 0 <= item["clipped"] <= rows - item["missing"])):
            return False

        forced = record["kind"] == "numeric" and (record.get("edges") == [] or (clip is not None and clip[0] == clip[1] and record.get("edges") is None and item["missing"] in (0, item["imputed"])))

        if forced and not item["constant_after"]:
            return False

    return True

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
        contract.check()
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
            "binding": json.loads(json.dumps(contract.packet(code_id))),
            "preprocessor_sha256": _fingerprint(root / "preprocessor.json")["sha256"],
            "removed": state["removed"], "retained": [record["column"] for record in state["records"]],
            "output_feature_count": len(columns), "transformed": False, "model_trained": False,
        }

        if report != expected_report or _read_file(root / "candidate.md", contract.max_candidate_bytes).decode("utf-8") != candidate_markdown(expected_report):
            return None

        return report
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        return None

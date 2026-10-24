"""승인·커밋된 분할 증거의 읽기 전용 검사. 원본 CSV 파싱·새 분할·승인 발급은 하지 않는다."""

from contextlib import closing
from dataclasses import asdict, dataclass
import csv
from io import StringIO
import json
import math
from pathlib import Path
import sqlite3

from src.preprocessing.split import SplitRules, SPLITS, _digest
from .approvals import ApprovalScope
from .artifacts import ArtifactContext, check_artifacts, _fingerprint, _root
from .isolated_eda import _read_file
from .state import LocalStateStore
from .worker_paths import _path


MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_SPLIT_ROWS = 1_000_000


def read_metadata(path):
    """제한된 JSON 메타데이터만 읽는다. 중복 키·비표준 숫자는 거부한다."""
    def pairs(items):
        result = {}

        for key, value in items:
            if key in result:
                raise ValueError("분할 증거 JSON에 중복 키가 있습니다.")

            result[key] = value

        return result

    def invalid(value):
        raise ValueError("분할 증거에 비표준 숫자를 사용할 수 없습니다.")

    value = json.loads(_read_file(path, MAX_METADATA_BYTES).decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)

    if type(value) is not dict:
        raise ValueError("분할 증거는 JSON 객체여야 합니다.")

    return value


@dataclass(frozen=True)
class CommittedArtifact:
    """신뢰하는 호출자가 전달한 승인 범위와 현재 커밋 기록에 결합된 산출물."""

    run_directory: str
    task_id: str
    scope_json: str
    marker_sha256: str

    @property
    def scope(self):
        return ApprovalScope(**json.loads(self.scope_json))

    @property
    def directory(self):
        return Path(self.run_directory) / "attempts" / self.task_id

    def check(self):
        """계약 객체를 수동 생성했어도 실제 DB·파일 검사를 다시 통과해야 한다."""
        if read_committed_artifact(self.run_directory, self.scope) != self:
            raise ValueError("승인·커밋 산출물 증거가 변경되었습니다.")

        return True


def read_committed_artifact(run_directory, expected_scope):
    """DB는 mode=ro로 열고 승인 소비·예약·커밋·완료 해시를 함께 확인한다."""
    if not isinstance(expected_scope, ApprovalScope):
        raise TypeError("신뢰하는 실행 관리자가 보존한 ApprovalScope가 필요합니다.")

    allowed = (
        ("commit_split", "commit_split"), ("ingest", "run_ingest"),
        ("derive_preparation", "derive_preparation"), ("run_eda", "run_eda"),
    )
    if expected_scope.policy_version == "tabular_split_v2":
        # v2 확정 증거는 실제로 승인·커밋된 선행 제안의 완료 표식에도 결합된다.
        allowed += (("propose_split", "propose_split"),)
    if (expected_scope.stage, expected_scope.tool_id) not in allowed:
        raise ValueError("분할·내보내기·후보 산출·승인된 EDA 작업의 증거가 필요합니다.")

    root = _root(run_directory)
    state_path = _path(root / "state.json", directory=False)
    before = _fingerprint(state_path)
    state = LocalStateStore(root).load()

    if (
        state.run_id != expected_scope.run_id or state.actor_id != expected_scope.actor_id
        or state.cancel_requested or "SECURITY_LOCKED" in state.blockers
        or expected_scope.target_id != f"{state.run_id}:{expected_scope.stage}"
    ):
        raise ValueError("분할 증거의 실행 주체·작업 범위 또는 상태가 일치하지 않습니다.")

    database = _path(root / "approvals.sqlite3", directory=False)
    scope_json = json.dumps(asdict(expected_scope), ensure_ascii=False, sort_keys=True)

    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        task = connection.execute(
            "SELECT task_id,status FROM task_runs WHERE scope_json=?", (scope_json,),
        ).fetchone()
        approval = connection.execute(
            "SELECT r.reservation_id,a.status FROM reservations r JOIN approvals a ON a.approval_id=r.approval_id "
            "WHERE r.scope_json=? AND a.scope_json=r.scope_json", (scope_json,),
        ).fetchall()

    if task is None or task[1] != "committed" or approval != [(task[0], "consumed")]:
        raise ValueError("승인 소비·예약·커밋된 작업 기록이 필요합니다.")

    # DB 내용도 신뢰하지 않고 내부 작업 ID 형식을 확인한 뒤 경로를 만든다.
    if type(task[0]) is not str or len(task[0]) != 32 or any(char not in "0123456789abcdef" for char in task[0]):
        raise ValueError("내부 작업 ID 형식이 올바르지 않습니다.")

    output = root / "attempts" / task[0]
    context = ArtifactContext(
        expected_scope.source_id, expected_scope.config_id, expected_scope.plan_id,
        expected_scope.policy_version, expected_scope.tool_id,
    )

    if not check_artifacts(output, context).reusable or _fingerprint(state_path) != before:
        raise ValueError("완료 산출물 또는 실행 상태의 정합성이 일치하지 않습니다.")

    return CommittedArtifact(str(root), task[0], scope_json, _fingerprint(output / "AGENT_COMPLETE.json")["sha256"])


@dataclass(frozen=True)
class ConfirmedSplit:
    """기존 split_proposal_v1의 위치 배정과 승인된 완료 산출물 연결."""

    artifact: CommittedArtifact
    proposal_json: str

    @property
    def proposal(self):
        return json.loads(self.proposal_json)

    @property
    def signature(self):
        return _digest(asdict(self))

    def positions(self, partition):
        if partition not in SPLITS:
            raise ValueError("등록된 분할 구간 이름이 필요합니다.")

        return tuple(self.proposal["partitions"][partition])

    def check(self):
        if read_confirmed_split(self.artifact.run_directory, self.artifact.scope) != self:
            raise ValueError("확정 분할 증거가 변경되었습니다.")

        return True


def read_split_files(directory, source_id, config_id, *, preview=False, committed=False, allow_review=False, expected_rows=None):
    """제안·확정 파일의 제한된 구조 검사. 승인 증거 및 원본 재계산을 대신하지 않는다."""
    root = _root(directory)
    names = {"proposal.json", "proposal.md"} if preview else {"split.json", "assignments.csv"}

    if committed:
        names.add("AGENT_COMPLETE.json")

    if {path.name for path in root.iterdir()} != names:
        raise ValueError("분할 산출물 파일 목록이 계약과 다릅니다.")

    proposal = read_metadata(root / ("proposal.json" if preview else "split.json"))
    required = {"schema_version", "source_id", "config_id", "rules", "partitions", "summary", "status", "warnings", "proposal_id"}

    statuses = {"passed", "review_required"} if preview and allow_review else {"passed"}

    if set(proposal) != required or proposal["schema_version"] != "split_proposal_v1" or proposal["status"] not in statuses:
        raise ValueError("통과·확정된 기존 분할 계약이 필요합니다.")

    if proposal["source_id"] != source_id or proposal["config_id"] != config_id:
        raise ValueError("분할의 원본·설정 식별값이 승인 범위와 다릅니다.")

    base = {key: value for key, value in proposal.items() if key != "proposal_id"}

    if _digest(base) != proposal["proposal_id"]:
        raise ValueError("분할 제안 식별값이 일치하지 않습니다.")

    if type(proposal["rules"]) is not dict or set(proposal["rules"]) != set(SplitRules.__dataclass_fields__):
        raise ValueError("분할 규칙 필드가 올바르지 않습니다.")

    SplitRules(**proposal["rules"])
    partitions = proposal["partitions"]

    if type(partitions) is not dict or set(partitions) != set(SPLITS):
        raise ValueError("Train·Validation·Test 배정 목록이 필요합니다.")

    flattened = []

    for name in SPLITS:
        rows = partitions[name]

        if type(rows) is not list or not rows or len(rows) > MAX_SPLIT_ROWS:
            raise ValueError("분할 구간은 제한된 비어 있지 않은 위치 목록이어야 합니다.")

        if any(type(row) is not int or row < 0 for row in rows) or rows != sorted(set(rows)):
            raise ValueError("분할 위치는 중복 없는 정렬된 음이 아닌 정수여야 합니다.")

        flattened.extend(rows)

    count = len(flattened)

    if expected_rows is not None and (type(expected_rows) is not int or count != expected_rows):
        raise ValueError("분할 배정 행 수가 승인된 입력 행 수와 다릅니다.")

    if count > MAX_SPLIT_ROWS or sorted(flattened) != list(range(count)):
        raise ValueError("분할 위치에 중복·누락이 있거나 메타데이터 한도를 초과했습니다.")

    summary = proposal["summary"]

    if type(summary) is not list or len(summary) != 3 or type(proposal["warnings"]) is not list:
        raise ValueError("분할 요약·경고의 형식이 올바르지 않습니다.")

    if any(warning != "CLASS_MISSING_IN_SPLIT" for warning in proposal["warnings"]):
        raise ValueError("지원하지 않는 분할 경고입니다.")

    for name, row in zip(SPLITS, summary):
        if type(row) is not dict or not {"split", "rows", "ratio", "classes"}.issubset(row):
            raise ValueError("분할 구간 요약 필드가 올바르지 않습니다.")

        if set(row) - {"split", "rows", "ratio", "classes", "group_count", "time_min", "time_max"}:
            raise ValueError("분할 구간 요약에 미지원 필드가 있습니다.")

        if row["split"] != name or type(row["rows"]) is not int or row["rows"] != len(partitions[name]):
            raise ValueError("분할 구간 요약의 행 수가 일치하지 않습니다.")

        if type(row["ratio"]) not in (int, float) or not math.isclose(row["ratio"], row["rows"] / count, abs_tol=1e-12):
            raise ValueError("분할 구간 요약의 비율이 일치하지 않습니다.")

        if type(row["classes"]) is not list:
            raise ValueError("분할 클래스 요약은 목록이어야 합니다.")

        total = 0

        for item in row["classes"]:
            if type(item) is not dict or set(item) != {"label", "count"} or type(item["count"]) is not int or item["count"] < 1:
                raise ValueError("분할 클래스 요약이 올바르지 않습니다.")

            total += item["count"]

        if row["classes"] and total != row["rows"]:
            raise ValueError("분할 클래스별 건수 합이 구간 행 수와 다릅니다.")

    expected = [None] * count

    for name in SPLITS:
        for position in partitions[name]:
            expected[position] = name

    if preview:
        _read_file(root / "proposal.md", 65536).decode("utf-8")
        return proposal

    rows = csv.reader(StringIO(_read_file(root / "assignments.csv", MAX_METADATA_BYTES).decode("utf-8-sig")), strict=True)

    if next(rows, None) != ["row_position", "split"]:
        raise ValueError("행 배정표의 헤더가 올바르지 않습니다.")

    actual_count = 0

    for index, row in enumerate(rows):
        if index >= count or len(row) != 2 or row != [str(index), expected[index]]:
            raise ValueError("행 배정표가 확정 분할과 일치하지 않습니다.")

        actual_count += 1

    if actual_count != count:
        raise ValueError("행 배정표에 누락이 있습니다.")

    return proposal


def read_confirmed_split(run_directory, expected_scope):
    """승인·커밋 증거와 구조 검사 결합. 그룹/시간 경계는 원본에서 재계산하지 않는다."""
    if not isinstance(expected_scope, ApprovalScope) or expected_scope.tool_id != "commit_split":
        raise ValueError("분할 확정 승인의 정확한 범위가 필요합니다.")

    artifact = read_committed_artifact(run_directory, expected_scope)
    proposal = read_split_files(artifact.directory, expected_scope.source_id, expected_scope.config_id, committed=True)
    if expected_scope.policy_version == "tabular_split_v2":
        from .tabular_source_contract import TabularEdaInputContract
        from .executor import _digest as arguments_digest, _json as arguments_json
        record = read_metadata(Path(run_directory) / "selection.json")
        if set(record) != {"contract", "code_id", "plan_id"}:
            raise ValueError("v2 분할 선택 기록의 필드가 다릅니다.")
        contract = TabularEdaInputContract.from_dict(record["contract"])
        contract.check_input()
        if sum(len(values) for values in proposal["partitions"].values()) != contract.source.bundle.rows:
            raise ValueError("확정 분할에 승인 입력의 행이 누락됐습니다.")
        arguments = {"version": "tabular_split_v2", "source": contract.source.packet(), "profile": contract.profile.to_dict(),
                     "limits": asdict(contract.limits), "code_id": record["code_id"], "proposal_sha256": None}
        from dataclasses import replace
        preview_scope = replace(expected_scope, stage="propose_split", tool_id="propose_split",
                                target_id=expected_scope.run_id + ":propose_split", arguments_id=arguments_digest(arguments_json(arguments)))
        preview = read_committed_artifact(run_directory, preview_scope)
        bound_arguments = {"arguments": arguments, "dependency": _fingerprint(preview.directory / "AGENT_COMPLETE.json")}
        if (expected_scope.source_id != contract.source.source_id or expected_scope.config_id != contract.profile.signature
                or expected_scope.plan_id != record["plan_id"] or expected_scope.arguments_id != arguments_digest(arguments_json(bound_arguments))
                or proposal["rules"] != asdict(contract.profile.split_rules)
                or proposal != read_split_files(preview.directory, expected_scope.source_id, expected_scope.config_id, preview=True, committed=True)):
            raise ValueError("v2 분할 승인 범위와 입력·역할·계획이 다릅니다.")
    artifact.check()

    return ConfirmedSplit(artifact, json.dumps(proposal, ensure_ascii=False, sort_keys=True, allow_nan=False))

"""공통 작업 조정기 + 표형 어댑터: 요청 → 진단 → 로컬 LLM 권장안 → 질문·계획 확인 → 전처리 → 통합 결과를 합성 CSV로 검사한다.

OS transport(AppContainer·Job)와 로컬 모델 응답만 모의한다. 진단·CSV 작업 계산·승인 저장·결과 검증은 실제 코드다.
"""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents import task_adapters, task_consent, task_orchestrator as orchestrator, task_store
from src.agents.llm_config import load_local_config
from src.agents.llm_task_assist import PREPARATION_FIELDS, SHARED_FIELDS
from src.agents.local_ollama import LocalProposalError
from src.agents.profile_store import SPLIT_FIELDS
from src.agents.task_plan import PlanError
from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules
from tests.agents import test_csv_workflow_service as csv_fixtures
from tests.agents.test_input_profile_pipeline import _transport as diagnostic_transport


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"wall_seconds": 60, "memory_mib": 2048, "processes": 2}
SPLIT = {key: value for key, value in asdict(SplitRules()).items() if key in SPLIT_FIELDS}
PREPARATION = {key: value for key, value in asdict(PreparationRules()).items() if key in PREPARATION_FIELDS}
ROLES = {"x": "수치형", "label": "target", "id": "식별자", "time": "제외"}


def plan(roles=ROLES, split=SPLIT, preparation=PREPARATION):
    return {"roles": dict(roles), "split": dict(split), "preparation": dict(preparation)}


def model_reply(roles=ROLES, changes=(), questions=()):
    """기본 계획 대비 변경만 내는 권장안. 역할은 모든 컬럼을 경로 단위 변경으로 표현한다."""
    items = [{"path": "roles." + name, "value": role, "reason": "의미에 맞는 역할"} for name, role in roles.items()]
    return json.dumps({"summary": ["120행, 결측 없음"], "notes": ["label을 target으로 두었습니다."],
                       "changes": items + list(changes), "questions": list(questions)}, ensure_ascii=False)


def request_draft(record, requester):
    """조정기와 같은 경로로 권장안을 다시 요청한다(어댑터 검증·병합 포함)."""
    from src.agents import task_recommend

    adapter = task_adapters.adapter(record["kind"])
    return task_recommend.request(record["goal"], adapter.recommendation_input(record), adapter.base_plan(record),
                                  adapter.plan_fields(record), lambda value: adapter.validate_plan(record, value), load_local_config(),
                                  requester=requester, on_rejected=lambda draft: adapter.on_rejected(record, draft))


def wide_frame(rows=200, sensors=590):
    """SECOM형 합성 넓은 표: 수치 센서 590개(일부 상수·결측), 시간 문자열, Pass/Fail(-1/1). 실제 데이터가 아니다."""
    import pandas as pd

    data = {}

    for column in range(sensors):
        if column % 20 == 0:
            values = [7.0] * rows
        else:
            values = [((index * (column + 3)) % 97) / 7.0 + column for index in range(rows)]

            if column % 7 == 0:
                values = [None if index % 9 == 0 else value for index, value in enumerate(values)]

        data[str(column)] = values

    data["Time"] = [f"2026-01-01 {index // 60:02d}:{index % 60:02d}:00" for index in range(rows)]
    data["Pass/Fail"] = [1 if index % 5 == 0 else -1 for index in range(rows)]
    return pd.DataFrame(data)


class TaskOrchestratorTest(unittest.TestCase):
    setUp_csv = csv_fixtures.CsvWorkflowIntegrationTest.setUp
    start_patch = csv_fixtures.CsvWorkflowIntegrationTest.start_patch
    transport = csv_fixtures.CsvWorkflowIntegrationTest.transport

    def setUp(self):
        self.setUp_csv()
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.store = Path(temp.name) / "tasks"
        (Path(temp.name) / "diag_runs").mkdir()
        self.start_patch("src.agents.pipeline_approval_cli._directory",
                         side_effect=lambda name, create_parent=False: Path(temp.name) / "diag_runs" / name)
        self.start_patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=diagnostic_transport)
        self.model = asdict(load_local_config())
        self.requests = []

    def requester(self, payload, config):
        self.requests.append(payload)
        return model_reply()

    def new_task(self, source=None):
        return orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", source or self.source, LIMITS, self.model,
                                       root=self.store)

    def run_to_apply(self, roles=ROLES):
        task = self.new_task()
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        return orchestrator.confirm_plan(task["task_id"], plan(roles), root=self.store)

    def test_request_to_integrated_result_with_four_decision_kinds(self):
        before = self.source.read_bytes()
        task = self.new_task()
        self.assertEqual((task["phase"], task["kind"]), ("created", "tabular_csv"))
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_review")
        self.assertEqual(task["draft"]["source"], "model")
        self.assertEqual(task["draft"]["plan"]["roles"], ROLES)
        self.assertEqual(task["diagnostic"]["summary"]["row_count"], 120)

        task = orchestrator.confirm_plan(task["task_id"], plan(), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        # 결정 D 대상은 검증된 후보 파일에서 만든 구조화된 표로 보여 준다.
        review = {item["label"]: item["value"] for item in task["candidate"]["facts"]["metrics"]}
        self.assertEqual(review["유지 입력 특징"], 1)
        self.assertIn("특징별 처리", [table["title"] for table in task["candidate"]["facts"]["tables"]])
        self.assertEqual(set(task["steps"]), {"split", "export", "candidate"})
        self.assertEqual(self.calls.count("apply_preparation"), 0)

        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertEqual(self.calls.count("apply_preparation"), 1)
        self.assertTrue(Path(task["report"]["path"]).is_file())
        self.assertIn("로컬 전달 묶음", task["report"]["text"])
        facts = task["report"]["facts"]

        # 검증된 결과 파일에서 재계산 없이 조립한다. 실측(적용 결과)과 예상(후보)을 나눈다(2026-10-22, 설계 7.22).
        self.assertEqual({item["label"]: item["value"] for item in facts["metrics"]}["입력 행"], 120)
        comparison = {row["항목"]: row for row in facts["comparison"]}
        applied = {name: int(text.split()[0][:-1]) for name, text in
                   (part.split(" ", 1) for part in comparison["입력 행"]["후 (실측)"].split(" · "))}
        self.assertEqual(set(applied), {"train", "validation"})
        # 변환 후 결측 칸은 적용 검증기가 출력 파일로 대조한 실측치다(이전에는 미집계).
        self.assertEqual(comparison["결측 칸"]["후 (실측)"], "train 0 · validation 0")
        self.assertEqual(comparison["test 구간"]["후 (실측)"], "변환 적용 안 함")
        self.assertTrue(comparison["train 구간"]["후 (실측)"].endswith("행 변환"))
        self.assertEqual({row["컬럼"]: row["역할"] for row in facts["processing"]}["label"], "target")
        self.assertTrue(facts["images"])
        # 2026-10-22부터 Test 미적용은 경고 절의 '미처리'로 분류한다.
        self.assertIn({"구분": "미처리", "내용": "Test 구간에는 변환을 적용하지 않았습니다."}, facts["warnings"])
        self.assertTrue(any(row["파일"] == "prepared/train_features.csv" for row in facts["files"]))
        body = task["report"]["text"].split("## 기술 상세")[0]

        for detail in (task["snapshot"]["snapshot"]["sha256"], task["steps"]["split"]["run_name"], str(self.root)):
            self.assertNotIn(detail, body)

        # 보고서 재조립은 새 파일만 만들고 기존 보고서·결과를 바꾸지 않는다.
        from src.agents.task_report import rebuild_report, verified_images

        first = Path(task["report"]["path"]).read_bytes()
        stages = list(self.calls)
        rebuilt = rebuild_report(task["task_id"], root=self.store)
        self.assertNotEqual(rebuilt["report"]["path"], task["report"]["path"])
        self.assertEqual(Path(task["report"]["path"]).read_bytes(), first)
        self.assertEqual(self.calls, stages)

        # 기록의 작업 이름·모듈을 바꿔도 다른 결과를 섞지 않는다.
        path = self.store / task["task_id"] / "task.json"

        for key, value in (("run_name", "other-export"), ("module_id", "csv_prepare_apply")):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["steps"]["export"][key] = value
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

            with self.subTest(key=key), self.assertRaises(orchestrator.TaskProblem):
                rebuild_report(task["task_id"], root=self.store)

            with self.subTest(key=key), self.assertRaises(orchestrator.TaskProblem):
                verified_images(task_store.load_task(task["task_id"], root=self.store))

            path.write_text(json.dumps(rebuilt, ensure_ascii=False), encoding="utf-8")

        self.assertTrue(verified_images(rebuilt))
        from src.agents.csv_handoff_contract import verify_bundle

        self.assertEqual(verify_bundle(Path(task["report"]["bundle"]))["status"], "handoff_verified")
        self.assertEqual(self.source.read_bytes(), before)

        # 승인 위임: 단계별 1회 승인을 발급했고 직접 확인 이벤트로 기록하지 않았다.
        self.assertEqual(len(task["steps"]["split"]["delegations"]), 5)
        self.assertEqual({key for key in task["consents"]}, {"A", "C", "D"})
        for step in task["steps"].values():
            for delegation in step["delegations"].values():
                self.assertEqual(delegation["consent_id"], task["consents"][step["decision"]]["consent_id"])
        events = "".join(path.read_text(encoding="utf-8") for path in self.root.rglob("events.jsonl"))
        events += "".join(path.read_text(encoding="utf-8") for path in self.store.parent.rglob("events.jsonl"))
        self.assertIn("DELEGATED_APPROVAL_RECORDED", events)
        self.assertNotIn("LOCAL_REVIEW_CONFIRMED", events)
        self.assertNotIn("PIPELINE_APPROVAL_RECORDED", events)
        # 위임 범위는 확정 계획의 단계와 카탈로그 승인 단계에서 계산해 동의 내용에 고정한다.
        self.assertIn(["csv_split_review", "review_eda"], task["consents"]["C"]["content"]["allowed"])
        self.assertNotIn(["csv_prepare_apply", "apply_preparation"], task["consents"]["C"]["content"]["allowed"])
        self.assertIn(["csv_prepare_apply", "apply_preparation"], task["consents"]["D"]["content"]["allowed"])

        # 작업 기록에 승인 ID·대화 원문을 저장하지 않는다.
        stored = (self.store / task["task_id"] / "task.json").read_text(encoding="utf-8")
        self.assertNotIn("approval_id", stored)

    def test_computed_delegation_scope_matches_previous_fixed_table(self):
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        previous = {
            "C": {("csv_split_review", stage) for stage in ("audit", "run_eda", "review_eda", "propose_split", "commit_split")}
                 | {("csv_partition_export", "ingest"), ("csv_prepare_candidate", "derive_preparation")},
            "D": {("csv_prepare_apply", "review_candidate"), ("csv_prepare_apply", "apply_preparation"),
                  ("csv_local_handoff", "ingest")},
        }

        for decision, expected in previous.items():
            self.assertEqual({tuple(item) for item in orchestrator._allowed(task, decision)}, expected)

    def test_model_payload_has_only_names_aggregates_and_base_plan(self):
        from src.agents import llm_task_assist, task_recommend

        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        content = self.requests[0]["messages"][1]["content"]
        payload = json.loads(content)
        self.assertEqual(set(payload), {"goal", "row_count", "repeated_rows", "subjects", "base_plan", "allowed_values", "scope_note"})
        # 허용 값은 코드 상수(선택지·범위)만 담는다. 컬럼별 역할은 하나로 묶는다.
        allowed = payload["allowed_values"]
        self.assertEqual(allowed["preparation.outlier"], ["none", "iqr_clip"])
        self.assertEqual(allowed["roles.*"], ["제외", "수치형", "범주형", "식별자", "target", "group", "timestamp"])
        self.assertNotIn("time", allowed["split.strategy"])
        self.assertFalse(any(key.startswith("roles.") and key != "roles.*" for key in allowed))
        # 응답 형식은 경로마다 값의 형태를 고정한다. 응답 뒤 검증 규칙과 같다.
        options = {}

        for item in self.requests[0]["format"]["properties"]["changes"]["items"]["anyOf"]:
            path = item["properties"]["path"]

            for name in path.get("enum", [path.get("const")]):
                options[name] = item["properties"]["value"]

        self.assertEqual(options["roles.x"], {"enum": ["제외", "수치형", "범주형", "식별자", "target", "group", "timestamp"]})
        self.assertEqual(options["split.random_state"], {"type": "integer", "minimum": 0, "maximum": 4294967295})
        self.assertEqual(options["split.stratify"], {"type": "boolean"})
        self.assertEqual(options["preparation.outlier"], {"enum": ["none", "iqr_clip"]})
        self.assertEqual(options["preparation.missing_ratio_threshold"],
                         {"anyOf": [{"type": "number", "minimum": 0, "maximum": 1}, {"type": "null"}]})
        # 작은 표는 모든 컬럼을 상세 집계로 보낸다. 상세 집계 항목은 공유 허용 필드뿐이다.
        subjects = payload["subjects"]
        self.assertEqual(subjects["detail_fields"], ["id", *[name for name in SHARED_FIELDS if name != "column"], "reasons"])
        self.assertEqual((len(subjects["detailed"]), subjects["named"], subjects["summarized"]["count"]), (4, [], 0))
        self.assertEqual(task["draft"]["coverage"]["summarized"], 0)
        self.assertEqual(set(payload["base_plan"]), {"roles", "split", "preparation"})
        self.assertEqual(set(payload["base_plan"]["roles"]), {"by_value"})

        for forbidden in ("snapshot", "input.bin", json.dumps(str(self.source))[1:-1], "\"min\"", "q25", "iqr_lower"):
            self.assertNotIn(forbidden, content)

        # 지시문은 코드 상수다. 입력 자료는 user 메시지에만 있다.
        self.assertEqual(self.requests[0]["messages"][0],
                         {"role": "system", "content": task_recommend.COMMON_INSTRUCTIONS + llm_task_assist.instructions()})
        self.assertEqual(len(self.requests[0]["messages"]), 2)
        self.assertEqual(task["phase"], "awaiting_review")

    def test_request_stays_within_token_budget_and_too_large_is_not_sent(self):
        """요청은 모델 문맥 예산 안으로 배치한다. 필수 상세만으로도 넘치면 보내지 않고 별도 코드로 남긴다."""
        from dataclasses import replace

        from src.agents import task_recommend

        def record(count, name="feature_{:03d}"):
            base = {key: None for key in SHARED_FIELDS}
            # 50번째 컬럼마다 값 종류 3개(target 후보, 상세 우선)를 둔다.
            columns = [dict(base, column=name.format(index), dtype_group="numeric", missing_count=0, missing_ratio=0.0,
                            observed_unique=3 if index % 50 == 0 else 50, finite_count=100, mean=1.0, std=0.5)
                       for index in range(count)]
            return {"kind": "tabular_csv", "goal": "예측",
                    "diagnostic": {"summary": {"row_count": 100, "repeated_rows": 0, "column_count": count}, "columns": columns}}

        sent = []
        draft = request_draft(record(600), lambda payload, config: sent.append(payload) or "응답 아님")
        tokens = task_recommend._messages_bytes(sent[0]) / task_recommend.BYTES_PER_TOKEN
        self.assertLessEqual(tokens, task_recommend.token_budget(load_local_config()))
        coverage = draft["coverage"]
        self.assertEqual(coverage["total"], 600)
        self.assertEqual(coverage["detailed"] + coverage["named"] + coverage["summarized"], 600)
        # 값 종류가 2~20개인 컬럼(12개)은 상세 우선(우선순위 2)이라 예산이 모자라도 먼저 상세로 간다.
        detailed_ids = {row[0] for row in json.loads(sent[0]["messages"][1]["content"])["subjects"]["detailed"]}
        self.assertTrue({f"feature_{index:03d}" for index in range(0, 600, 50)} <= detailed_ids)
        self.assertEqual(len(detailed_ids), coverage["detailed"])

        # 문맥이 아주 작으면 모델에 보내지 않는다(조용히 잘라 보내지 않음).
        calls = []

        with patch("src.agents.llm_config.LocalModelConfig.__post_init__", lambda self: None):
            small = replace(load_local_config(), num_ctx=1024, num_predict=512)

        from src.agents import task_recommend as module
        adapter = task_adapters.adapter("tabular_csv")
        value = record(600)
        draft = module.request("예측", adapter.recommendation_input(value), adapter.base_plan(value), adapter.plan_fields(value),
                               lambda plan: adapter.validate_plan(value, plan), small, requester=lambda payload, config: calls.append(1))
        self.assertEqual((draft["source"], draft["code"], calls), ("default", "RECOMMENDATION_REQUEST_TOO_LARGE", []))

    def test_wide_table_from_record_reload_to_recommendation_and_plan(self):
        """합성 SECOM형 넓은 표(592컬럼): 64 KiB를 넘는 작업 기록 재조회 → 예산 안 권장안 → 검토 범위 기록 → 계획 확인·후보까지."""
        source = self.root / "wide.csv"
        wide_frame().to_csv(source, index=False)
        sent = []

        def requester(payload, config):
            sent.append(payload)
            return model_reply({"Pass/Fail": "target", "Time": "식별자"})

        task = orchestrator.start_task("Pass/Fail 예측용 전처리", source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        # 작업 기록이 worker IPC 상한(64 KiB)보다 크지만 다시 읽힌다. 큰 진단 구역은 해시 참조로 분리 저장된다.
        folder = self.store / task["task_id"]
        size = (folder / "task.json").stat().st_size + sum(path.stat().st_size for path in (folder / "parts").iterdir())
        self.assertGreater(size, 64 * 1024)
        self.assertIn("$part", json.loads((folder / "task.json").read_text(encoding="utf-8"))["diagnostic"])
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["diagnostic"]["summary"]["column_count"], 592)
        # 권장안은 예산 안에서 실제로 보냈고, 모델이 본 범위가 기록된다. 요약 대상 역할은 모델 허용 항목에서 빠진다.
        from src.agents import task_recommend

        self.assertLessEqual(task_recommend._messages_bytes(sent[0]) / task_recommend.BYTES_PER_TOKEN,
                             task_recommend.token_budget(load_local_config()))
        coverage = task["draft"]["coverage"]
        self.assertEqual((task["draft"]["source"], coverage["total"]), ("model", 592))
        self.assertGreater(coverage["summarized"], 0)
        summarized = coverage["summarized_ids"][0]
        paths = {name for item in sent[0]["format"]["properties"]["changes"]["items"]["anyOf"]
                 for name in item["properties"]["path"].get("enum", [item["properties"]["path"].get("const")])}
        self.assertNotIn("roles." + summarized, paths)
        self.assertIn("roles.Pass/Fail", paths)
        self.assertEqual(task["draft"]["plan"]["roles"]["Pass/Fail"], "target")

        # 요약 대상 역할을 바꾸자는 응답은 권장안 전체를 거부한다. 사용자는 같은 항목을 계획에서 직접 바꿀 수 있다.
        draft = request_draft(task, lambda payload, config: model_reply({"Pass/Fail": "target", summarized: "제외"}))
        self.assertEqual((draft["source"], draft["code"]), ("default", "RECOMMENDATION_CHANGE_INVALID"))
        edited = json.loads(json.dumps(task["draft"]["plan"]))
        edited["roles"][summarized] = "제외"
        final, _ = orchestrator.check_plan(task, edited, {})
        self.assertEqual(final["roles"][summarized], "제외")

        task = orchestrator.confirm_plan(task["task_id"], edited, {}, root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        # 결과 파일도 64 KiB를 넘을 수 있다. 결정 D 표의 수치가 미집계로 바뀌지 않는다.
        review = {item["label"]: item["value"] for item in task["candidate"]["facts"]["metrics"]}
        self.assertNotEqual(review["유지 입력 특징"], "미집계")
        self.assertNotEqual(review["적용 후 예상 출력 특징"], "미집계")
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["phase"], "awaiting_apply")

    def test_numeric_identifier_is_asked_but_continuous_measurement_is_not(self):
        from src.agents import llm_task_assist

        summary = {"row_count": 6, "repeated_rows": 0}
        base = {"missing_count": 0, "dtype_group": "numeric", "observed_unique": 6}
        columns = [dict(base, column="PassengerId", min=1.0, max=900.0),
                   dict(base, column="row_seq_free", min=101.0, max=106.0),
                   dict(base, column="Fare", min=1.25, max=512.3),
                   dict(base, column="label", observed_unique=2, min=0.0, max=1.0)]
        roles = {"PassengerId": "수치형", "row_seq_free": "수치형", "Fare": "수치형", "label": "target"}
        current = {"roles": roles, "split": SPLIT, "preparation": PREPARATION}
        questions = llm_task_assist.plan_questions("label 예측", summary, columns, current)
        asked = [item["subjects"] for item in questions if item["id"].startswith("identifier:")]
        # 이름 단서(PassengerId)와 연속 정수 범위(101~106)는 묻고, 값이 모두 다른 연속형 측정값(Fare)은 묻지 않는다.
        self.assertEqual(asked, [["PassengerId"], ["row_seq_free"]])
        self.assertTrue(all(item["level"] == "advisory" and item["recommended"] == "식별자"
                            for item in questions if item["id"].startswith("identifier:")))
        # 역할은 자동으로 바꾸지 않는다.
        self.assertEqual(current["roles"]["PassengerId"], "수치형")

    def test_conflicting_answers_name_the_conflicting_items(self):
        """답변 하나씩은 허용 값이어도 함께 쓸 수 없는 조합이면, 충돌하는 항목을 알려 주고 아무것도 만들지 않는다."""
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=lambda payload, config: model_reply(changes=[], questions=[
            {"path": "preparation.binning", "text": "구간화할까요?", "options": ["quantile", "none"]},
            {"path": "preparation.scaling", "text": "스케일링할까요?", "options": ["standard", "none"]}]))
        asked = {item["path"]: item["id"] for item in orchestrator.review_state(task)["questions"] if item["source"] == "model"}
        answers = {asked["preparation.binning"]: "quantile", asked["preparation.scaling"]: "standard"}

        for answer in answers.items():
            orchestrator.check_plan(task, plan(), dict([answer]))

        with self.assertRaises(PlanError) as caught:
            orchestrator.confirm_plan(task["task_id"], plan(), answers, root=self.store)

        self.assertEqual((caught.exception.code, caught.exception.subjects),
                         ("BINNING_WITH_SCALING", ("preparation.binning", "preparation.scaling")))
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["steps"], {})

    def test_numeric_identifier_question_appears_in_base_plan_without_model(self):
        import pandas as pd

        frame = pd.read_csv(self.source)
        frame.insert(0, "PassengerId", range(1, len(frame) + 1))
        source = self.root / "with_id.csv"
        frame.to_csv(source, index=False)
        task = orchestrator.start_task("label 예측", source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=lambda payload, config: "응답 아님")
        self.assertEqual(task["draft"]["source"], "default")
        asked = [item["subjects"] for item in orchestrator.review_state(task)["questions"] if item["id"].startswith("identifier:")]
        self.assertIn(["PassengerId"], asked)

    def test_processing_failure_records_exception_type_and_stage_without_message(self):
        from src.agents import tabular_task

        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        original = tabular_task.create_step

        def failing(record, directory, step_id, name):
            if step_id == "export":
                raise RuntimeError("민감한 원문 " + str(self.root))

            return original(record, directory, step_id, name)

        with patch.object(tabular_task, "create_step", side_effect=failing):
            task = orchestrator.confirm_plan(task["task_id"], plan(), root=self.store)

        problem = task["problem"]
        self.assertEqual(problem["code"], "PROCESSING_FAILED")
        self.assertEqual(problem["error"], {"type": "RuntimeError", "cause_type": None, "stage": "export", "step_status": "reserved"})
        self.assertNotIn("민감한", json.dumps(task, ensure_ascii=False))

    def test_degenerate_columns_do_not_reveal_individual_values(self):
        from src.agents.llm_task_assist import shared_columns

        base = {name: None for name in SHARED_FIELDS}
        columns = [dict(base, column="const", observed_unique=1, finite_count=10, mean=73125.5, std=0.0),
                   dict(base, column="two", observed_unique=2, finite_count=2, mean=20.0, std=14.14),
                   dict(base, column="many", observed_unique=9, finite_count=9, mean=5.0, std=2.0)]
        shared = {item["column"]: item for item in shared_columns(columns)}
        self.assertIsNone(shared["const"]["mean"])
        self.assertIsNone(shared["two"]["mean"])
        self.assertIsNone(shared["two"]["std"])
        self.assertEqual(shared["many"]["mean"], 5.0)

    def test_model_failure_falls_back_to_base_plan_without_retry(self):
        calls = []

        def failing(payload, config):
            calls.append(1)
            raise LocalProposalError("OLLAMA_CONNECTION_FAILED")

        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=failing)
        self.assertEqual(len(calls), 1)
        self.assertEqual((task["draft"]["source"], task["draft"]["code"]), ("default", "OLLAMA_CONNECTION_FAILED"))
        self.assertEqual(task["draft"]["plan"], task_adapters.adapter("tabular_csv").base_plan(task))
        self.assertEqual(task["phase"], "awaiting_review")

    def test_recommendation_is_rejected_as_a_whole_with_fixed_code(self):
        from src.agents.task_report import code_label

        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        base = task_adapters.adapter("tabular_csv").base_plan(task)
        reply = json.loads(model_reply())
        cases = {
            "RECOMMENDATION_RESPONSE_INVALID": "응답 아님",
            "RECOMMENDATION_FIELDS_INVALID": json.dumps({key: value for key, value in reply.items() if key != "notes"}),
            "ROLES_INVALID": model_reply(dict(ROLES, id="target")),
            "SPLIT_INVALID": model_reply(changes=[{"path": "split.test_ratio", "value": 0.25, "reason": "r"}]),
            "RECOMMENDATION_CHANGE_INVALID": model_reply(changes=[{"path": "preparation.scaling", "value": "magic", "reason": "r"}]),
            "RECOMMENDATION_TEXT_INVALID": json.dumps(dict(reply, notes=["가" * 400])),
            "SPLIT_ROLE_MISMATCH": model_reply(changes=[{"path": "split.strategy", "value": "group", "reason": "r"}]),
        }

        for code, response in cases.items():
            draft = request_draft(task, lambda payload, config, response=response: response)
            # 한 항목이라도 검증에 실패하면 권장 변경 전체를 채택하지 않는다(역할 변경도 버린다).
            self.assertEqual((draft["source"], draft["code"], draft["changes"]), ("default", code, []), code)
            self.assertEqual(draft["plan"], base)
            self.assertIsNotNone(code_label(task, code))

        # 거부 사유는 코드만 남는다. 오류 원문(계약 메시지)은 저장하지 않는다.
        self.assertNotIn("합은 1", json.dumps(draft, ensure_ascii=False))

    def test_model_questions_are_advisory_and_out_of_schema_ones_are_dropped(self):
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        questions = [{"path": "split.stratify", "text": "계층 분할을 쓸까요?", "options": [True, False]},
                     {"path": "roles.nowhere", "text": "없는 컬럼", "options": ["target"]},
                     {"text": "자유 질문"}]
        draft = request_draft(task, lambda payload, config: model_reply(questions=questions))
        self.assertEqual(draft["source"], "model")
        self.assertEqual([item["level"] for item in draft["questions"]], ["advisory", "advisory"])
        self.assertEqual(draft["dropped_questions"], 1)

    def test_instructions_come_from_contract_and_match_split_checks(self):
        from types import SimpleNamespace

        import pandas as pd

        from src.agents import llm_task_assist as assist
        from src.agents.profile_store import LIST_ROLES, ROLE_LABELS, SINGLE_ROLES
        from src.preprocessing.contracts import ColumnRoles
        from src.preprocessing.split import _partition

        self.assertEqual(set(assist.ROLE_MEANINGS), set(LIST_ROLES.values()) | set(SINGLE_ROLES.values()))

        for label in ROLE_LABELS:
            self.assertIn(label, assist.instructions())

        split, preparation = assist._defaults()
        self.assertEqual(split, SPLIT)
        # 작업 분할 필드에 시간 형식이 없으므로 time 분할은 안내하지 않는다.
        self.assertFalse(assist.TIME_SPLIT_AVAILABLE)
        self.assertIn("strategy=time을 쓰지 말고", assist.instructions())

        # 지시문·교차 검사의 방식별 필요 역할이 실제 분할 실행 검사와 같은지 대조한다.
        frame = pd.DataFrame({"x": range(20), "g": [index % 10 for index in range(20)], "y": [index % 2 for index in range(20)],
                              "t": [f"2026-01-{index + 1:02d}" for index in range(20)]})
        rejected = [(SplitRules(strategy="group"), ColumnRoles(numeric=("x",))),
                    (SplitRules(strategy="time"), ColumnRoles(numeric=("x",))),
                    (SplitRules(strategy="time"), ColumnRoles(numeric=("x",), timestamp="t")),
                    (SplitRules(), ColumnRoles(numeric=("x",), group="g")),
                    (SplitRules(stratify=True), ColumnRoles(numeric=("x",)))]

        for rules, roles in rejected:
            with self.assertRaises(ValueError):
                _partition(SimpleNamespace(frame=frame, roles=roles), rules)

        _partition(SimpleNamespace(frame=frame, roles=ColumnRoles(numeric=("x",), group="g")), SplitRules(strategy="group"))
        _partition(SimpleNamespace(frame=frame, roles=ColumnRoles(numeric=("x",), target="y")), SplitRules(stratify=True))

    def test_invalid_confirmation_creates_no_runs(self):
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)

        for roles in ({"x": "제외", "label": "target", "id": "식별자", "time": "제외"},
                      {"x": "수치형", "label": "target", "id": "target", "time": "제외"}):
            with self.subTest(roles=roles), self.assertRaises(ValueError):
                orchestrator.confirm_plan(task["task_id"], plan(roles), root=self.store)

        with self.assertRaises(ValueError):
            orchestrator.confirm_plan(task["task_id"], plan(split=dict(SPLIT, train_ratio=0.9)), root=self.store)

        # 표시하지 않은 질문·선택지 밖 답변은 거부한다.
        # 표시한 질문(기본 계획의 식별자 의심)에 선택지 밖 값을 답하거나 표시하지 않은 질문 ID를 쓰면 거부한다.
        default = orchestrator.diagnose(self.new_task()["task_id"], root=self.store,
                                        requester=lambda payload, config: "응답 아님")
        shown = orchestrator.review_state(default)["questions"]
        identifier = next(item["id"] for item in shown if item["id"].startswith("identifier:"))

        for answers in ({"nope": "x"}, {identifier: "target"}):
            with self.subTest(answers=answers), self.assertRaises(PlanError):
                orchestrator.confirm_plan(default["task_id"], plan(dict(ROLES, time="범주형")), answers, root=self.store)

        self.assertEqual(task_store.load_task(default["task_id"], root=self.store)["phase"], "awaiting_review")

        self.assertEqual(list((self.root / "split").iterdir()), [])
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["phase"], "awaiting_review")

    def test_split_role_cross_check_is_shared_by_draft_and_confirmation(self):
        from src.agents.llm_task_assist import SplitRoleError, check_split_roles

        grouped = dict(ROLES, time="group")
        cases = {
            "TIME_SPLIT_UNSUPPORTED": (ROLES, dict(SPLIT, strategy="time")),
            "SPLIT_STRATEGY_MISSING": (ROLES, dict(SPLIT, strategy=None)),
            "SPLIT_ROLE_MISMATCH": (ROLES, dict(SPLIT, strategy="group")),
        }
        mismatches = [(grouped, SPLIT), (dict(ROLES, label="수치형"), dict(SPLIT, stratify=True))]

        for code, (roles, split) in [*cases.items(), *(("SPLIT_ROLE_MISMATCH", item) for item in mismatches)]:
            with self.subTest(code=code, split=split), self.assertRaises(SplitRoleError) as caught:
                check_split_roles(roles, split)

            self.assertEqual(caught.exception.code, code)

        check_split_roles(grouped, dict(SPLIT, strategy="group"))
        check_split_roles(ROLES, dict(SPLIT, stratify=True))

        # 사용자 확인도 같은 검사로 거부하고 아무 작업도 만들지 않는다.
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)

        for roles, split in [*cases.values(), *mismatches]:
            with self.subTest(split=split), self.assertRaises(ValueError):
                orchestrator.confirm_plan(task["task_id"], plan(roles, split), root=self.store)

        self.assertEqual(list((self.root / "split").iterdir()), [])
        record = task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((record["phase"], record["attempt"], record["confirmed"]), ("awaiting_review", 0, None))

    def test_time_split_draft_is_not_replaced_with_random(self):
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        draft = request_draft(task, lambda payload, config: model_reply(dict(ROLES, time="timestamp"), changes=[
            {"path": "split.strategy", "value": "time", "reason": "시간 순서"}]))

        self.assertEqual((draft["source"], draft["code"]), ("default", "TIME_SPLIT_UNSUPPORTED"))
        self.assertIsNone(draft["plan"]["split"]["strategy"])
        # 모델 역할도 채택하지 않는다. 기본 계획 역할에는 timestamp가 없다.
        self.assertNotIn("timestamp", draft["plan"]["roles"].values())
        self.assertTrue(any("random으로 자동 변경하지 않았습니다" in text for text in draft["notes"]))

        # 분할 방식이 비어 있으면 필수 질문이 생기고, 답하지 않으면 서버가 확정을 거부한다.
        record = dict(task, draft=draft)
        questions = orchestrator.review_state(record)["questions"]
        required = {item["id"]: item for item in questions if item["level"] == "required"}
        self.assertEqual(set(required), {"split.strategy", "target"})
        self.assertNotIn("time", [option["value"] for option in required["split.strategy"]["options"]])

        with self.assertRaises(PlanError) as caught:
            orchestrator.check_plan(record, draft["plan"], {})

        self.assertEqual(caught.exception.code, "REQUIRED_QUESTION_UNANSWERED")
        final, applied = orchestrator.check_plan(record, dict(draft["plan"], roles=dict(ROLES)), {"split.strategy": "random"})
        self.assertEqual((final["split"]["strategy"], applied[0]["path"]), ("random", "split.strategy"))

    def test_survival_goal_with_2urvived_column_cannot_be_confirmed_without_target(self):
        """실제 사례 재현: 목표 '생존여부 확인 위한 데이터 전처리 수행', 예측 대상 '2urvived'. 모델 요약은 target을 말했지만
        변경 목록에는 target이 없었다. 확인 화면의 필수 질문으로 target을 정하기 전에는 확정하지 않는다."""
        import pandas as pd

        goal = "생존여부 확인 위한 데이터 전처리 수행"
        frame = pd.read_csv(self.source).rename(columns={"label": "2urvived"})
        source = self.root / "survival.csv"
        frame.to_csv(source, index=False)
        sent = []
        roles = {"x": "수치형", "id": "식별자", "time": "제외"}

        def requester(payload, config):
            sent.append(json.loads(payload["messages"][1]["content"]))
            reply = json.loads(model_reply(roles))
            reply["summary"] = ["2urvived 는 생존 여부 예측 대상 (target) 으로 설정합니다."]
            return json.dumps(reply, ensure_ascii=False)

        task = orchestrator.start_task(goal, source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=requester)
        # 목표 문장은 기록·동의 A·모델 입력에 그대로 유지된다.
        self.assertEqual((task["goal"], sent[0]["goal"]), (goal, goal))
        self.assertEqual(task["draft"]["source"], "model")
        self.assertEqual(task["draft"]["plan"]["roles"]["2urvived"], "수치형")
        target = next(item for item in orchestrator.review_state(task)["questions"] if item["id"] == "target")
        self.assertEqual(target["level"], "required")
        self.assertIn("2urvived", [option["value"] for option in target["options"]])
        self.assertIn("__no_target__", [option["value"] for option in target["options"]])
        without = plan(dict(roles, **{"2urvived": "수치형"}))

        # 답하지 않으면 화면 검사와 서버 확인 모두 거부하고 아무 작업도 만들지 않는다.
        with self.assertRaises(PlanError) as caught:
            orchestrator.check_plan(task, without, {})

        self.assertEqual((caught.exception.code, caught.exception.subjects), ("REQUIRED_QUESTION_UNANSWERED", ("target",)))

        with self.assertRaises(PlanError):
            orchestrator.confirm_plan(task["task_id"], without, {}, root=self.store)

        self.assertEqual(list((self.root / "split").iterdir()), [])
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["attempt"], 0)

        # 후보를 고르면 roles.2urvived 한 항목만 target이 되고 다른 역할은 그대로다.
        final, applied = orchestrator.check_plan(task, without, {"target": "2urvived"})
        self.assertEqual(final["roles"], dict(without["roles"], **{"2urvived": "target"}))
        self.assertEqual(applied, [{"id": "target", "path": "roles.2urvived", "value": "2urvived"}])
        task = orchestrator.confirm_plan(task["task_id"], without, {"target": "2urvived"}, root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(task["confirmed"]["plan"]["roles"]["2urvived"], "target")
        self.assertEqual(task["consents"]["C"]["content"]["answers"], applied)
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertEqual({row["컬럼"]: row["역할"] for row in task["report"]["facts"]["processing"]}["2urvived"], "target")
        self.assertEqual(task["goal"], goal)

    def test_non_prediction_goal_proceeds_only_after_explicit_no_target_answer(self):
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store,
                                     requester=lambda payload, config: model_reply({"label": "수치형", "id": "식별자", "time": "제외"}))
        without = plan({"x": "수치형", "label": "수치형", "id": "식별자", "time": "제외"})
        final, applied = orchestrator.check_plan(task, without, {"target": "__no_target__"})
        self.assertEqual((final, applied), (without, [{"id": "target", "path": None, "value": "__no_target__"}]))
        task = orchestrator.confirm_plan(task["task_id"], without, {"target": "__no_target__"}, root=self.store)
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertTrue(any("target(예측 대상) 없이 처리했습니다" in text for text in task["report"]["facts"]["cautions"]))

        # 같은 질문에 target 컬럼과 "없음"을 동시에 고를 수는 없다. 답변은 질문당 하나다.
        with self.assertRaises(PlanError):
            orchestrator.check_plan(task, without, {"target": "nowhere"})

    def test_answered_required_question_does_not_pass_while_plan_still_invalid(self):
        """경로를 바꾸는 필수 질문은 답만으로 해결되지 않는다. 최종 계획에서 조건이 남으면 거부한다."""
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        grouped = plan(dict(ROLES, time="group"))
        mismatch = next(item for item in orchestrator.review_state(dict(task, draft=dict(task["draft"], plan=grouped)))["questions"]
                        if item["id"] == "split.strategy.mismatch")

        with self.assertRaises(PlanError) as caught:
            orchestrator.check_plan(dict(task, draft=dict(task["draft"], plan=grouped)), grouped, {mismatch["id"]: "random"})

        self.assertEqual(caught.exception.code, "REQUIRED_QUESTION_UNANSWERED")
        final, _ = orchestrator.check_plan(dict(task, draft=dict(task["draft"], plan=grouped)), grouped, {mismatch["id"]: "group"})
        self.assertEqual(final["split"]["strategy"], "group")

    def test_reserved_no_target_value_does_not_break_questions(self):
        from src.agents import llm_task_assist

        summary = {"row_count": 4, "repeated_rows": 0}
        columns = [{"column": name, "observed_unique": 2, "missing_count": 0, "dtype_group": "numeric"}
                   for name in ("__no_target__", "y")]
        current = {"roles": {"__no_target__": "수치형", "y": "수치형"}, "split": SPLIT, "preparation": PREPARATION}
        target = next(item for item in llm_task_assist.plan_questions("예측", summary, columns, current) if item["id"] == "target")
        self.assertEqual([option["value"] for option in target["options"]], ["y", "__no_target__"])
        self.assertEqual(target["options"][-1].get("path"), None)

    def test_identifier_like_feature_is_asked_not_changed_and_answer_changes_one_path(self):
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=lambda payload, config: "응답 아님")
        state = orchestrator.review_state(task)
        identifier = [item for item in state["questions"] if item["id"].startswith("identifier:")]
        # 값이 행마다 다른 문자열 컬럼(time)을 범주형 기본 역할로 두면 안내 질문만 만들고 역할은 바꾸지 않는다.
        # 수치형 x(연속 정수 범위)와 id(이름 단서)도 숫자형 식별자 안내 질문 대상이다.
        self.assertEqual([item["subjects"] for item in identifier], [["time"], ["x"], ["id"]])
        self.assertTrue(all(item["level"] == "advisory" for item in identifier))
        self.assertEqual((state["plan"]["roles"]["x"], state["plan"]["roles"]["id"]), ("수치형", "수치형"))
        self.assertEqual((identifier[0]["level"], identifier[0]["recommended"]), ("advisory", "식별자"))
        self.assertEqual(state["plan"]["roles"]["time"], "범주형")
        edited = plan(dict(ROLES, time="범주형"))
        final, applied = orchestrator.check_plan(task, edited, {identifier[0]["id"]: "식별자"})
        self.assertEqual(final["roles"]["time"], "식별자")
        self.assertEqual({key: value for key, value in final["roles"].items() if key != "time"},
                         {key: value for key, value in edited["roles"].items() if key != "time"})
        # 답하지 않으면 범주형 그대로 확정할 수 있다(차단하지 않음).
        orchestrator.check_plan(task, edited, {})

    def test_questions_accept_any_column_name(self):
        """질문 ID에 컬럼 이름을 넣지 않으므로 대문자·한글·공백 이름에서도 확인 화면이 깨지지 않는다."""
        from src.agents import llm_task_assist

        names = ("Name", "이름", "a b")
        summary = {"row_count": 5, "repeated_rows": 0}
        columns = [{"column": name, "observed_unique": 5, "missing_count": 0, "dtype_group": "text"} for name in names]
        current = {"roles": {name: "범주형" for name in names}, "split": SPLIT, "preparation": PREPARATION}
        questions = llm_task_assist.plan_questions("분류", summary, columns, current)
        identifier = [item for item in questions if item["id"].startswith("identifier:")]
        self.assertEqual([item["subjects"] for item in identifier], [["Name"], ["이름"], ["a b"]])
        self.assertEqual(len({item["id"] for item in identifier}), 3)

    def test_apply_failure_separates_worker_classification_and_estimated_cause(self):
        """실제 실패 사례 재현: 행마다 다른 범주형 특징으로 확정하면 Validation에 Train에 없던 범주가 생긴다."""
        task = self.run_to_apply(dict(ROLES, time="범주형"))
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        problem = task["problem"]
        self.assertEqual((task["phase"], problem["code"], problem["retryable"]), ("needs_attention", "STEP_FAILED", False))
        # 확인된 분류는 worker가 남긴 고정 진단만, 추정 원인은 어댑터 코드와 관련 항목 이름으로 분리한다.
        self.assertEqual(problem["diagnostic"]["code"], "INPUT_INVALID")
        self.assertEqual(problem["hint"], {"code": "UNSEEN_CATEGORY_LIKELY", "subjects": ["time"]})
        self.assertNotIn("time", problem["message"])
        self.assertEqual(self.calls.count("apply_preparation"), 1)

    def test_apply_requires_decision_d_and_unchanged_candidate(self):
        task = self.run_to_apply()

        with patch("src.agents.tabular_task.prepared_pipeline") as current:
            current.return_value = (None, None, None, type("E", (), {"signature": "changed"})(), None)
            stopped = orchestrator.approve_apply(task["task_id"], root=self.store)

        self.assertEqual(stopped["problem"]["code"], "CANDIDATE_CHANGED")
        self.assertEqual(self.calls.count("apply_preparation"), 0)

        with self.assertRaises(ValueError):
            orchestrator.approve_apply(task["task_id"], root=self.store)

    def tamper_snapshot(self, task):
        """같은 크기의 다른 데이터로 작업 복사본과 manifest를 바꾼다(보안 검토 재현 사례)."""
        from src.agents.input_snapshot import create_snapshot

        snapshot_dir = Path(task["snapshot"]["directory"])
        data = bytearray((snapshot_dir / "input.bin").read_bytes())
        data[-3] = ord("9") if data[-3] != ord("9") else ord("8")
        other = self.root / "tampered.csv"
        other.write_bytes(bytes(data))
        staging = self.root / "tampered_snapshot"
        create_snapshot(other, staging, max_input_bytes=len(data))
        (snapshot_dir / "input.bin").write_bytes((staging / "input.bin").read_bytes())
        (snapshot_dir / "snapshot.json").write_bytes((staging / "snapshot.json").read_bytes())

    def test_snapshot_tampering_after_diagnosis_blocks_creation_and_delegation(self):
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        self.tamper_snapshot(task)
        task = orchestrator.confirm_plan(task["task_id"], plan(), root=self.store)
        self.assertEqual(task["problem"]["code"], "SNAPSHOT_CHANGED")
        self.assertFalse(task["problem"]["retryable"])
        self.assertEqual(list((self.root / "split").iterdir()), [])
        self.assertNotIn("C", task["consents"])

    def test_snapshot_tampering_before_apply_blocks_d(self):
        task = self.run_to_apply()
        self.assertEqual(task["phase"], "awaiting_apply")
        self.tamper_snapshot(task)
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["problem"]["code"], "SNAPSHOT_CHANGED")
        self.assertEqual(self.calls.count("apply_preparation"), 0)
        self.assertNotIn("D", task["consents"])

    def fail_completed_save(self):
        """완료 기록 저장만 실패시킨다(모든 단계·보고서 작성 이후). 실행 중 기록은 그대로 남는다."""
        write = task_store._write

        def failing(path, value, **options):
            if path.name == "task.json" and value.get("phase") == "completed":
                raise ValueError("작업 기록이 크기 상한을 초과했습니다.")

            return write(path, value, **options)

        return patch("src.agents.task_store._write", side_effect=failing)

    def test_completion_save_failure_is_recovered_without_worker_or_new_approval(self):
        """모든 단계와 보고서 작성이 끝났지만 완료 저장이 실패하면, 검증 후 재실행·재승인 없이 완료 기록을 복구한다."""
        task = self.run_to_apply()

        with self.fail_completed_save(), self.assertRaises(ValueError):
            orchestrator.approve_apply(task["task_id"], root=self.store)

        stuck = task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stuck["phase"], stuck["report"]), ("applying", None))
        self.assertTrue(orchestrator.recoverable(stuck))
        reports = list((self.store / task["task_id"] / "report").iterdir())
        self.assertEqual(len(reports), 1)
        calls, delegations = list(self.calls), {step: dict(item["delegations"]) for step, item in stuck["steps"].items()}
        approvals = sorted(path.name for path in self.root.rglob("ui_approval_*.json"))

        # 동의 내용이 바뀐 기록은 복구하지 않는다. 기록은 바꾸지 않는다.
        path = self.store / task["task_id"] / "task.json"
        body = path.read_bytes()
        value = json.loads(body)
        value["consents"]["D"]["content"]["plan_signature"] = "other"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

        with self.assertRaises(orchestrator.RecoveryError) as caught:
            orchestrator.recover_completion(task["task_id"], root=self.store)

        self.assertEqual(caught.exception.code, "RECOVERY_CONSENT_INVALID")
        path.write_bytes(body)

        # 동의는 그대로여도 확인한 적용 대상 서명이 동의와 다르면 복구하지 않는다.
        value = json.loads(body)
        value["candidate"]["signature"] = "0" * 64
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

        with self.assertRaises(orchestrator.RecoveryError) as caught:
            orchestrator.recover_completion(task["task_id"], root=self.store)

        self.assertEqual(caught.exception.code, "RECOVERY_CONSENT_INVALID")
        path.write_bytes(body)

        with self.no_execution():
            task = orchestrator.recover_completion(task["task_id"], root=self.store)

        self.assertEqual(task["phase"], "completed")
        self.assertEqual((task["recovered"]["worker_runs"], task["recovered"]["new_approvals"], task["recovered"]["read_only_steps"]),
                         (0, 0, []))
        # worker·승인 추가 없음. 같은 내용의 기존 보고서 파일을 그대로 참조한다.
        self.assertEqual(self.calls, calls)
        self.assertEqual({step: item["delegations"] for step, item in task["steps"].items()}, delegations)
        self.assertEqual(sorted(path.name for path in self.root.rglob("ui_approval_*.json")), approvals)
        self.assertEqual(list((self.store / task["task_id"] / "report").iterdir()), reports)
        self.assertEqual(Path(task["report"]["path"]), reports[0])
        import hashlib
        self.assertEqual(task["report"]["sha256"], hashlib.sha256(reports[0].read_bytes()).hexdigest())
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["report"]["text"], reports[0].read_text(encoding="utf-8"))
        self.assertFalse(orchestrator.recoverable(task))

        with self.assertRaises(orchestrator.RecoveryError):
            orchestrator.recover_completion(task["task_id"], root=self.store)

    def changed_code(self):
        """실행 뒤 연결 코드가 바뀐 상황. 현재 지문으로는 기존 전달 실행을 열 수 없다."""
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("src.agents.csv_handoff_workflow.code_id", return_value="f" * 64))
        stack.enter_context(patch("src.agents.tabular_task.csv_code_id", return_value="f" * 64))
        return stack

    def no_execution(self):
        """복구 경로에서 새 승인·위임·실행·전달 묶음 생성이 호출되면 실패시킨다."""
        from contextlib import ExitStack

        stack = ExitStack()

        for target in ("src.agents.csv_workflow_service.CsvWorkflowService.next",
                       "src.agents.csv_workflow_service.CsvWorkflowService.record_delegated_approval",
                       "src.agents.approval_store.ApprovalStore.record_confirmed_approval",
                       "src.agents.executor.RulesExecutor.run_next", "src.agents.task_consent.grant",
                       "src.agents.csv_handoff_workflow.export_bundle"):
            stack.enter_context(patch(target, side_effect=AssertionError(target)))

        return stack

    def test_recovery_after_code_change_verifies_handoff_read_only_without_execution(self):
        """실행 뒤 코드가 바뀌어도 기존 승인 검사는 그대로 거부하고, 복구는 기록 간 코드 지문을 대조한 읽기 전용 검증으로만 한다."""
        from src.agents import csv_handoff_workflow
        from src.agents.workflow_service_registry import workflow_service

        task = self.run_to_apply()

        with self.fail_completed_save(), self.assertRaises(ValueError):
            orchestrator.approve_apply(task["task_id"], root=self.store)

        stuck = task_store.load_task(task["task_id"], root=self.store)
        name = stuck["steps"]["handoff"]["run_name"]
        recorded = stuck["consents"]["D"]["content"]["csv_code_id"]
        service = workflow_service("csv_local_handoff", "1.0.0")
        calls = list(self.calls)
        files = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file() and "tasks" not in path.parts}

        with self.changed_code():
            # 기존 실행·승인 경로는 완화하지 않는다.
            with self.assertRaises(ValueError):
                service.status(name)

            with self.assertRaises(ValueError):
                service.next(name, expected_stage="ingest", expected_revision=0)

            # 읽기 전용으로 연 실행은 실행 함수를 호출할 수 없다.
            tool, checker = csv_handoff_workflow.open_recorded(name, recorded)

            for registered in checker.tools.values():
                with self.assertRaises(ValueError):
                    registered.execute(self.root, {})

            for method in (tool.preflight, tool.execute):
                with self.assertRaises(ValueError):
                    method(self.root, tool.arguments)

            # 기록 간 코드 지문이 다르면 읽기 전용으로도 열지 않는다.
            with self.assertRaises(ValueError):
                service.recorded_results(name, "e" * 64)

            # 전달 선택 기록·승인 증거·전달 묶음이 바뀌면 복구하지 않는다. 기록은 바꾸지 않는다.
            run = self.handoff_runs / name
            proof = next(run.glob("ui_approval_*.json"))
            bundle = self.bundles / name / "README.md"
            damages = {"selection": (run / "selection.json", lambda text: text.replace(recorded, "d" * 64)),
                       "proof": (proof, lambda text: text.replace(recorded, "d" * 64)),
                       "bundle": (bundle, lambda text: text + "\n변경")}
            # 승인이 필요한 단계의 증거가 둘이면(복제) 열지 않는다.
            duplicate = run / ("ui_approval_" + "0" * 32 + ".json")
            duplicate.write_bytes(proof.read_bytes())

            with self.assertRaises(ValueError):
                service.recorded_results(name, recorded)

            duplicate.unlink()

            for label, (path, damage) in damages.items():
                original = path.read_bytes()
                path.write_text(damage(original.decode("utf-8")), encoding="utf-8")

                with self.subTest(label), self.no_execution(), self.assertRaises(orchestrator.RecoveryError) as caught:
                    orchestrator.recover_completion(task["task_id"], root=self.store)

                self.assertEqual((caught.exception.code, caught.exception.step), ("RECOVERY_STEP_UNVERIFIED", "handoff"))
                self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["phase"], "applying")
                path.write_bytes(original)

            with self.no_execution():
                task = orchestrator.recover_completion(task["task_id"], root=self.store)

        self.assertEqual(task["phase"], "completed")
        self.assertEqual(task["recovered"]["read_only_steps"], ["handoff"])
        self.assertEqual(self.calls, calls)
        # 기존 산출물(실행 폴더·승인 저장소·전달 묶음)은 바뀌지 않는다.
        self.assertEqual({path: path.read_bytes() for path in files}, files)

    def test_tampered_handoff_is_not_read_through_recorded_path_when_code_is_unchanged(self):
        """코드가 그대로면 읽기 전용 검증을 쓰지 않는다. 결과가 바뀐 실행은 그대로 거부한다."""
        task = self.run_to_apply()

        with self.fail_completed_save(), self.assertRaises(ValueError):
            orchestrator.approve_apply(task["task_id"], root=self.store)

        name = task_store.load_task(task["task_id"], root=self.store)["steps"]["handoff"]["run_name"]
        bundle = self.bundles / name / "README.md"
        bundle.write_text(bundle.read_text(encoding="utf-8") + "\n변경", encoding="utf-8")

        with patch("src.agents.csv_workflow_service.CsvWorkflowService.recorded_results", side_effect=AssertionError("recorded")), \
                self.assertRaises(orchestrator.RecoveryError) as caught:
            orchestrator.recover_completion(task["task_id"], root=self.store)

        self.assertEqual((caught.exception.code, caught.exception.step), ("RECOVERY_STEP_UNVERIFIED", "handoff"))

    def test_tampered_run_names_in_task_record_are_rejected(self):
        task = self.run_to_apply()
        path = self.store / task["task_id"] / "task.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["steps"]["candidate"]["run_name"] = "other-candidate"
        value["candidate"]["run_name"] = "other-candidate"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["problem"]["code"], "STATE_UNCLEAR")
        self.assertEqual(self.calls.count("apply_preparation"), 0)

    def test_reserved_step_is_not_recreated_and_foreign_task_is_not_opened(self):
        task = self.run_to_apply()
        directory = self.store / task["task_id"]

        with self.assertRaises(orchestrator.TaskProblem) as caught:
            orchestrator._run_step(task, directory, "split", "csv_split_review", "C", self.store)

        self.assertEqual(caught.exception.code, "STATE_UNCLEAR")

        with patch("src.agents.task_store._current_actor", return_value="OTHER\\user"):
            with self.assertRaises(ValueError):
                task_store.load_task(task["task_id"], root=self.store)

    def test_quality_review_is_fixed_in_place_with_new_attempt(self):
        """품질 검토로 멈추면 그 자리에서 역할을 고쳐 새 분할 작업으로 이어 간다. 이전 작업은 보존한다."""
        import pandas as pd

        frame = pd.read_csv(self.source)
        frame["const"] = 5
        frame.to_csv(self.source, index=False)
        roles = dict(ROLES, const="수치형")
        task = self.run_to_apply(roles)
        self.assertEqual(task["phase"], "needs_attention")
        # 간헐 실패 추적: 다르면 문제 기록 전체(예외 종류·단계 포함, 원문 없음)를 실패 메시지에 남긴다.
        self.assertEqual(task["problem"]["code"], "QUALITY_REVIEW", task["problem"])
        self.assertTrue(task["problem"]["retryable"])
        # 확인된 지적(검증된 품질 보고서)을 코드·대상으로 남긴다. 품질 기준은 바꾸지 않는다.
        self.assertIn({"code": "CONSTANT_FEATURE", "subject": "const"}, task["problem"]["findings"])
        self.assertEqual(orchestrator.review_state(task)["findings"], task["problem"]["findings"])
        # 이전 형식 기록(지적 없음)도 검증된 결과로 다시 읽어 보여 준다. 기록은 바꾸지 않는다.
        legacy = dict(task, problem={key: value for key, value in task["problem"].items() if key != "findings"})
        self.assertEqual(orchestrator.review_state(legacy)["findings"], task["problem"]["findings"])
        # 기록의 분할 작업 이름이 바뀌면 다른 작업의 지적을 보여 주지 않는다.
        tampered = dict(legacy, steps=dict(legacy["steps"], split=dict(legacy["steps"]["split"], run_name="other-split")))
        self.assertEqual(orchestrator.review_state(tampered)["findings"], [])
        first = task["steps"]["split"]["run_name"]

        task = orchestrator.confirm_plan(task["task_id"], plan(dict(ROLES, const="제외")), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(task["attempt"], 2)
        self.assertNotEqual(task["steps"]["split"]["run_name"], first)
        self.assertTrue((self.root / "split" / first).exists())

    def category_source(self, name, unique):
        """범주 수 상한 검사용 합성 입력. name 컬럼은 행마다 unique개 값을 돌려 쓴다."""
        import pandas as pd

        frame = pd.read_csv(self.source)
        frame[name] = ["v" + str(index % unique) for index in range(len(frame))]
        frame.to_csv(self.source, index=False)

    def test_category_limit_is_asked_before_execution_and_time_candidate_is_linked(self):
        """반드시 상한을 넘는 범주형 특징은 필수 질문으로 확정을 막고, 시간 후보는 역할 확인 질문으로 잇는다. 역할은 바꾸지 않는다.

        통합 작업은 시간 형식을 지정할 수 없어 timestamp 역할을 권하지 않는다(형식 없이 지정하면 품질 검사가 멈춘다).
        """
        self.category_source("stamp_time", 120)
        roles = dict(ROLES, stamp_time="범주형")
        limited = plan(roles, preparation=dict(PREPARATION, max_categories=8))
        task = self.new_task()
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=lambda payload, config: model_reply(roles))
        record = dict(task, draft=dict(task["draft"], plan=limited))
        question = next(item for item in orchestrator.review_state(record)["questions"] if item["id"].startswith("category_limit:"))
        self.assertEqual((question["level"], question["path"], question["recommended"]), ("required", "roles.stamp_time", "제외"))
        self.assertEqual([option["value"] for option in question["options"]], ["제외", "식별자"])
        # group 분할은 Train 행 수 하한을 보장하지 않으므로 안내 질문이다.
        grouped = dict(limited, split=dict(limited["split"], strategy="group"))
        self.assertEqual(next(item for item in orchestrator.review_state(dict(task, draft=dict(task["draft"], plan=grouped)))["questions"]
                              if item["id"].startswith("category_limit:"))["level"], "advisory")
        self.assertEqual(record["draft"]["plan"]["roles"]["stamp_time"], "범주형")

        with self.assertRaises(PlanError) as caught:
            orchestrator.confirm_plan(task["task_id"], limited, root=self.store)

        self.assertEqual(caught.exception.code, "REQUIRED_QUESTION_UNANSWERED")
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["steps"], {})
        # 상한 안의 시간 이름 컬럼은 안내 질문으로 확인한다. 수치형 측정값·제외 컬럼은 묻지 않는다.
        small = plan(dict(roles, stamp_time="범주형"))
        asked = [item for item in orchestrator.review_state(dict(task, draft=dict(task["draft"], plan=small)))["questions"]
                 if item["id"].startswith(("time_candidate:", "category_limit:"))]
        self.assertEqual([(item["level"], item["subjects"], item["recommended"]) for item in asked],
                         [("advisory", ["stamp_time"], "제외")])
        # 형식 없는 timestamp 지정은 실행 전에 필수 질문으로 막는다. 이전에는 분할 전 품질 검사(TIMESTAMP_FORMAT_REQUIRED)에서 멈췄다.
        timed = plan(dict(roles, stamp_time="timestamp"))
        question = next(item for item in orchestrator.review_state(dict(task, draft=dict(task["draft"], plan=timed)))["questions"]
                        if item["id"].startswith("time_role:"))
        self.assertEqual((question["level"], question["recommended"]), ("required", "제외"))

        with self.assertRaises(PlanError):
            orchestrator.confirm_plan(task["task_id"], timed, root=self.store)

        task = orchestrator.confirm_plan(task["task_id"], plan(dict(roles, stamp_time="제외"),
                                                               preparation=dict(PREPARATION, max_categories=8)), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))

    def test_candidate_failure_has_fixed_detail_code_and_is_revised_as_new_attempt(self):
        """Train 범주 수가 상한을 넘으면 worker가 세부 코드를 남기고, 사용자가 계획을 고쳐 새 시도로 진행한다. 이전 기록은 보존한다."""
        self.category_source("code", 40)
        roles = dict(ROLES, code="범주형")
        preparation = dict(PREPARATION, max_categories=8)
        task = self.new_task()
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=lambda payload, config: model_reply(roles))
        # 전체 고유값으로는 Train 초과가 확실하지 않아 안내 질문이다. 확정을 막지 않는다.
        question = next(item for item in orchestrator.review_state(dict(task, draft=dict(task["draft"], plan=plan(roles, preparation=preparation))))["questions"]
                        if item["id"].startswith("category_limit:"))
        self.assertEqual(question["level"], "advisory")
        task = orchestrator.confirm_plan(task["task_id"], plan(roles, preparation=preparation), root=self.store)
        self.assertEqual((task["phase"], task["problem"]["code"]), ("needs_attention", "STEP_FAILED"), task.get("problem"))
        self.assertEqual(task["problem"]["diagnostic"], {"stage": "INPUT_VALIDATION", "code": "INPUT_INVALID",
                                                         "detail": "CATEGORY_LIMIT_EXCEEDED"})
        self.assertEqual(task["problem"]["hint"], {"code": "CATEGORY_LIMIT_EXCEEDED", "subjects": ["code"]})
        self.assertFalse(task["problem"]["retryable"])
        self.assertTrue(orchestrator.revisable(task))
        # 세부 코드가 없는 이전 기록은 추정으로만 안내한다. 기록은 바꾸지 않는다.
        legacy = dict(task, problem=dict(task["problem"], hint=None, diagnostic={"stage": "INPUT_VALIDATION", "code": "INPUT_INVALID"}))
        self.assertEqual(orchestrator.problem_view(legacy)["hint"]["code"], "CATEGORY_LIMIT_LIKELY")
        # 환경 실패·결정 D 이후는 계획 수정으로 진행하지 않는다.
        self.assertFalse(orchestrator.revisable(dict(task, problem=dict(task["problem"], diagnostic={"stage": "SNAPSHOT", "code": "ACCESS_DENIED"}))))
        self.assertFalse(orchestrator.revisable(dict(task, consents=dict(task["consents"], D={}))))
        before = {step: dict(item) for step, item in task["steps"].items()}
        consent = task["consents"]["C"]

        task = orchestrator.confirm_plan(task["task_id"], plan(dict(roles, code="제외"), preparation=preparation), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(task["attempts"], [{"attempt": 1, "steps": before, "consent_c": consent, "problem_code": "STEP_FAILED",
                                             "plan_signature": task["attempts"][0]["plan_signature"]}])
        self.assertEqual({item["status"] for step, item in before.items() if step != "candidate"}, {"completed"})
        self.assertNotEqual(task["consents"]["C"]["consent_id"], consent["consent_id"])
        self.assertTrue(all((self.root / module_dir / item["run_name"]).exists()
                            for module_dir, item in (("split", before["split"]),)))
        self.assertTrue(set(item["run_name"] for item in task["steps"].values()).isdisjoint(item["run_name"] for item in before.values()))

    def test_duplicate_header_is_resolved_with_consented_numbered_copy(self):
        """중복 컬럼명으로 진단이 막히면 사유·개수와 해결 동작을 제시하고, 동의하면 번호 붙인 복사본으로 끝까지 진행한다."""
        import pandas as pd

        frame = pd.read_csv(self.source)
        frame.insert(1, "zero", frame["x"] * 2)
        frame.insert(2, "zero_", frame["x"] * 3)
        source = self.root / "duplicate.csv"
        frame.to_csv(source, index=False, header=["x", "zero", "zero", "label", "id", "time"])
        before = source.read_bytes()
        calls = []

        def requester(payload, config):
            calls.append(json.loads(payload["messages"][1]["content"]))
            return model_reply(dict(ROLES, zero_1="수치형", zero_2="수치형"))

        task = orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=requester)
        problem = task["problem"]
        self.assertEqual((task["phase"], problem["code"], problem["actions"]), ("needs_attention", "DUPLICATE_COLUMN_NAMES", ["header_dedupe"]))
        self.assertEqual(problem["detail"], {"duplicated_names": 1, "affected_columns": 2})
        self.assertNotIn("zero", problem["message"])
        self.assertEqual(calls, [])

        # 제시하지 않은 동작은 거부한다. 동의 전에는 정비하지 않는다.
        with self.assertRaises(ValueError):
            orchestrator.resolve_problem(task["task_id"], "drop_columns", root=self.store, requester=requester)

        task = orchestrator.resolve_problem(task["task_id"], "header_dedupe", root=self.store, requester=requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        self.assertEqual([row["column"] for row in task["diagnostic"]["columns"]], ["x", "zero_1", "zero_2", "label", "id", "time"])
        self.assertEqual(task["source"]["derived"]["renamed"], [{"original": "zero", "count": 2, "first": "zero_1", "last": "zero_2"}])
        self.assertNotEqual(task["snapshot"]["snapshot"]["sha256"], task["source"]["derived"]["original_snapshot"]["snapshot"]["sha256"])
        content = task["consents"]["A"]["content"]
        self.assertEqual((content["header_dedupe"], content["allowed"]),
                         (True, [["header_dedupe", "run_pipeline"], ["input_profile", "run_pipeline"]]))
        subjects = calls[0]["subjects"]
        sent_names = [row[0] for row in subjects["detailed"]] + subjects["named"]
        self.assertTrue({"x", "zero_1", "zero_2"} <= set(sent_names))
        self.assertNotIn("zero", sent_names)
        self.assertTrue(any("번호를 붙였습니다" in line for line in task_adapters.adapter("tabular_csv").input_notices(task)))

        # 이미 정비한 작업에서는 같은 동작을 다시 고를 수 없다.
        with self.assertRaises(ValueError):
            orchestrator.resolve_problem(task["task_id"], "header_dedupe", root=self.store, requester=requester)

        task = orchestrator.confirm_plan(task["task_id"], plan(dict(ROLES, zero_1="수치형", zero_2="수치형")), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        rows = [row for row in task["report"]["facts"]["problems"] if row["구분"] == "중복 컬럼명"]
        self.assertEqual([row["컬럼"] for row in rows], ["zero"])
        self.assertEqual(source.read_bytes(), before)

    def blocked_duplicate(self):
        """중복 컬럼명으로 진단이 막힌 작업과 원본 파일을 만든다."""
        import pandas as pd

        frame = pd.read_csv(self.source)
        frame.insert(1, "zero", frame["x"] * 2)
        frame.insert(2, "zero_", frame["x"] * 3)
        source = self.root / "duplicate.csv"
        frame.to_csv(source, index=False, header=["x", "zero", "zero", "label", "id", "time"])
        task = orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["problem"]["code"], "DUPLICATE_COLUMN_NAMES")
        return task, source

    def resolve(self, task):
        return orchestrator.resolve_problem(task["task_id"], "header_dedupe", root=self.store,
                                            requester=lambda payload, config: model_reply(dict(ROLES, zero_1="수치형", zero_2="수치형")))

    def runs(self):
        return sorted(path.name for path in (self.store.parent / "diag_runs").iterdir())

    def test_dedupe_uses_verified_snapshot_after_source_moves_deletes_or_changes(self):
        for change in ("move", "delete", "resize"):
            with self.subTest(change=change):
                task, source = self.blocked_duplicate()

                if change == "move":
                    source.rename(source.with_name("moved.csv"))
                elif change == "delete":
                    source.unlink()
                else:
                    source.write_bytes(source.read_bytes() + b"9,9,9,0,x,y\n")

                task = self.resolve(task)
                self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
                # 원본 경로는 출처 정보로만 남고 현재 원본으로 바뀌지 않는다.
                self.assertEqual(task["source"]["path"], str(source.resolve()))
                self.assertEqual(task["diagnostic"]["summary"]["row_count"], 120)

                for path in self.root.glob("moved.csv"):
                    path.unlink()

    def test_dedupe_history_links_runs_consents_and_snapshots(self):
        task, source = self.blocked_duplicate()
        before = source.read_bytes()
        first_run, first_consent = task["diagnostic"]["run_name"], task["consents"]["A"]["consent_id"]
        original = task["snapshot"]
        task = self.resolve(task)
        entry = task["history"][-1]
        self.assertEqual(entry, {
            "action": "header_dedupe", "status": "completed", "previous_diagnostic_run": first_run,
            "previous_consent_id": first_consent, "consent_id": task["consents"]["A"]["consent_id"],
            "snapshot_before_sha256": original["snapshot"]["sha256"],
            "snapshot_after_sha256": task["snapshot"]["snapshot"]["sha256"],
            "resolution_run": task["dedupe"]["run_name"], "rediagnostic_run": task["diagnostic"]["run_name"]})
        self.assertNotEqual(entry["previous_consent_id"], entry["consent_id"])
        self.assertEqual(len({first_run, entry["resolution_run"], entry["rediagnostic_run"]}), 3)
        # 이전 실행 기록은 지우지 않는다. 원본은 그대로이고 정비 복사본은 헤더 뒤 바이트가 원본 snapshot과 같다.
        self.assertTrue({first_run, entry["resolution_run"], entry["rediagnostic_run"]} <= set(self.runs()))
        self.assertEqual(source.read_bytes(), before)
        old = (Path(original["directory"]) / "input.bin").read_bytes()
        new = (Path(task["snapshot"]["directory"]) / "input.bin").read_bytes()
        self.assertEqual(old.split(b"\n", 1)[1], new.split(b"\n", 1)[1])
        # 줄 끝 표기(CRLF·LF)는 원본 그대로 유지되므로 이름만 비교한다.
        self.assertEqual(new.split(b"\n", 1)[0].decode().rstrip("\r").split(","), ["x", "zero_1", "zero_2", "label", "id", "time"])
        self.assertEqual(old.split(b"\n", 1)[0].endswith(b"\r"), new.split(b"\n", 1)[0].endswith(b"\r"))

        # 정비한 입력은 다시 정비하지 않는다. 거부는 새 동의·실행·이력을 만들기 전에 일어난다.
        consent, runs, history = task["consents"]["A"]["consent_id"], self.runs(), list(task["history"])
        record = task_store.load_task(task["task_id"], root=self.store)
        path = self.store / task["task_id"] / "task.json"
        path.write_text(json.dumps(dict(record, phase="needs_attention", problem={
            "code": "DUPLICATE_COLUMN_NAMES", "message": "m", "retryable": False, "diagnostic": None, "hint": None,
            "actions": ["header_dedupe"], "detail": {}}), ensure_ascii=False), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.resolve(task)

        stored = task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["consents"]["A"]["consent_id"], stored["history"], self.runs()), (consent, history, runs))

    def test_unresolvable_requests_create_no_consent_or_run(self):
        task, _ = self.blocked_duplicate()
        consent, runs = task["consents"]["A"]["consent_id"], self.runs()
        path = self.store / task["task_id"] / "task.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        cases = [("drop_columns", record), ("header_dedupe", dict(record, phase="awaiting_review")),
                 ("header_dedupe", dict(record, problem=dict(record["problem"], code="EMPTY_COLUMN_NAME")))]

        for action, value in cases:
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

            with self.subTest(action=action, phase=value["phase"]), self.assertRaises(ValueError):
                orchestrator.resolve_problem(task["task_id"], action, root=self.store, requester=self.requester)

            stored = task_store.load_task(task["task_id"], root=self.store)
            self.assertEqual((stored["consents"]["A"]["consent_id"], stored.get("history"), self.runs()), (consent, None, runs))

    def test_tampered_snapshot_blocks_dedupe_without_new_consent(self):
        task, _ = self.blocked_duplicate()
        consent, runs = task["consents"]["A"]["consent_id"], self.runs()
        self.tamper_snapshot(task)
        task = self.resolve(task)
        self.assertEqual((task["phase"], task["problem"]["code"]), ("needs_attention", "SNAPSHOT_CHANGED"))
        self.assertEqual((task["consents"]["A"]["consent_id"], task.get("history"), self.runs()), (consent, None, runs))

    def test_previous_diagnostic_consent_cannot_authorize_dedupe(self):
        from types import SimpleNamespace

        from src.agents import tabular_task

        task, _ = self.blocked_duplicate()
        directory = self.store / task["task_id"]
        context = SimpleNamespace(save=lambda value: task_store.save_task(value, expected_revision=value["revision"], root=self.store),
                                  authorizer=task_consent.authorizer)

        # 새 동의 없이 어댑터를 직접 불러도 이전 진단 동의(정비 미허용)로는 정비 승인을 발급하지 않는다.
        with task_store.task_lock(task["task_id"], root=self.store):
            with self.assertRaises(ValueError):
                tabular_task.resolve(task_store.load_task(task["task_id"], root=self.store), directory, context, "header_dedupe")

        events = "".join(path.read_text(encoding="utf-8") for path in (self.store.parent / "diag_runs").rglob("events.jsonl")
                         if "-dedu" in str(path))
        self.assertNotIn("DELEGATED_APPROVAL_RECORDED", events)
        self.assertNotIn("derived", task_store.load_task(task["task_id"], root=self.store)["source"])

    def test_dedupe_copy_with_mismatched_hash_or_size_is_not_used(self):
        from src.agents import tabular_task
        from src.agents.input_snapshot import create_snapshot as real

        for kind in ("hash", "size"):
            with self.subTest(kind=kind):
                task, _ = self.blocked_duplicate()
                original = task["snapshot"]

                def forged(source, destination, *, max_input_bytes, kind=kind):
                    data = bytearray(Path(source).read_bytes())

                    if kind == "hash":
                        data[-2] = ord("8") if data[-2] != ord("8") else ord("7")
                    else:
                        data += b"1\n"

                    fake = self.root / ("forged_" + kind + ".csv")
                    fake.write_bytes(bytes(data))
                    return real(fake, destination, max_input_bytes=max(max_input_bytes, len(data)))

                with patch.object(tabular_task, "create_snapshot", side_effect=forged):
                    task = self.resolve(task)

                self.assertEqual((task["phase"], task["problem"]["code"]), ("needs_attention", "DEDUPE_FAILED"))
                self.assertEqual(task["snapshot"], original)
                self.assertNotIn("derived", task["source"])
                self.assertEqual(task["history"][-1]["status"], "started")
                # 실패 뒤 자동으로 다시 정비하지 않는다.
                with self.assertRaises(ValueError):
                    orchestrator.resolve_problem(task["task_id"], "header_dedupe", root=self.store, requester=self.requester)

    def test_empty_column_name_is_explained_without_automatic_fix(self):
        import pandas as pd

        frame = pd.read_csv(self.source)
        source = self.root / "empty_name.csv"
        frame.to_csv(source, index=False, header=["x", " ", "id", "time"])
        task = orchestrator.start_task("label 예측", source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual((task["problem"]["code"], task["problem"]["actions"]), ("EMPTY_COLUMN_NAME", []))

    def test_legacy_csv_task_record_is_read_as_tabular_kind(self):
        task = self.new_task()
        path = self.store / task["task_id"] / "task.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["schema_version"] = "csv_task_v1"
        value.pop("kind")
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["kind"], "tabular_csv")


class TaskStoreSizeTest(unittest.TestCase):
    """작업 기록 상한(512 KiB)은 worker IPC 상한(64 KiB)과 별개다. 엄격한 JSON 규칙은 그대로다."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "tasks"

    def record(self, padding):
        task = task_store.create_task("목표", {"path": "x.csv", "size_bytes": 1}, "tabular_csv", root=self.root)
        task["diagnostic"] = {"padding": "가" * padding}
        return task

    def test_record_between_64_kib_and_512_kib_saves_and_reloads(self):
        task = self.record(60 * 1024)
        saved = task_store.save_task(task, expected_revision=task["revision"], root=self.root)
        reloaded = task_store.load_task(task["task_id"], root=self.root)
        self.assertEqual((reloaded["revision"], reloaded["diagnostic"]), (saved["revision"], task["diagnostic"]))

    def test_wide_record_with_attempts_and_final_report_is_split_and_verified(self):
        """넓은 표 진단·재시도 이력·최종 보고서가 함께 있어 본문 상한을 넘는 기록을 분리 저장하고 그대로 다시 읽는다."""
        task = task_store.create_task("목표", {"path": "x.csv", "size_bytes": 1}, "tabular_csv", root=self.root)
        columns = [{"column": f"sensor_{index}", "dtype_group": "numeric", "missing_ratio": 0.01, "observed_unique": 900,
                    "mean": 1.2345, "std": 0.5, "min": -3.0, "max": 3.0, "q25": -0.5, "median": 0.0, "q75": 0.5}
                   for index in range(1200)]
        task.update(diagnostic={"summary": {"row_count": 1567, "column_count": 1200}, "columns": columns},
                    attempts=[{"attempt": index, "steps": {"split": {"run_name": f"t{index}-spli", "delegations": {}}},
                               "consent_c": {"consent_id": "c" * 32}, "problem_code": "STEP_FAILED", "plan_signature": "p"}
                              for index in range(1, 4)],
                    candidate={"facts": {"tables": [{"rows": [{"특징": f"sensor_{index}", "처리": "결측 대치 0.1"} for index in range(1200)]}]}},
                    report={"path": "report/report_3_1.md", "sha256": "0" * 64, "text": "| 특징 | 처리 |\n" * 20000,
                            "facts": {"tables": []}}, phase="completed")
        self.assertGreater(len(task_store._encode(task)), task_store.MAX_TASK_BYTES)
        saved = task_store.save_task(task, expected_revision=task["revision"], root=self.root)
        folder = self.root / task["task_id"]
        main = json.loads((folder / "task.json").read_text(encoding="utf-8"))
        self.assertLess((folder / "task.json").stat().st_size, task_store.MAX_TASK_BYTES)
        self.assertEqual({key for key in task_store.PART_SECTIONS if "$part" in (main.get(key) or {})},
                         {"diagnostic", "candidate", "report"})
        reloaded = task_store.load_task(task["task_id"], root=self.root)
        self.assertEqual(reloaded, saved)
        # 같은 내용을 다시 저장하면 같은 파일을 참조한다. 이전 파일은 지우지 않는다.
        again = task_store.save_task(reloaded, expected_revision=reloaded["revision"], root=self.root)
        self.assertEqual(len(list((folder / "parts").iterdir())), 3)
        self.assertEqual(task_store.load_task(task["task_id"], root=self.root), again)

        part = folder / "parts" / main["report"]["$part"]["file"]
        original = part.read_bytes()
        tampered = original.replace("특징".encode("utf-8"), "변경".encode("utf-8"), 1)
        cases = {"내용 변경": lambda: part.write_bytes(tampered), "파일 없음": part.unlink,
                 "참조 이름 변경": lambda: (folder / "task.json").write_text(json.dumps(dict(
                     json.loads((folder / "task.json").read_text(encoding="utf-8")),
                     report={"$part": dict(main["report"]["$part"], file="../task.json")})), encoding="utf-8")}

        for name, damage in cases.items():
            body = (folder / "task.json").read_bytes()
            damage()

            with self.subTest(name), self.assertRaises(task_store.TaskRecordError) as caught:
                task_store.load_task(task["task_id"], root=self.root)

            self.assertEqual(caught.exception.code, "RECORD_PART_INVALID")
            part.write_bytes(original)
            (folder / "task.json").write_bytes(body)

    def test_oversized_duplicate_or_nonstandard_records_are_rejected_with_cause(self):
        task = self.record(10)
        path = self.root / task["task_id"] / "task.json"

        # 분리 저장 대상이 아닌 진행 제어 구역이 본문 상한을 넘으면 저장하지 않는다.
        with self.assertRaises(ValueError):
            task_store.save_task(dict(task, confirmed={"padding": "가" * 200 * 1024}), expected_revision=task["revision"], root=self.root)

        cases = {"RECORD_TOO_LARGE": json.dumps(dict(task, diagnostic={"padding": "a" * 600 * 1024})),
                 "RECORD_UNREADABLE": path.read_text(encoding="utf-8")[:-1] + ', "phase": "created"}',
                 "RECORD_INVALID": json.dumps(dict(task, phase="unknown"))}
        nan = json.dumps(dict(task, revision=float("nan")))

        for code, text in [*cases.items(), ("RECORD_UNREADABLE", nan)]:
            path.write_text(text, encoding="utf-8")

            with self.subTest(code=code), self.assertRaises(task_store.TaskRecordError) as caught:
                task_store.load_task(task["task_id"], root=self.root)

            self.assertEqual(caught.exception.code, code)

        path.write_text(json.dumps(task), encoding="utf-8")

        with patch("src.agents.task_store._current_actor", return_value="OTHER\\user"):
            with self.assertRaises(task_store.TaskRecordError) as caught:
                task_store.load_task(task["task_id"], root=self.root)

        self.assertEqual(caught.exception.code, "RECORD_OWNER_MISMATCH")

        with self.assertRaises(task_store.TaskRecordError) as caught:
            task_store.load_task("0" * 32, root=self.root)

        self.assertEqual(caught.exception.code, "RECORD_NOT_FOUND")


class ReportReuseTest(unittest.TestCase):
    """완료 기록 복구의 보고서 재참조는 '기술 상세' JSON의 순서 차이만 같은 내용으로 본다."""

    def text(self, body, technical):
        # 2026-10-22부터 기술 상세는 접힌 구역(<details>)으로 렌더링한다. 같은 구역 표식으로 만든다.
        from src.agents.task_report import TECHNICAL_HEAD, TECHNICAL_TAIL

        return ("\n".join(["# 결과 보고서", *body]) + "\n" + TECHNICAL_HEAD
                + json.dumps(technical, ensure_ascii=False, indent=2) + TECHNICAL_TAIL + "\n")

    def test_only_technical_order_difference_is_same_report(self):
        from src.agents.task_report import _same_report

        rebuilt = self.text(["문제점 A", "처리 B"], {"runs": {"split": "s", "export": "e"}, "reports": ["x", "y"]})
        self.assertTrue(_same_report(rebuilt, rebuilt))
        self.assertTrue(_same_report(self.text(["문제점 A", "처리 B"], {"runs": {"export": "e", "split": "s"}, "reports": ["y", "x"]}),
                                     rebuilt))
        # 본문 줄 순서가 바뀌거나, 줄이 다른 구역으로 옮겨지거나, 값이 다르면 같은 보고서가 아니다.
        for existing in (self.text(["처리 B", "문제점 A"], {"runs": {"split": "s", "export": "e"}, "reports": ["x", "y"]}),
                         self.text(["문제점 A"], {"runs": {"split": "s", "export": "e"}, "reports": ["x", "y"], "note": "처리 B"}),
                         self.text(["문제점 A", "처리 B"], {"runs": {"split": "s", "export": "o"}, "reports": ["x", "y"]})):
            self.assertFalse(_same_report(existing, rebuilt))


class CommonCoreBoundaryTest(unittest.TestCase):
    """공통 핵심은 데이터 형태·도메인·입력 형식 모듈을 import하지 않고 그 이름·필드명을 코드에 쓰지 않는다."""

    COMMON = ("src/agents/task_orchestrator.py", "src/agents/task_consent.py", "src/agents/task_store.py",
              "src/agents/task_plan.py", "src/agents/task_recommend.py", "src/agents/task_report.py",
              "src/agents/delegated_approval.py", "src/agents/strict_json.py", "src/ui/task_view.py")

    def test_common_core_has_no_type_specific_imports_or_names(self):
        import ast
        import re

        for name in self.COMMON:
            source = (PROJECT_ROOT / name).read_text(encoding="utf-8")
            imported = set()

            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    imported.add((node.module or "") + ":" + ",".join(alias.name for alias in node.names))
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)

            for item in imported:
                with self.subTest(file=name, imported=item):
                    self.assertIsNone(re.search(r"tabular|csv|llm_task_assist|wm_|wafer|profile_store|snapshot_split", item))

            code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))

            for word in ("waferMap", "failureType", "lotName", "wm_", "wafer_dl", "csv_split_review", "csv_prepare"):
                with self.subTest(file=name, word=word):
                    self.assertNotIn(word, code)


class TaskConsentTest(unittest.TestCase):
    def task(self, decisions):
        record = {"task_id": "a" * 32, "consents": {}}
        for decision, content in decisions.items():
            record["consents"][decision] = task_consent.grant(decision, record["task_id"], content)
        return record

    def preview(self, stage="audit", limits=LIMITS, **extra):
        return {"summary": {"stage": stage, "limits": limits, **extra}}

    def test_delegation_requires_matching_decision_scope_and_values(self):
        content = {"plan_signature": "p", "limits": LIMITS, "csv_code_id": "c", "binding_code_id": "b",
                   "allowed": [["csv_split_review", "audit"]]}
        record = self.task({"C": content})
        authorize = task_consent.authorizer(record, "C", "csv_split_review", "audit", expected=dict(content))
        authorize(self.preview())

        with self.assertRaises(ValueError):
            authorize(self.preview(limits=dict(LIMITS, memory_mib=4096)))

        with self.assertRaises(ValueError):
            authorize(self.preview(stage="commit_split"))

        for key in ("plan_signature", "csv_code_id", "limits"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                task_consent.authorizer(record, "C", "csv_split_review", "audit", expected=dict(content, **{key: "x"}))

        # 동의 내용의 허용 목록 밖 단계는 위임하지 않는다. B만으로는 어떤 단계도 위임하지 않는다.
        with self.assertRaises(ValueError):
            task_consent.authorizer(record, "C", "csv_prepare_apply", "apply_preparation", expected={})

        with self.assertRaises(ValueError):
            task_consent.authorizer(record, "D", "csv_prepare_apply", "apply_preparation", expected={})

        with self.assertRaises(ValueError):
            task_consent.grant("B", record["task_id"], {"allowed": [["m", "s"]]})

        # 허용 단계가 없는 동의는 만들지 않는다.
        with self.assertRaises(ValueError):
            task_consent.grant("C", record["task_id"], {"limits": LIMITS})

    def test_expired_tampered_or_foreign_consent_is_rejected(self):
        content = {"limits": LIMITS, "allowed": [["input_profile", "run_pipeline"]]}
        record = self.task({"A": content})
        later = datetime.now(timezone.utc) + timedelta(hours=13)

        with self.assertRaises(ValueError):
            task_consent.check(record["consents"]["A"], "A", record["task_id"], now=later)

        record["consents"]["A"]["content"]["limits"] = dict(LIMITS, processes=8)

        with self.assertRaises(ValueError):
            task_consent.check(record["consents"]["A"], "A", record["task_id"])

        fresh = self.task({"A": content})

        with self.assertRaises(ValueError):
            task_consent.check(fresh["consents"]["A"], "A", "b" * 32)

        with patch("src.agents.task_consent._current_actor", return_value="OTHER\\user"):
            with self.assertRaises(ValueError):
                task_consent.check(fresh["consents"]["A"], "A", fresh["task_id"])

        # 허용 목록을 고쳐 범위를 넓히면 내용 해시가 달라져 거부된다.
        widened = self.task({"A": content})
        widened["consents"]["A"]["content"]["allowed"].append(["csv_prepare_apply", "apply_preparation"])

        with self.assertRaises(ValueError):
            task_consent.authorizer(widened, "A", "csv_prepare_apply", "apply_preparation", expected={})

    def test_bound_values_are_checked_inside_lock(self):
        record = self.task({"A": {"limits": LIMITS, "allowed": [["input_profile", "run_pipeline"]]}})
        authorize = task_consent.authorizer(record, "A", "input_profile", "run_pipeline", expected={},
                                            bound={"snapshot_sha256": "s1"}, summary_stage=False)
        authorize({"summary": {"limits": LIMITS, "snapshot_sha256": "s1"}})

        with self.assertRaises(ValueError):
            authorize({"summary": {"limits": LIMITS, "snapshot_sha256": "s2"}})

        # 기본값은 단계 키가 없는 승인 요약을 거부한다(진단만 고정 단계·계획 서명으로 따로 결합).
        strict = task_consent.authorizer(record, "A", "input_profile", "run_pipeline", expected={})

        with self.assertRaises(ValueError):
            strict({"summary": {"limits": LIMITS}})


if __name__ == "__main__":
    unittest.main()

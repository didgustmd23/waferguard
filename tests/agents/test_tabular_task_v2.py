"""공통 작업 조정기 + v2 표형 어댑터: 요청 → 입력 검사·생성 → 진단 → 권장안 → 계획 → 처리 → 적용·전달 → 보고서를 합성 CSV로 검사한다.

OS transport(AppContainer·Job)와 로컬 모델 응답만 모의한다. 입력 검사·생성, 진단, v2 작업 계산, 승인 저장, 결과 검증은 실제 코드다.
"""

from dataclasses import asdict
import json
from pathlib import Path

from src.agents import task_adapters, task_orchestrator as orchestrator, task_store
from src.agents.csv_handoff_contract import load_handoff_partition
from src.agents.llm_config import load_local_config
from src.agents.llm_task_assist import PREPARATION_FIELDS
from src.agents.local_ollama import LocalProposalError
from src.agents.profile_store import SPLIT_FIELDS
from src.agents.task_plan import PlanError
from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules
from tests.agents.test_tabular_workflow_v2 import TabularWorkflowFixture
from tests.agents.test_task_orchestrator import model_reply


LIMITS = {"wall_seconds": 60, "memory_mib": 2048, "processes": 2}
ROLES = {"x": "수치형", "flag": "식별자", "label": "target", "g": "group", "id": "식별자"}
PREPARATION = {key: value for key, value in asdict(PreparationRules()).items() if key in PREPARATION_FIELDS}


def add_v2_patches(test, store, handoff_runs):
    """CSV v1 fixture(test_csv_workflow_service) 위에 v2 입력·서비스 경로의 모의를 더한다. 화면에서 새로 만든 작업은 v2로 간다.

    입력 도구 운송·실행 주체·v2 런타임 안내·관리 루트만 바꾸고, 기존 v1 결과 폴더 배치는 그대로 둔다.
    """
    from tests.agents import test_tabular_input as input_fixture

    test.start_patch("src.agents.tabular_workflow_service.TASK_ROOT", new=Path(store))
    test.start_patch("src.agents.tabular_input_service.run_input_worker", side_effect=input_fixture.transport)
    for module in ("tabular_input_service", "tabular_workflow_service"):
        test.start_patch("src.agents." + module + "._current_actor", return_value="synthetic_actor")
    test.start_patch("src.agents.tabular_workflow_service.TabularWorkflowService.runtime",
                     return_value={"supported": True, "notice": "synthetic OS only"})
    test.start_patch("src.agents.csv_handoff_workflow.directory_for",
                     side_effect=lambda name, **kw: Path(handoff_runs) / (name + "-v2" if kw.get("typed") else name))


class TabularTaskV2Fixture(TabularWorkflowFixture):
    """v2 서비스 fixture에 작업 저장소·진단 run 폴더·모델 응답을 더한다."""

    def setUp(self):
        super().setUp()
        self.store = self.root / "tasks"
        self.store.mkdir()
        # v2 서비스는 작업 폴더의 input_runs 아래 생성 실행만 신뢰한다. 작업 저장소를 같은 관리 루트로 둔다.
        self.start_patch("src.agents.tabular_workflow_service.TASK_ROOT", new=self.store)
        (self.root / "diag_runs").mkdir()
        self.start_patch("src.agents.pipeline_approval_cli._directory",
                         side_effect=lambda name, create_parent=False: self.root / "diag_runs" / name)
        self.model = asdict(load_local_config())
        self.requests = []
        self.split = {key: value for key, value in asdict(self.profile.split_rules).items() if key in SPLIT_FIELDS}
        self.base_split = {key: value for key, value in asdict(SplitRules()).items() if key in SPLIT_FIELDS}

    def requester(self, payload, config):
        # group 역할을 권하면 분할 방식·비율도 함께 권해야 계획 전체 검증을 통과한다.
        self.requests.append(payload)
        changes = [{"path": "split." + key, "value": value, "reason": "group 단위 분할"}
                   for key, value in self.split.items() if value != self.base_split.get(key)]
        return model_reply(ROLES, changes=changes)

    def refusing(self, payload, config):
        self.requests.append(payload)
        raise LocalProposalError("OLLAMA_CONNECTION_FAILED")

    def plan(self, roles=ROLES):
        return {"roles": dict(roles), "split": dict(self.split), "preparation": dict(PREPARATION)}

    def new_task(self, *, generation="current"):
        return orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", self.source, LIMITS, self.model, root=self.store,
                                       input_generation=generation)

    def diagnosed(self, requester=None):
        task = self.new_task()
        self.assertEqual(task["kind"], "tabular_v2")
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=requester or self.requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        return task


class TabularTaskV2Test(TabularTaskV2Fixture):
    def test_current_csv_runs_v2_from_input_to_report_with_typed_handoff(self):
        before = self.source.read_bytes()
        task = self.diagnosed()
        selection = task["input_selection"]
        self.assertEqual(selection["status"], "materialized")
        self.assertEqual(selection["attempt"], 1)
        self.assertEqual(task["draft"]["source"], "model")
        self.assertEqual(task["diagnostic"]["summary"]["row_count"], 6)
        self.assertEqual([column["column"] for column in task["diagnostic"]["columns"]], ["x", "flag", "label", "g", "id"])
        # 앞자리 0 ID는 생성된 표에서 문자열로 보존되고 진단도 같은 분류를 쓴다.
        self.assertEqual({column["column"]: column["dtype_group"] for column in task["diagnostic"]["columns"]}["id"], "text")
        self.assertEqual(set(task["consents"]), {"A"})
        source_id = task_adapters.adapter("tabular_v2")._source(task).source_id
        self.assertEqual(task["diagnostic"]["source_id"], source_id)

        task = orchestrator.confirm_plan(task["task_id"], self.plan(), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(set(task["steps"]), {"split", "export", "candidate"})
        self.assertEqual({item["module_id"] for item in task["steps"].values()},
                         {"tabular_split_review", "tabular_partition_export", "tabular_prepare_candidate"})
        review = {item["label"]: item["value"] for item in task["candidate"]["facts"]["metrics"]}
        self.assertEqual(review["Train 행 수"], 3)
        self.assertIn("특징별 처리", [table["title"] for table in task["candidate"]["facts"]["tables"]])
        self.assertIn("후보", task["candidate"]["facts"]["text"])
        self.assertEqual(self.calls.count("apply_preparation"), 0)
        self.assertEqual(task["consents"]["C"]["content"]["source_contract_id"], source_id)

        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertEqual(self.calls.count("apply_preparation"), 1)
        self.assertEqual(self.source.read_bytes(), before)
        facts = task["report"]["facts"]
        self.assertTrue(Path(task["report"]["path"]).is_file())
        self.assertIn("로컬 전달 묶음", task["report"]["text"])
        self.assertEqual({item["label"]: item["value"] for item in facts["metrics"]}["입력 행"], 6)
        comparison = {row["항목"]: row for row in facts["comparison"]}
        self.assertEqual(comparison["입력 행"]["후 (실측)"], "train 3행 변환 · validation 2행 변환")
        self.assertTrue(any(line.startswith("입력 해석: ") for line in facts["lines"]))
        self.assertTrue(any("load_handoff_partition" in line for line in facts["usage"]))
        uses = {row["파일"]: row["용도"] for row in facts["files"]}
        self.assertIn("원래 자료형 보존", uses["prepared/train_rows.json"])
        self.assertEqual(facts["technical"]["source_contract_id"], source_id)
        self.assertEqual(len(facts["technical"]["manifest_sha256"]), 64)
        self.assertTrue(any(row["컬럼"] == "id" and row["역할"] == "식별자" for row in facts["processing"]))
        # 결과 파일은 자료형을 보존한 reader로 바로 쓸 수 있다.
        features, rows = load_handoff_partition(Path(facts["outputs"]["bundle"]), "train")
        self.assertEqual(rows["id"].tolist(), ["0012", "0034", "0056"])
        self.assertEqual(rows["flag"].tolist(), [1, 0, 1])
        self.assertEqual(len(features), 3)
        self.assertEqual(set(task["consents"]), {"A", "C", "D"})

    def test_model_refusal_uses_default_plan_and_numeric_role_type_is_not_auto_fixed(self):
        task = self.diagnosed(self.refusing)
        self.assertEqual(task["draft"]["source"], "default")
        self.assertEqual(len(self.requests), 1)
        wrong = dict(ROLES, id="수치형")
        with self.assertRaises(PlanError) as caught:
            orchestrator.preview_plan(task, self.plan(wrong), {})
        self.assertEqual(caught.exception.code, "NUMERIC_ROLE_TYPE_UNSUPPORTED")
        self.assertEqual(tuple(caught.exception.subjects), ("roles.id",))
        with self.assertRaises(PlanError):
            orchestrator.confirm_plan(task["task_id"], self.plan(wrong), root=self.store)
        stored = task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["phase"], stored["steps"]), ("awaiting_review", {}))
        self.assertNotIn("C", stored["consents"])
        # 기본 계획에서 시작해도 사용자가 고친 계획으로 끝까지 진행한다.
        task = orchestrator.confirm_plan(task["task_id"], self.plan(), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertEqual(task["report"]["facts"]["recommendation"]["source"], "default")


class TabularTaskV2DiagnosticTest(TabularTaskV2Fixture):
    def test_diagnostic_table_is_read_from_sealed_bytes_and_bound_to_this_source(self):
        task = self.diagnosed()
        adapter = task_adapters.adapter(task["kind"])
        name, source = task["diagnostic"]["run_name"], adapter._source(task)
        self.assertEqual(adapter.diagnostic_table(name, source), (task["diagnostic"]["summary"], task["diagnostic"]["columns"]))
        run = self.root / "diag_runs" / name
        table = next(run.rglob("attempts/*/pipeline/*/input_profile.csv"))
        original = table.read_bytes()
        # 크기는 같고 내용만 다른 교체: 진단 검증 뒤 바뀌면 완료 표식과 달라 거부한다.
        table.write_bytes(original[:-2] + (b"9" if original[-2:-1] != b"9" else b"8") + original[-1:])
        with self.assertRaises(ValueError):
            adapter.diagnostic_table(name, source)
        table.write_bytes(original)
        self.assertEqual(adapter.diagnostic_table(name, source)[1], task["diagnostic"]["columns"])

        # 진단 검증(committed_diagnostic)이 끝난 직후 같은 크기로 바꾸면 표식에 묶인 바이트 읽기가 거부한다.
        from dataclasses import replace
        from unittest.mock import patch
        from src.agents import tabular_v2_task

        real = tabular_v2_task.committed_diagnostic
        swapped = []

        def tamper_after_check(run, module_id):
            result = real(run, module_id)
            swapped.append(module_id)
            table.write_bytes(original[:-2] + (b"9" if original[-2:-1] != b"9" else b"8") + original[-1:])
            return result

        try:
            with patch.object(tabular_v2_task, "committed_diagnostic", side_effect=tamper_after_check):
                with self.assertRaisesRegex(ValueError, "완료 기록과 다릅니다"):
                    adapter.diagnostic_table(name, source)
            self.assertEqual(swapped, ["input_profile"])
        finally:
            table.write_bytes(original)

        # 다른 생성 출처를 넘기면 진단 run 바인딩의 출처와 달라 거부한다.
        with self.assertRaisesRegex(ValueError, "생성된 표와 다릅니다"):
            adapter.diagnostic_table(name, replace(source, inspection_id="0" * 64))

        # 저장된 v2 선택 기록의 출처를 바꾸면 진단 run을 열지 않는다.
        selection = run / "selection.json"
        saved = selection.read_bytes()
        value = json.loads(saved)
        value["source"]["inspection_id"] = "0" * 64
        selection.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(ValueError):
            adapter.diagnostic_table(name, source)
        selection.write_bytes(saved)


class TabularTaskV2QualityTest(TabularTaskV2Fixture):
    csv_text = ("x,flag,label,g,id,c\n1.0,1,True,a,0012,5\n,0,False,a,0034,5\n9.0,1,True,a,0056,5\n"
                "100.0,0,False,b,0078,5\n,1,True,b,0090,5\n700.0,0,False,c,0102,5\n")

    def test_quality_review_stops_split_and_shows_verified_findings_for_revision(self):
        task = self.diagnosed()
        task = orchestrator.confirm_plan(task["task_id"], self.plan(dict(ROLES, c="수치형")), root=self.store)
        problem = task["problem"]
        # CSV v1과 같이 정지하되 같은 작업에서 계획을 고쳐 이어 갈 수 있다(계획 확인 화면이 열린다).
        self.assertEqual((task["phase"], problem["code"], problem["retryable"]), ("needs_attention", "QUALITY_REVIEW", True))
        self.assertTrue(orchestrator.revisable(task))
        # 멈춘 품질 검사는 커밋되지 않아 완료 표식이 없다. 지적은 한 번 읽은 비공개 사본을 검증한 결과만 보인다.
        self.assertIn({"code": "CONSTANT_FEATURE", "subject": "c"}, problem["findings"])
        adapter = task_adapters.adapter(task["kind"])
        self.assertEqual(adapter.quality_findings(task), problem["findings"])
        self.assertEqual(self.calls.count("derive_preparation"), 0)

        # 보고서를 바꾸거나 다른 파일을 두면 검증을 통과하지 못해 지적을 보이지 않는다(추정해 채우지 않음).
        report = next((self.root / "split").rglob("attempts/*/quality/quality.json"))
        saved = report.read_bytes()
        try:
            value = json.loads(saved)
            # 다른 실제 특징으로 지적 대상을 바꾼 보고서는 함께 저장된 안내문과 맞지 않는다.
            value["findings"][0]["column"] = "x"
            report.write_text(json.dumps(value), encoding="utf-8")
            self.assertIsNone(adapter.quality_findings(task))
            report.write_bytes(saved)
            (report.parent / "extra.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(adapter.quality_findings(task))
        finally:
            report.write_bytes(saved)
            (report.parent / "extra.json").unlink(missing_ok=True)
        self.assertEqual(adapter.quality_findings(task), problem["findings"])
        task = orchestrator.confirm_plan(task["task_id"], self.plan(dict(ROLES, c="제외")), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))


class TabularTaskV2InputTest(TabularTaskV2Fixture):
    csv_text = "x,x,label\n1.0,2.0,a\n3.0,4.0,b\n5.0,6.0,a\n"

    def test_duplicate_headers_wait_then_new_selection_continues_and_stale_submissions_are_rejected(self):
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_input", task.get("problem"))
        self.assertEqual(self.requests, [])
        self.assertIsNone(task["input_selection"]["materialize_run"])
        view = task_adapters.adapter(task["kind"]).input_review(task)
        self.assertEqual(view["code"], "DUPLICATE_COLUMN_NAMES")
        self.assertEqual(view["detail"], {"duplicated_names": 1, "affected_columns": 2})
        self.assertEqual(view["choices"]["header_repair"]["current"], "reject")
        stale = task["revision"]
        consent = task["consents"]["A"]["consent_id"]

        for selection, options in (({"header_repair": "reject"}, {}),
                                   ({"header_repair": "number_duplicates"}, {"expected_revision": stale - 1}),
                                   ({"header_repair": "number_duplicates"}, {"expected_inspection_id": "0" * 64}),
                                   ({"header_repair": "number_duplicates", "encoding": "cp949"}, {})):
            with self.subTest(selection=selection, options=options), self.assertRaises(ValueError):
                orchestrator.resume_input(task["task_id"], selection, expected_revision=options.get("expected_revision", stale),
                                          expected_inspection_id=options.get("expected_inspection_id", view["inspection_id"]),
                                          root=self.store, requester=self.requester)
        unchanged = task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((unchanged["revision"], unchanged["consents"]["A"]["consent_id"]), (stale, consent))

        task = orchestrator.resume_input(task["task_id"], {"header_repair": "number_duplicates"}, expected_revision=stale,
                                         expected_inspection_id=view["inspection_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        self.assertNotEqual(task["consents"]["A"]["consent_id"], consent)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual([column["column"] for column in task["diagnostic"]["columns"]], ["x_1", "x_2", "label"])
        notices = task_adapters.adapter(task["kind"]).input_notices(task)
        self.assertTrue(any("번호를 붙였습니다" in line for line in notices))
        # 같은 대기 상태로 다시 제출하면 거부한다(중복 제출).
        with self.assertRaises(ValueError):
            orchestrator.resume_input(task["task_id"], {"header_repair": "number_duplicates"}, expected_revision=stale,
                                      expected_inspection_id=view["inspection_id"], root=self.store)


class TabularTaskV2SelectionTest(TabularTaskV2Fixture):
    def test_generations_choose_fixed_kinds_and_legacy_records_keep_their_kind(self):
        self.assertEqual(task_adapters.select(self.source), "tabular_csv")
        self.assertEqual(task_adapters.select(self.source, generation="current"), "tabular_v2")
        self.assertEqual(task_adapters.select(self.root / "a.npz", generation="current"), "array_npz")
        self.assertEqual([item["kind_id"] for item in task_adapters.input_specs()], ["tabular_csv", "array_npz"])
        self.assertEqual([item["kind_id"] for item in task_adapters.input_specs(generation="current")], ["tabular_v2", "array_npz"])
        for bad in ("next", None, "tabular_v2"):
            with self.subTest(generation=bad), self.assertRaises(ValueError):
                task_adapters.select(self.source, generation=bad)
        legacy = self.new_task(generation="legacy")
        self.assertEqual(legacy["kind"], "tabular_csv")
        self.assertIs(task_adapters.adapter(task_store.load_task(legacy["task_id"], root=self.store)["kind"]),
                      task_adapters.ADAPTERS["tabular_csv"])
        with self.assertRaises(ValueError):
            orchestrator.start_task("목표", self.source, LIMITS, self.model, root=self.store, input_generation="tabular_v2")


if __name__ == "__main__":
    import unittest

    unittest.main()

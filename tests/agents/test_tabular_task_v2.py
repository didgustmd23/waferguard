"""공통 작업 조정기 + v2 표형 어댑터: 요청 → 입력 검사·생성 → 진단 → 권장안 → 계획 → 처리 → 적용·전달 → 보고서를 합성 CSV로 검사한다.

OS transport(AppContainer·Job)와 로컬 모델 응답만 모의한다. 입력 검사·생성, 진단, v2 작업 계산, 승인 저장, 결과 검증은 실제 코드다.
"""

from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.agents import task_adapters, task_orchestrator as orchestrator, task_store
from src.agents.csv_handoff_contract import load_handoff_partition
from src.agents.llm_config import load_local_config
from src.agents.llm_task_assist import PREPARATION_FIELDS
from src.agents.local_ollama import LocalProposalError
from src.agents.profile_store import SPLIT_FIELDS
from src.agents.task_plan import PlanError, TaskProblem
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


class TabularTaskV2SourceBindingTest(TabularTaskV2Fixture):
    def test_record_source_must_come_from_this_tasks_materialize_run(self):
        """다른 작업의 온전한 출처로 바꾼 기록은 계획 확정(동의 C) 전에 거부한다. 두 작업의 입력 바이트가 같아도 거부한다."""
        first, second = self.diagnosed(), self.diagnosed()
        record = task_store.load_task(first["task_id"], root=self.store)
        other = task_store.load_task(second["task_id"], root=self.store)
        record["input_selection"]["source"] = other["input_selection"]["source"]
        task_store.save_task(record, expected_revision=record["revision"], root=self.store)
        with self.assertRaises((ValueError, TaskProblem)):
            orchestrator.confirm_plan(first["task_id"], self.plan(), root=self.store)
        self.assertNotIn("C", task_store.load_task(first["task_id"], root=self.store)["consents"])

    def test_split_profile_must_match_the_consented_profile_signature(self):
        """분할 규칙이 같아도 역할이 다른 프로파일로 바꾼 경로는 위임 전에 거부한다(동의 C의 profile_signature 대조)."""
        from dataclasses import replace
        from src.agents import tabular_v2_task as adapter
        from src.agents.profile_store import save_profile

        original = adapter.prepare

        def swapped(record, directory, plan, label):
            prepared = original(record, directory, plan, label)
            profile, _ = adapter.validate_plan(record, plan)
            # flag(식별자)를 수치형 특징으로 바꾼 같은 분할 규칙의 온전한 프로파일
            other = replace(profile, profile_id=label + "_other", adapter_id=adapter.PROFILE_ADAPTER,
                            roles=replace(profile.roles, numeric=("x", "flag"), identifiers=("id",)))
            saved = save_profile(other, root=directory / "profiles")
            self.assertNotEqual(saved["profile_signature"], prepared["profile_signature"])
            return dict(prepared, profile_path=saved["local_file"])

        with patch.object(adapter, "prepare", side_effect=swapped):
            task = orchestrator.confirm_plan(self.diagnosed()["task_id"], self.plan(), root=self.store)
        self.assertNotEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(self.calls.count("derive_preparation"), 0)
        self.assertNotIn("audit", self.calls)

    def test_saved_workflow_must_match_the_confirmed_plan_signature(self):
        """같은 설정이라도 서명이 다른 저장 실행안으로 바꾸면 단계 생성 전에 거부한다(동의에 결합한 계획 서명 대조)."""
        import tempfile
        from src.agents import llm_workflow_store as plans, tabular_task as shared, tabular_v2_task as adapter

        task = orchestrator.confirm_plan(self.diagnosed()["task_id"], self.plan(), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        directory = task_store.task_directory(task["task_id"], root=self.store)
        profile, preparation = adapter.validate_plan(task, task["confirmed"]["plan"])
        proposal, context = adapter._proposal("other_label", profile, preparation)
        self.assertNotEqual(proposal.plan_signature, task["confirmed"]["plan_signature"])
        with tempfile.TemporaryDirectory() as scratch:
            plans.save_workflow(task["confirmed"]["plan_name"], proposal, context, root=Path(scratch))
            forged = (Path(scratch) / task["confirmed"]["plan_name"] / "plan.json").read_bytes()
        (shared._plan_root(directory) / task["confirmed"]["plan_name"] / "plan.json").write_bytes(forged)
        # 결정 D를 발급하지 않고 멈추며 계획 수정(새 시도·새 동의)으로 이어 갈 수 있다.
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual((task["phase"], task["problem"]["code"]), ("needs_attention", "PLAN_STALE"))
        self.assertNotIn("D", task["consents"])
        self.assertTrue(orchestrator.revisable(task))
        self.assertNotIn("apply", task["steps"])
        self.assertEqual(self.calls.count("apply_preparation"), 0)

    def test_consent_c_from_another_attempt_is_not_delegated(self):
        """동의 C의 시도 번호가 현재 시도와 다르면(이전 시도로 되돌린 기록) 위임 승인을 발급하지 않는다."""
        from src.agents import task_consent

        original = task_consent.grant

        def stale(decision, task_id, content, **options):
            return original(decision, task_id, dict(content, attempt=content["attempt"] - 1) if decision == "C" else content, **options)

        with patch.object(task_consent, "grant", side_effect=stale):
            task = orchestrator.confirm_plan(self.diagnosed()["task_id"], self.plan(), root=self.store)
        self.assertNotEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertNotIn("audit", self.calls)


class TabularTaskV2RecommendationRejectionTest(TabularTaskV2Fixture):
    def test_rejected_split_recommendation_is_stored_with_limited_diagnostic_and_explained(self):
        """v2에서도 group·계층 분할 권장은 역할·분할 방식 불일치로 전체 거부되고, 저장·재조회한 제한 진단으로 안내한다."""
        from src.agents.task_report import recommendation_notice

        def requester(payload, config):
            self.requests.append(payload)
            return model_reply(ROLES, changes=[{"path": "split.strategy", "value": "group", "reason": "이유"},
                                               {"path": "split.stratify", "value": True, "reason": "이유"}])

        task = self.diagnosed(requester=requester)
        stored = task_store.load_task(task["task_id"], root=self.store)
        draft = stored["draft"]
        self.assertEqual((draft["source"], draft["code"], draft["changes"]), ("default", "SPLIT_ROLE_MISMATCH", []))
        self.assertEqual(draft["rejection"]["reason"], "SPLIT_STRATIFY_STRATEGY")
        self.assertEqual(draft["rejection"]["from_model"], ["strategy", "stratify"])
        self.assertEqual(draft["plan"], task_adapters.adapter("tabular_v2").base_plan(stored))
        lines = recommendation_notice(stored)
        self.assertTrue(lines[0].startswith("로컬 LLM 권장안 전체를 거부해 기본 계획에서 시작했습니다"), lines)
        self.assertIn("권장안의 변경은 적용하지 않았습니다", lines[0])
        self.assertTrue(any(line.startswith("세부 원인: target 기준 계층 분할을") for line in lines), lines)
        self.assertFalse(any("역할" in line and "버려" in line for line in lines), lines)

        # 하위 진단이 없는 이전 기록(SPLIT_INVALID)은 상세 원인 미확인으로 보이고, 정상 읽기·안내가 된다.
        old = dict(stored, draft={key: value for key, value in dict(draft, code="SPLIT_INVALID").items() if key != "rejection"})
        lines = recommendation_notice(old)
        self.assertTrue(lines[0].startswith("로컬 LLM 권장안 전체를 거부해 기본 계획에서 시작했습니다"), lines)
        self.assertIn("세부 원인: 확인되지 않음(이 기록에는 하위 진단이 없습니다)", lines)
        # 형태가 맞지 않는(손상·변조된) 하위 진단은 해석하지 않고 상세 원인 미확인으로 보인다.
        for broken in ("문자열", {"reason": "SPLIT_RATIO_SUM"}, dict(draft["rejection"], values=["x"])):
            lines = recommendation_notice(dict(stored, draft=dict(draft, rejection=broken)))
            self.assertIn("세부 원인: 확인되지 않음(하위 진단을 읽지 못했습니다)", lines)
        # 모델 요청 자체가 실패한 기록은 '거부' 안내를 쓰지 않는다.
        failed = dict(stored, draft={key: value for key, value in dict(draft, code="OLLAMA_CONNECTION_FAILED").items()
                                     if key != "rejection"})
        self.assertTrue(recommendation_notice(failed)[0].startswith("로컬 LLM 권장안을 쓰지 못해 기본 계획에서 시작했습니다"))

    def test_tampered_stored_rejection_is_not_interpreted_after_reload(self):
        """정상 비율 합 진단은 저장·재조회 후 그대로 보이고, 저장 기록을 변조하면 재조회 후 값을 해석하지 않는다."""
        from src.agents.task_report import recommendation_notice
        from tests.agents.test_task_orchestrator import UNREADABLE_REJECTION, tampered_rejections

        def requester(payload, config):
            self.requests.append(payload)
            return model_reply(ROLES, changes=[{"path": "split.train_ratio", "value": 0.6, "reason": "이유"}])

        task = self.diagnosed(requester=requester)
        stored = task_store.load_task(task["task_id"], root=self.store)
        draft = stored["draft"]
        self.assertEqual((draft["code"], draft["rejection"]["reason"], draft["rejection"]["from_model"]),
                         ("SPLIT_INVALID", "SPLIT_RATIO_SUM", ["train_ratio"]))
        self.assertTrue(task_adapters.adapter("tabular_v2").rejection_valid(draft["code"], draft["rejection"]))
        lines = recommendation_notice(stored)
        self.assertIn("세부 원인: Train·Validation·Test 비율의 합이 1이 아닙니다", lines)
        # 저장 후 다시 읽은 기록은 키가 정렬되므로 값의 위치가 아니라 포함 여부를 본다.
        self.assertTrue(any(line.startswith("거부한 분할 값(병합 결과):") and "train_ratio=0.6 " in line for line in lines), lines)

        cases = tampered_rejections(draft["rejection"])
        for name in ("unknown_value_key", "ratio_text", "unknown_section"):
            with self.subTest(case=name):
                current = task_store.load_task(task["task_id"], root=self.store)
                task_store.save_task(dict(current, draft=dict(current["draft"], rejection=cases[name])),
                                     expected_revision=current["revision"], root=self.store)
                lines = recommendation_notice(task_store.load_task(task["task_id"], root=self.store))
                self.assertTrue(lines[0].startswith("로컬 LLM 권장안 전체를 거부해 기본 계획에서 시작했습니다"), lines)
                self.assertEqual(lines[1:], [UNREADABLE_REJECTION])

        for name, broken in cases.items():
            with self.subTest(case=name):
                self.assertEqual(recommendation_notice(dict(stored, draft=dict(draft, rejection=broken)))[1:], [UNREADABLE_REJECTION])


class TabularTaskV2SplitClassTest(TabularTaskV2Fixture):
    """그룹 c(False만)가 한 구간을 채우면 그 구간에 True가 없다. 기본 조건(모든 구간에 모든 클래스)은 분할 제안에서 멈춘다."""

    def test_class_missing_stop_names_the_split_cause_and_the_existing_setting_continues(self):
        task = self.diagnosed()
        strict = self.plan()
        strict["split"]["require_all_classes"] = True
        task = orchestrator.confirm_plan(task["task_id"], strict, root=self.store)
        problem = task["problem"]
        self.assertEqual((task["phase"], problem["code"], problem["retryable"]), ("needs_attention", "SPLIT_CLASS_MISSING", True))
        self.assertTrue(orchestrator.revisable(task))
        # 품질 검사는 통과했으므로 품질 지적 문구·제외 안내를 쓰지 않는다. 빠진 클래스는 검증한 제안 근거에서만 읽는다.
        self.assertNotIn("품질 검사가 검토 필요", problem["message"])
        self.assertIn("모든 구간에 모든 클래스 요구", problem["message"])
        # 작업 기록에는 구간 이름·개수만 저장하고 라벨 값은 두지 않는다. 화면 view만 검증한 근거에서 라벨을 다시 읽어 보인다.
        self.assertEqual(problem["findings"], [{"code": "CLASS_MISSING_IN_SPLIT", "subject": "Test"}])
        self.assertEqual(problem["detail"], {"CLASS_MISSING_IN_SPLIT": 1})
        stored = task_store.load_task(task["task_id"], root=self.store)
        self.assertNotIn("True", json.dumps(stored["problem"], ensure_ascii=False))
        labelled = [{"code": "CLASS_MISSING_IN_SPLIT", "subject": "Test · True"}]
        self.assertEqual(orchestrator.problem_view(stored)["findings"], labelled)
        self.assertEqual(self.calls.count("derive_preparation"), 0)

        # 근거 파일을 바꾸거나 다른 파일을 두면 검증을 통과하지 못해 클래스를 보이지 않는다(추정해 채우지 않음).
        adapter = task_adapters.adapter(task["kind"])
        proposal = next((self.root / "split").rglob("attempts/*/proposal.json"))
        saved = proposal.read_bytes()
        try:
            value = json.loads(saved)
            value["summary"][2]["classes"][0]["label"] = "True"
            proposal.write_text(json.dumps(value), encoding="utf-8")
            unverified = adapter.blocked_problem(task, "split")
            self.assertEqual((unverified.code, unverified.findings), ("SPLIT_REVIEW", []))
            self.assertIn("확인하지 못했습니다", unverified.message)
            # 저장된 정지의 화면 표시도 라벨을 추정해 채우지 않고 구간 이름만 보인다.
            self.assertEqual(orchestrator.problem_view(task)["findings"], problem["findings"])
            proposal.write_bytes(saved)
            (proposal.parent / "extra.json").write_text("{}", encoding="utf-8")
            self.assertEqual(adapter.blocked_problem(task, "split").findings, [])
        finally:
            proposal.write_bytes(saved)
            (proposal.parent / "extra.json").unlink(missing_ok=True)

        # 이전 형식 기록(품질 문구·빈 지적)도 화면에서는 확인된 원인으로 다시 읽는다. 기록은 바꾸지 않는다.
        old = dict(task, problem=dict(problem, code="QUALITY_REVIEW", message="분할 전 품질 검사가 검토 필요 또는 차단으로 중단했습니다.",
                                      findings=[], detail={}))
        view = orchestrator.problem_view(old)
        self.assertEqual((view["code"], view["findings"]), ("SPLIT_CLASS_MISSING", labelled))
        self.assertEqual(view["message"], problem["message"])
        self.assertEqual(old["problem"]["code"], "QUALITY_REVIEW")

        # 기존 분할 설정(클래스 누락 허용)으로 계획을 고치면 새 동의로 같은 작업에서 이어 간다.
        task = orchestrator.confirm_plan(task["task_id"], self.plan(), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertFalse(task["confirmed"]["plan"]["split"]["require_all_classes"])
        # 분할 확정 결과에서 확인한 클래스 누락은 주요 문제(확인된 클래스 누락)로, 사용자가 조건을 해제해 진행한 사실과 함께 남긴다.
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        facts = task["report"]["facts"]
        rows = [row for row in facts["problems"] if row["구분"] == "확인된 클래스 누락"]
        self.assertEqual(len(rows), 1, facts["problems"])
        self.assertEqual((rows[0]["컬럼"], rows[0]["처리"]), ("label", "사용자가 '모든 구간에 모든 클래스 요구'를 해제해 진행함"))
        self.assertTrue(rows[0]["내용"].startswith("test 구간에 클래스 True 없음(분할 확정 결과)"))
        self.assertFalse(any("클래스" in item["내용"] for item in facts["warnings"] if item["구분"] == "미검증"))


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


class TabularTaskV2FingerprintTest(unittest.TestCase):
    """v2 어댑터는 동의 A(입력·진단 위임)와 C·D(v2 작업 위임)의 내용·대조값을 정한다. 바뀌면 그 동의의 지문이 달라져야 한다."""

    ADAPTER = "tabular_v2_task.py"

    def fingerprints(self):
        from src.agents import csv_workflow_creation, tabular_input_service, tabular_v2_task, tabular_workflow_service
        from src.agents.llm_workflow_binding import binding_code_id

        return {"diagnostic_code_id": tabular_v2_task._diagnostic_code(), "input_code_id": tabular_input_service.code_id(),
                "tabular_code_id": tabular_workflow_service.code_id(), "binding_code_id": binding_code_id(),
                "csv_code_id": csv_workflow_creation.code_id()}

    def test_adapter_change_changes_every_consent_fingerprint_but_not_csv_v1(self):
        from src.agents import artifacts

        before = self.fingerprints()
        real_read, real_fingerprint = Path.read_bytes, artifacts._fingerprint

        def read_bytes(path):
            data = real_read(path)
            return data + b"\n# changed\n" if path.name == self.ADAPTER else data

        def fingerprint(path):
            result = real_fingerprint(path)
            return dict(result, sha256="0" * 64) if Path(path).name == self.ADAPTER else result

        with patch.object(Path, "read_bytes", read_bytes), \
                patch("src.agents.module_pipeline_transport._fingerprint", side_effect=fingerprint), \
                patch("src.agents.tabular_input_transport._fingerprint", side_effect=fingerprint):
            after = self.fingerprints()

        # 진단 위임(A)·입력 위임(A)·v2 작업 위임(C·D)·바인더 지문이 바뀐다. v1 CSV 작업 지문은 v2 어댑터와 무관하다.
        self.assertEqual({key for key in before if before[key] != after[key]},
                         {"diagnostic_code_id", "input_code_id", "tabular_code_id", "binding_code_id"})

    def test_saved_plan_store_and_module_plan_changes_change_binding_fingerprint(self):
        """저장 실행안 검사(store)와 결합 검사가 쓰는 전처리 모듈 계획 해석이 바뀌면 바인더 지문이 달라진다."""
        before = self.fingerprints()["binding_code_id"]
        real_read = Path.read_bytes

        for name in ("llm_workflow_store.py", "preparation_module_plan.py", "tabular_input_flow.py"):
            def read_bytes(path, name=name):
                data = real_read(path)
                return data + b"\n# changed\n" if path.name == name else data

            with self.subTest(file=name), patch.object(Path, "read_bytes", read_bytes):
                self.assertNotEqual(self.fingerprints()["binding_code_id"], before)


if __name__ == "__main__":
    unittest.main()

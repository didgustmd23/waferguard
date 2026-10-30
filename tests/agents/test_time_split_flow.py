"""표형 v2 time 분할: 시간 복원 결과(파생 표 또는 복원 0행이면 기존 생성 표)를 출처로 고정해 분할·내보내기·후보·적용·전달까지 진행한다.

합성 CSV fixture(test_tabular_task_v2)의 실제 조정기·어댑터·입력/시간/분할/내보내기 서비스와 부모 검증 코드를 쓴다.
OS 운송·공유 런타임 ID·모델 응답만 모의한다. 날짜는 고정 합성 값이다.
"""

import json
from pathlib import Path
import unittest
import unittest.mock

from src.agents import task_adapters, task_consent, task_orchestrator as orchestrator, task_store
from src.agents import tabular_time_flow as time_flow
from src.agents.csv_workflow_creation import _split
from src.agents.task_plan import PlanError
from src.preprocessing.tabular_bundle import inspect_bundle, load_bundle
from tests.agents.test_time_restore_flow import TIME, TimeRestoreFlowFixture


FULL_CSV = ("x,flag,label,g,id,t\n1.0,1,True,a,0012,2026-10-19\n2.0,0,False,a,0034,2026-10-20\n9.0,1,True,a,0056,2026-10-21\n"
            "100.0,0,False,b,0078,2026-10-22\n3.0,1,True,b,0090,2026-10-23\n700.0,0,False,c,0102,2026-10-24\n")


class TimeSplitFixture(TimeRestoreFlowFixture):
    def restored(self, plan):
        """T → R로 파생 표를 만든 작업."""
        task = self.diagnosed()
        task = self.act(task, "time_preview", plan)
        view = self.preview(task, plan)["plan_actions"]
        task = self.act(task, "time_apply", plan, expected_preview=view["expected"])
        self.assertEqual(task["time_restore"]["status"], "applied", task["time_restore"])
        return task

    def interrupt(self, task):
        """완료 저장 전에 멈춘 적용처럼 기록을 되돌린다(완료 복구 대상)."""
        record = self.stored(task)
        record.update(phase="applying", report=None)
        task_store.save_task(record, expected_revision=record["revision"], root=self.store)

    def confirm(self, task, plan):
        shown = self.preview(task, plan)
        return orchestrator.confirm_plan(task["task_id"], plan, root=self.store, expected_digest=shown["digest"])

    def assert_not_confirmed(self, task, plan, code):
        before = self.stored(task)
        with self.assertRaises(PlanError) as caught:
            self.confirm(task, plan)
        self.assertEqual(caught.exception.code, code)
        after = self.stored(task)
        self.assertEqual((after["attempt"], after["steps"], "C" in after["consents"], after["phase"]),
                         (before["attempt"], {}, False, before["phase"]))
        return caught.exception


class RestoredTimeSplitTest(TimeSplitFixture):
    def test_derived_table_is_split_in_time_order_exported_and_reported(self):
        plan = self.time_plan()
        task = self.restored(plan)
        derived = time_flow.restored_source(self.stored(task))
        original = time_flow._source(task)
        parent_files = {path.name: path.read_bytes() for path in Path(original.bundle_directory).iterdir()}
        shown = self.preview(task, plan)
        self.assertIsNone(shown["blocked"])
        self.assertTrue(any("파생 표(복원 2행)" in text for _, text in shown["plan_actions"]["lines"]))

        task = self.confirm(task, plan)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        confirmed = task["confirmed"]
        self.assertEqual((confirmed["source_kind"], confirmed["time_source"]["kind"], confirmed["time_source"]["restored_count"]),
                         ("time_restored", "time_restored", 2))
        content = task["consents"]["C"]["content"]
        self.assertEqual((content["source_contract_id"], content["input_source_contract_id"]), (derived.source_id, original.source_id))
        self.assertEqual(content["time_source"], confirmed["time_source"])
        # 분할 규칙의 시간 형식은 확정 time 설정과 같다. 분할·내보내기 실행 계약의 출처는 파생 표다.
        split, _, evidence = _split(task["steps"]["split"]["run_name"], typed=True)
        self.assertEqual(split.contract.source_id, derived.source_id)
        self.assertEqual(split.contract.profile.split_rules.timestamp_format, TIME["timestamp_format"])
        self.assertEqual(split.contract.profile.split_rules.strategy, "time")
        # 시간 순서(독립 정답: 합성 날짜 6개를 1/3씩)와 경계
        bounds = [(row["split"], row["time_min"][:10], row["time_max"][:10]) for row in evidence.proposal["summary"]]
        self.assertEqual(bounds, [("train", "2026-10-19", "2026-10-20"), ("validation", "2026-10-21", "2026-10-22"),
                                  ("test", "2026-10-23", "2026-10-24")])
        export = next(Path(item["path"]) for item in task_adapters.adapter("tabular_v2").service("tabular_partition_export")
                      .results(task["steps"]["export"]["run_name"])["items"] if item["kind"] == "artifact_list" and item["label"] == "ingest")
        rows = {}
        for name in ("train", "validation"):
            reference = inspect_bundle(export / name, expected=None, max_bytes=8 * 1024 * 1024)
            rows[name] = load_bundle(export / name, expected=reference)["t"].tolist()
        # 복원한 시간 셀(2026-10-20, 2026-10-23 중 Train·Validation에 있는 값)이 내보낸 구간에 그대로 있다.
        self.assertEqual(rows, {"train": ["2026-10-19", "2026-10-20"], "validation": ["2026-10-21", "2026-10-22"]})
        # 결정 D 화면에 사용 출처와 구간별 시간 범위가 보인다.
        table = next(item for item in task["candidate"]["facts"]["tables"] if item["title"] == "time 분할 구간별 시간 범위")
        self.assertTrue(table["notes"][0].startswith("입력 출처: 파생 표(시간 결측 복원 2행 적용)"))
        self.assertEqual(table["rows"][0]["시작 시각"], "2026-10-19 00:00:00 (현지 시각, Asia/Seoul)")

        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertEqual(task["consents"]["D"]["content"]["source_contract_id"], derived.source_id)
        facts = task["report"]["facts"]
        titles = [item["title"] for item in facts["decision_tables"]]
        self.assertIn("시간 분할 입력과 규칙", titles)
        rules = {row["항목"]: row for row in next(item for item in facts["decision_tables"] if item["title"] == "시간 분할 입력과 규칙")["rows"]}
        self.assertEqual(rules["시간 형식"]["실행 값 대조"], "실행 값과 같음")
        self.assertEqual(rules["원래 작업용 표 ID"]["실행 값 대조"], "실행 값과 같음")
        self.assertTrue(any("복원으로 채운 2개 시간 셀" in text for text in facts["cautions"]))
        self.assertIn("2026-10-23 00:00:00 (현지 시각, Asia/Seoul)", task["report"]["text"])
        self.assertEqual(facts["technical"]["source_contract_id"], derived.source_id)
        # 원래 생성 표는 바뀌지 않았다. 보고서 재조립(완료 기록의 현재 코드 재검증)도 같은 출처로 성공한다.
        self.assertEqual({path.name: path.read_bytes() for path in Path(original.bundle_directory).iterdir()}, parent_files)
        adapter = task_adapters.adapter("tabular_v2")
        self.assertEqual(adapter.report_facts(self.stored(task))["decision_tables"][-1]["rows"], facts["decision_tables"][-1]["rows"])
        # 완료 기록 복구 정상 경로(파생 표), 그 뒤 파생 묶음을 바꾸면 복구·보고서 재조립을 거부한다.
        self.interrupt(task)
        self.assertEqual(orchestrator.recover_completion(task["task_id"], root=self.store)["phase"], "completed")
        data = Path(derived.bundle_directory) / "data.jsonl"
        data.write_bytes(data.read_bytes().replace(b"2026-10-20", b"2026-10-29"))
        with self.assertRaises(ValueError):
            adapter.report_facts(self.stored(task))
        self.interrupt(task)
        with self.assertRaises(orchestrator.RecoveryError):
            orchestrator.recover_completion(task["task_id"], root=self.store)
        self.assertEqual(self.stored(task)["phase"], "applying")

    def test_not_ready_states_and_plan_change_after_apply_block_confirmation(self):
        plan = self.time_plan()
        task = self.diagnosed()
        error = self.assert_not_confirmed(task, plan, "TIME_SPLIT_NOT_READY")
        self.assertIn("복원 미리보기를 실행하세요", error.message)
        task = self.act(task, "time_preview", plan)
        error = self.assert_not_confirmed(task, plan, "TIME_SPLIT_NOT_READY")
        self.assertIn("결정 R", error.message)
        stopped = self.time_plan(frequency="weekdays")
        task = self.act(task, "time_preview", stopped)
        self.assertEqual(task["time_restore"]["status"], "stopped")
        self.assert_not_confirmed(task, stopped, "TIME_SPLIT_NOT_READY")
        # 적용 뒤 계획(비율)이 바뀌면 다시 막는다.
        task = self.act(task, "time_preview", plan)
        view = self.preview(task, plan)["plan_actions"]
        task = self.act(task, "time_apply", plan, expected_preview=view["expected"])
        changed = self.time_plan()
        changed["split"]["random_state"] += 1
        self.assert_not_confirmed(task, changed, "TIME_SPLIT_NOT_READY")
        self.assertEqual(self.calls.count("propose_split"), 0)

    def test_forged_derived_table_or_swapped_source_is_refused_without_side_effects(self):
        plan = self.time_plan()
        task = self.restored(plan)
        derived = time_flow.restored_source(self.stored(task))
        data = Path(derived.bundle_directory) / "data.jsonl"
        original = data.read_bytes()
        data.write_bytes(original.replace(b"2026-10-20", b"2026-10-29"))
        error = self.assert_not_confirmed(task, plan, "TIME_SOURCE_UNVERIFIED")
        self.assertNotIn("2026-10-29", error.message)
        data.write_bytes(original)

        task = self.confirm(task, plan)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        # 결정 C 동의 내용의 실행 출처·time_source가 현재 고정 출처와 다르면 위임 승인을 발급하지 않는다.
        adapter = task_adapters.adapter("tabular_v2")
        record = self.stored(task)
        for key, value in (("time_source", dict(record["confirmed"]["time_source"], restored_count=3)),
                           ("source_contract_id", time_flow._source(record).source_id)):
            forged = json.loads(json.dumps(record))
            content = dict(forged["consents"]["C"]["content"], **{key: value})
            forged["consents"]["C"] = dict(forged["consents"]["C"], content=content, content_sha256=task_consent._digest(content))
            with self.subTest(key=key), self.assertRaises(ValueError):
                task_consent.authorizer(forged, "C", "tabular_split_review", "propose_split",
                                        expected=adapter.expected(record, "C"))
        # 고정 time_source의 적용 실행 이름을 다른 실행으로 바꾼 기록은 처리 단계 출처 확인에서 멈춘다.
        forged = json.loads(json.dumps(record))
        forged["confirmed"]["time_source"]["apply_run"] = "t" + record["task_id"][:10] + "-ta9"
        with self.assertRaises(Exception) as caught:
            adapter.check_source(forged)
        self.assertEqual(getattr(caught.exception, "code", None), "SNAPSHOT_CHANGED")
        # 고정 출처를 원래 생성 표로 바꾼 기록은 결정 D를 발급하지 않고 멈춘다.
        record["confirmed"]["source"] = time_flow._source(record).to_dict()
        task_store.save_task(record, expected_revision=record["revision"], root=self.store)
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "needs_attention")
        self.assertNotIn("D", task["consents"])
        self.assertEqual(self.calls.count("apply_preparation"), 0)


class ZeroRowTimeSplitTest(TimeSplitFixture):
    csv_text = FULL_CSV

    def test_zero_restoration_uses_the_generated_table_without_decision_r(self):
        plan = self.time_plan()
        task = self.diagnosed()
        task = self.act(task, "time_preview", plan)
        self.assertEqual((task["time_restore"]["status"], task["time_restore"]["preview"]["restored_count"]), ("previewed", 0))
        shown = self.preview(task, plan)
        self.assertIsNone(shown["blocked"])
        self.assertEqual(shown["plan_actions"]["action"]["label"], "다시 미리보기 (새 시도)")
        self.assertTrue(any("기존 작업용 표(복원 대상 0행)" in text for _, text in shown["plan_actions"]["lines"]))
        task = self.confirm(task, plan)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        original = time_flow._source(task)
        self.assertEqual((task["confirmed"]["source_kind"], task["confirmed"]["time_source"]["kind"]), ("generated", "generated"))
        self.assertEqual(task["consents"]["C"]["content"]["source_contract_id"], original.source_id)
        self.assertNotIn("R", task["consents"])
        self.assertEqual(task["time_restore"]["status"], "previewed")
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        facts = task["report"]["facts"]
        self.assertTrue(any("기존 작업용 표(복원 대상 0행" in text for text in facts["lines"]))
        self.assertFalse(any("복원으로 채운" in text for text in facts["cautions"]))
        # 완료 기록 복구: 0행이면 보고서 재조립이 커밋된 미리보기를 현재 코드로 다시 열어 검증한다.
        self.interrupt(task)
        self.assertEqual(orchestrator.recover_completion(task["task_id"], root=self.store)["phase"], "completed")
        self.interrupt(task)
        with unittest.mock.patch("src.agents.tabular_time_service.code_id", return_value="1" * 64), \
                self.assertRaises(orchestrator.RecoveryError) as caught:
            orchestrator.recover_completion(task["task_id"], root=self.store)
        self.assertEqual(caught.exception.code, "RECOVERY_RESULT_UNVERIFIED")
        self.assertEqual(self.stored(task)["phase"], "applying")

    def test_failed_confirmation_is_recovered_by_a_new_preview(self):
        plan = self.time_plan()
        task = self.act(self.diagnosed(), "time_preview", plan)
        with unittest.mock.patch("src.agents.tabular_time_service.code_id", return_value="1" * 64), \
                unittest.mock.patch("src.agents.tabular_time_flow.code_id", return_value="1" * 64):
            self.assert_not_confirmed(task, plan, "TIME_SOURCE_UNVERIFIED")
            view = self.preview(task, plan)["plan_actions"]
            self.assertEqual(view["action"]["id"], "time_preview")
            task = self.act(task, "time_preview", plan)
            entry = task["time_restore"]
            self.assertEqual((entry["status"], entry["attempt"], entry["history"][0]["status"]), ("previewed", 2, "previewed"))
            self.assertIsNone(self.preview(task, plan)["blocked"])
            # 저장소 권한 이상은 데이터 손상과 구분한 저장소 안내로 거부한다.
            from src.agents.protected_store import StoreUnavailable
            with unittest.mock.patch("src.agents.tabular_time_flow.ready_source",
                                     side_effect=StoreUnavailable("TIME_STORE_PERMISSION_INVALID")):
                error = self.assert_not_confirmed(task, plan, "TIME_STORE_PERMISSION_INVALID")
            self.assertIn("권한", error.message)
            task = self.confirm(task, plan)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(task["confirmed"]["time_source"]["preview_run"], entry["preview_run"])

    def test_forged_zero_row_preview_is_refused(self):
        plan = self.time_plan()
        task = self.act(self.diagnosed(), "time_preview", plan)
        record = self.stored(task)
        run = Path(record["snapshot"]["directory"]).parent / "time_runs/preview" / record["time_restore"]["preview_run"]
        preview = next(run.glob("attempts/*/preview.json"))
        preview.write_bytes(preview.read_bytes().replace(b'"row_count": 6', b'"row_count": 7'))
        self.assert_not_confirmed(task, plan, "TIME_SOURCE_UNVERIFIED")

    def test_time_split_failure_is_not_replaced_by_random(self):
        plan = self.time_plan()
        plan["split"]["require_all_classes"] = True
        task = self.diagnosed()
        task = self.act(task, "time_preview", plan)
        # 시간 순서 1/3씩 나누면 Train(True·False)·Validation(True·False)·Test(True·False)이 된다. 비율을 바꿔 Test에 한 클래스만 남긴다.
        plan["split"].update(train_ratio=0.5, validation_ratio=0.3333333333333333, test_ratio=0.16666666666666666)
        task = self.act(task, "time_preview", plan)
        task = self.confirm(task, plan)
        self.assertEqual(task["phase"], "needs_attention", task.get("problem"))
        problem = task["problem"]
        self.assertIn(problem["code"], ("SPLIT_CLASS_MISSING", "SPLIT_REVIEW"))
        self.assertIn("다른 분할 방식으로 자동 변경하지 않습니다", problem["message"])
        self.assertEqual(task["confirmed"]["plan"]["split"]["strategy"], "time")
        self.assertEqual(set(task["steps"]), {"split"})
        self.assertEqual(self.calls.count("propose_split"), 1)
        self.assertEqual(self.calls.count("ingest"), 0)


if __name__ == "__main__":
    unittest.main()

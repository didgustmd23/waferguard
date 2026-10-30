"""표형 v2 작업에서 시간 결측 복원 미리보기(결정 T)·적용(결정 R)을 공통 조정기 계획 동작으로 진행한다.

합성 CSV fixture(test_tabular_task_v2)의 실제 조정기·어댑터·입력/시간 서비스 코드를 쓴다. OS 운송·공유 런타임 ID·모델 응답만 모의한다.
"""

import hashlib
import json
from pathlib import Path
import unittest

from src.agents import task_orchestrator as orchestrator, task_store
from src.agents import tabular_time_flow as time_flow
from src.agents.task_plan import PlanError, signature
from src.preprocessing.tabular_bundle import load_bundle
from tests.agents.test_tabular_task_v2 import ROLES, TabularTaskV2Fixture
from tests.agents.test_tabular_time import ORIGINAL, RUNTIME_ID, time_transport


TIME_CSV = ("x,flag,label,g,id,t\n1.0,1,True,a,0012,2026-10-19\n,0,False,a,0034,\n9.0,1,True,a,0056,2026-10-21\n"
            "100.0,0,False,b,0078,2026-10-22\n,1,True,b,0090,\n700.0,0,False,c,0102,2026-10-24\n")
TIME_ROLES = dict(ROLES, g="제외", t="timestamp")
TIME = {"column": "t", "timestamp_format": "%Y-%m-%d", "timezone": "Asia/Seoul", "frequency": "daily", "interval_minutes": None,
        "row_order_confirmed": True}


class TimeRestoreFlowFixture(TabularTaskV2Fixture):
    csv_text = TIME_CSV

    def setUp(self):
        super().setUp()
        self.start_patch("src.agents.tabular_time_service.run_time_worker", side_effect=time_transport)
        self.start_patch("src.agents.tabular_time_flow.current_runtime_id", return_value=RUNTIME_ID)
        self.start_patch("src.agents.tabular_time_service._current_actor", return_value="synthetic_actor")

    def time_plan(self, **time):
        plan = self.plan(TIME_ROLES)
        plan["split"]["strategy"] = "time"
        plan["time"] = dict(TIME, **time)
        return plan

    def stored(self, task):
        return task_store.load_task(task["task_id"], root=self.store)

    def preview(self, task, plan):
        return orchestrator.preview_plan(self.stored(task), plan, {})

    def act(self, task, action, plan, expected_preview=None, digest=None):
        shown = self.preview(task, plan)
        return orchestrator.run_plan_action(task["task_id"], action, plan, {}, root=self.store,
                                            expected_digest=digest or shown["digest"], expected_preview=expected_preview)


class TimeRestoreFlowTest(TimeRestoreFlowFixture):
    def test_preview_then_apply_creates_derived_table_bound_to_t_and_r(self):
        task = self.diagnosed()
        before_csv = self.source.read_bytes()
        source = time_flow._source(task)
        parent_files = {path.name: path.read_bytes() for path in Path(source.bundle_directory).iterdir()}
        plan = self.time_plan()
        shown = self.preview(task, plan)
        self.assertEqual(shown["blocked"]["code"], "TIME_SPLIT_NOT_READY")
        view = shown["plan_actions"]
        self.assertEqual((view["status"], view["action"]["id"], view["expected"]), ("none", "time_preview", None))
        self.assertTrue(any("결정 T" in line for line in view["consent"]))
        # 미리보기 전에는 작업용 표가 생기지 않는다.
        self.assertFalse((Path(source.snapshot_directory).parent / "time_runs").exists())

        task = self.act(task, "time_preview", plan)
        self.assertEqual(task["phase"], "awaiting_review")
        entry = task["time_restore"]
        self.assertEqual(entry["status"], "previewed", entry)
        self.assertEqual((entry["preview"]["rows"], entry["preview"]["restored_count"]), ([1, 4], 2))
        self.assertEqual(entry["preview"]["times"], ["2026-10-20T00:00:00+09:00", "2026-10-23T00:00:00+09:00"])
        content = task["consents"]["T"]["content"]
        self.assertEqual((content["plan_digest"], content["time_signature"], content["source_contract_id"], content["allowed"]),
                         (signature(plan), signature(TIME), source.source_id, [["tabular_time_preview", "ingest"]]))
        self.assertEqual(entry["delegations"]["preview"]["consent_id"], task["consents"]["T"]["consent_id"])
        self.assertFalse((Path(source.snapshot_directory).parent / "time_runs/apply").exists())

        shown = self.preview(task, plan)
        view = shown["plan_actions"]
        self.assertEqual((view["status"], view["action"]["id"]), ("previewed", "time_apply"))
        self.assertEqual([row["표에 기록할 값"] for row in view["table"]["rows"]], ["2026-10-20", "2026-10-23"])
        self.assertEqual([row["데이터 행(1부터)"] for row in view["table"]["rows"]], [2, 5])
        task = self.act(task, "time_apply", plan, expected_preview=view["expected"])
        entry = task["time_restore"]
        self.assertEqual(entry["status"], "applied", entry)
        content = task["consents"]["R"]["content"]
        self.assertEqual((content["preview_sha256"], content["restored_count"], content["allowed"]),
                         (entry["preview"]["preview_sha256"], 2, [["tabular_time_apply", "ingest"]]))
        self.assertEqual(set(task["consents"]), {"A", "T", "R"})
        self.assertEqual((task["phase"], task["steps"], task["attempt"]), ("awaiting_review", {}, 0))
        derived = time_flow.restored_source(self.stored(task))
        self.assertEqual(derived.source_id, entry["source_id"])
        frame = load_bundle(derived.bundle_directory, expected=derived.bundle)
        self.assertEqual(frame["t"].tolist()[1::3], ["2026-10-20", "2026-10-23"])
        # 원본 CSV와 기존 생성 표는 그대로다.
        self.assertEqual(self.source.read_bytes(), before_csv)
        self.assertEqual({path.name: path.read_bytes() for path in Path(source.bundle_directory).iterdir()}, parent_files)
        shown = self.preview(task, plan)
        # 적용 완료 뒤에도 새 미리보기(새 시도)는 고를 수 있다. 적용(R)은 다시 고를 수 없다.
        self.assertEqual((shown["plan_actions"]["status"], shown["plan_actions"]["action"]["id"]), ("applied", "time_preview"))
        self.assertEqual(shown["plan_actions"]["action"]["label"], "다시 미리보기 (새 시도)")
        # 3단계부터 적용 완료면 time 분할 확정을 연다(확정·분할은 test_time_split_flow). 화면은 파생 표를 입력으로 안내한다.
        self.assertIsNone(shown["blocked"])
        self.assertTrue(any("확정 시 입력: 파생 표(복원 2행)" in text for _, text in shown["plan_actions"]["lines"]))
        self.assertNotIn("C", self.stored(task)["consents"])
        # 적용을 다시 요청해도 소비된 확인을 재사용하지 않는다.
        with self.assertRaises(ValueError):
            self.act(task, "time_apply", plan, expected_preview=entry["preview"]["preview_sha256"])
        # 작업 기록의 시간 복원 항목에는 원본 시간값이 없다(복원 시각·행 번호·지문만).
        recorded = json.dumps(self.stored(task)["time_restore"], ensure_ascii=False)
        self.assertFalse([value for value in ORIGINAL if value in recorded])

    def test_apply_without_preview_or_with_other_preview_or_rules_is_rejected(self):
        task = self.diagnosed()
        plan = self.time_plan()
        with self.assertRaises(ValueError):
            self.act(task, "time_apply", plan, expected_preview="0" * 64)
        self.assertEqual(set(self.stored(task)["consents"]), {"A"})
        # 미리보기 결합값이 오래되면(revision 변경) 동의를 만들지 않는다.
        stale = self.preview(task, plan)["digest"]
        record = self.stored(task)
        task_store.save_task(record, expected_revision=record["revision"], root=self.store)
        with self.assertRaises(PlanError) as caught:
            self.act(task, "time_preview", plan, digest=stale)
        self.assertEqual(caught.exception.code, "PLAN_PREVIEW_CHANGED")
        self.assertEqual(set(self.stored(task)["consents"]), {"A"})

        task = self.act(task, "time_preview", plan)
        sha = task["time_restore"]["preview"]["preview_sha256"]
        with self.assertRaises(ValueError):
            self.act(task, "time_apply", plan, expected_preview="0" * 64)
        # 미리보기 뒤 규칙이 바뀌면 이전 미리보기로 적용하지 않고 새 미리보기를 요구한다.
        changed = self.time_plan(timezone="UTC")
        view = self.preview(task, changed)["plan_actions"]
        self.assertEqual((view["status"], view["action"]["id"]), ("stale", "time_preview"))
        with self.assertRaises(ValueError):
            self.act(task, "time_apply", changed, expected_preview=sha)
        # 규칙과 무관한 계획 항목이 바뀌어도 계획 지문이 달라 이전 미리보기를 쓰지 않는다.
        other = self.time_plan()
        other["split"]["random_state"] = other["split"]["random_state"] + 1
        self.assertEqual(self.preview(task, other)["plan_actions"]["status"], "stale")
        self.assertNotIn("R", self.stored(task)["consents"])
        self.assertFalse(any((Path(self.stored(task)["snapshot"]["directory"]).parent / "time_runs/apply").glob("*")))

        # 새 규칙으로 다시 미리보기하면 이전 기록은 history로 보존한다.
        task = self.act(task, "time_preview", changed)
        entry = task["time_restore"]
        self.assertEqual((entry["attempt"], entry["preview_run"], len(entry["history"])), (2, entry["preview_run"], 1))
        self.assertEqual(entry["history"][0]["status"], "previewed")
        self.assertNotEqual(entry["preview"]["preview_sha256"], sha)

    def test_stopped_preview_shows_cause_and_next_action_and_cannot_be_applied(self):
        task = self.diagnosed()
        cases = {"TIME_FREQUENCY_MISMATCH": {"frequency": "weekdays"}, "TIME_PARSE_FAILED": {"timestamp_format": "%d.%m.%Y"},
                 "TIME_AMBIGUOUS": {"timestamp_format": "%Y-%m-%d %H", "timezone": "America/New_York"}}
        for code, changes in cases.items():
            with self.subTest(code=code):
                plan = self.time_plan(**changes)
                task = self.act(task, "time_preview", plan)
                entry = task["time_restore"]
                if code == "TIME_AMBIGUOUS":
                    # 날짜만 있는 값은 "%Y-%m-%d %H" 형식으로 읽히지 않는다(형식 불일치가 먼저 멈춘다).
                    code = "TIME_PARSE_FAILED"
                self.assertEqual((entry["status"], entry["code"]), ("stopped", code))
                view = self.preview(task, plan)["plan_actions"]
                self.assertIsNone(view["action"])
                texts = [text for _, text in view["lines"]]
                self.assertTrue(any(text.startswith("다음 행동: ") for text in texts))
                self.assertFalse([value for value in ORIGINAL if value in json.dumps(view, ensure_ascii=False)])
                with self.assertRaises(ValueError):
                    self.act(task, "time_apply", plan, expected_preview=None)

    def test_runtime_failure_is_recorded_without_retry_and_a_new_preview_is_possible(self):
        from src.agents.worker_runtime import RuntimeUnavailable

        task = self.diagnosed()
        plan = self.time_plan()
        with unittest.mock.patch("src.agents.tabular_time_flow.current_runtime_id",
                                 side_effect=RuntimeUnavailable("RUNTIME_SOURCE_INVALID")):
            task = self.act(task, "time_preview", plan)
        entry = task["time_restore"]
        self.assertEqual((entry["status"], entry["code"], task["phase"]), ("failed", "RUNTIME_SOURCE_INVALID", "awaiting_review"))
        view = self.preview(task, plan)["plan_actions"]
        self.assertEqual(view["action"]["id"], "time_preview")
        task = self.act(task, "time_preview", plan)
        self.assertEqual(task["time_restore"]["status"], "previewed")


if __name__ == "__main__":
    unittest.main()

"""v2 표형 작업의 time 분할 시간 설정(계획 time 구획). 설정·검증·동의 결합만 하고 실행은 결정 C 발급 전에 멈추는지 확인한다.

합성 CSV fixture(test_tabular_task_v2)에서 실제 조정기·어댑터·권장안 병합 코드를 쓴다. OS transport와 모델 응답만 모의한다.
"""

import json
import unittest

from src.agents import task_adapters, task_orchestrator as orchestrator, task_store
from src.agents.task_plan import PlanError, flatten
from src.agents.tabular_v2_task import TIME_DEFAULT
from tests.agents.test_tabular_task_v2 import ROLES, TabularTaskV2Fixture
from tests.agents.test_task_orchestrator import model_reply


TIME_ROLES = dict(ROLES, g="제외", id="timestamp")
TIME = {"column": "id", "timestamp_format": "%Y-%m-%d", "timezone": "Asia/Seoul", "frequency": "daily", "interval_minutes": None,
        "row_order_confirmed": True}


class TimeSplitPlanTest(TabularTaskV2Fixture):
    def time_plan(self, **time):
        plan = self.plan(TIME_ROLES)
        plan["split"]["strategy"] = "time"
        plan["time"] = dict(TIME, **time)
        return plan

    def stored(self, task):
        return task_store.load_task(task["task_id"], root=self.store)

    def assert_rejected(self, task, plan, code):
        with self.assertRaises(PlanError) as caught:
            orchestrator.preview_plan(self.stored(task), plan, {})

        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_fields_are_user_only_and_time_is_a_choice_with_empty_default(self):
        task = self.diagnosed()
        adapter = task_adapters.adapter("tabular_v2")
        fields = {field["path"]: field for field in adapter.plan_fields(task)}
        self.assertTrue(all(fields["time." + name]["user_only"] for name in TIME_DEFAULT))
        self.assertNotIn("unsupported", fields["split.strategy"])
        self.assertIn("time", fields["split.strategy"]["options"])
        self.assertEqual(adapter.base_plan(task)["time"], TIME_DEFAULT)
        self.assertEqual(orchestrator.review_state(task)["plan"]["time"], TIME_DEFAULT)
        # 표형 v1(기존 작업)은 time 분할을 계속 지원하지 않는다.
        v1 = {field["path"]: field for field in task_adapters.adapter("tabular_csv").plan_fields(task)}
        self.assertEqual(v1["split.strategy"]["unsupported"], ["time"])
        self.assertFalse([path for path in v1 if path.startswith("time.")])

    def test_valid_settings_pass_validation_but_stop_before_attempt_and_consent(self):
        task = self.diagnosed()
        plan = self.time_plan()
        preview = orchestrator.preview_plan(self.stored(task), plan, {})
        self.assertEqual(preview["plan"]["time"], TIME)
        self.assertEqual(preview["blocked"]["code"], "TIME_SPLIT_NOT_READY")
        before = self.stored(task)

        with self.assertRaises(PlanError) as caught:
            orchestrator.confirm_plan(task["task_id"], plan, root=self.store, expected_digest=preview["digest"])

        self.assertEqual(caught.exception.code, "TIME_SPLIT_NOT_READY")
        after = self.stored(task)
        self.assertEqual((after["revision"], after["attempt"], set(after["consents"]), after["steps"], after["phase"]),
                         (before["revision"], before["attempt"], {"A"}, {}, "awaiting_review"))
        self.assertEqual(self.calls.count("apply_preparation"), 0)

    def test_confirmed_time_format_fills_split_and_quality_rules_without_new_split_fields(self):
        from src.agents import llm_task_assist
        from src.agents.profile_store import SPLIT_FIELDS
        task = self.diagnosed()
        adapter = task_adapters.adapter("tabular_v2")
        profile, _ = adapter.validate_plan(self.stored(task), self.time_plan())
        self.assertEqual((profile.split_rules.timestamp_format, profile.quality_rules.timestamp_format), ("%Y-%m-%d", "%Y-%m-%d"))
        profile, _ = adapter.validate_plan(self.stored(task), dict(self.plan(), time=dict(TIME_DEFAULT)))
        self.assertIsNone(profile.split_rules.timestamp_format)
        # 분할 설정 항목·v1 time 안내는 그대로다.
        self.assertNotIn("timestamp_format", SPLIT_FIELDS)
        self.assertFalse(llm_task_assist.TIME_SPLIT_AVAILABLE)

    def test_missing_conflicting_or_invalid_settings_are_rejected_before_execution(self):
        task = self.diagnosed()
        cases = [
            ({"column": None}, "TIME_SETTINGS_MISSING"),
            ({"frequency": None}, "TIME_SETTINGS_MISSING"),
            ({"row_order_confirmed": False}, "TIME_SETTINGS_MISSING"),
            ({"column": "x"}, "TIME_COLUMN_ROLE_MISMATCH"),
            ({"timestamp_format": None}, "TIME_FORMAT_REQUIRED"),
            ({"timezone": None}, "TIME_TIMEZONE_REQUIRED"),
            ({"timestamp_format": "%Y-%m-%d %H:%M%z"}, "TIME_TIMEZONE_CONFLICT"),
            ({"timezone": "Mars/Base"}, "TIME_TIMEZONE_INVALID"),
            ({"timestamp_format": "%A %d"}, "TIME_FORMAT_INVALID"),
            ({"frequency": "fixed"}, "TIME_FREQUENCY_INVALID"),
            ({"interval_minutes": 30}, "TIME_FREQUENCY_INVALID"),
        ]

        for changes, code in cases:
            with self.subTest(changes=changes):
                self.assert_rejected(task, self.time_plan(**changes), code)

        # 형식 검사(허용 값)를 통과하지 못한 값과 구획 구성 오류
        self.assert_rejected(task, self.time_plan(row_order_confirmed="yes"), "TIME_SETTINGS_MISSING")
        plan = self.time_plan()
        plan["time"]["extra"] = 1
        self.assert_rejected(task, plan, "TIME_SETTINGS_INVALID")
        # time 분할이 아닌데 시간 설정이 남아 있으면 확정하지 않는다(규칙이 다른 계획을 조용히 받지 않음).
        plan = self.plan()
        plan["time"] = dict(TIME)
        self.assert_rejected(task, plan, "TIME_SETTINGS_UNUSED")
        # 기본값과 같다고 비교되지만 타입이 다른 값(0·0.0)도 기본값으로 받지 않는다.
        for changes in ({"row_order_confirmed": 0}, {"row_order_confirmed": 0.0}):
            with self.subTest(changes=changes):
                plan = self.plan()
                plan["time"] = dict(TIME_DEFAULT, **changes)
                self.assert_rejected(task, plan, "TIME_SETTINGS_UNUSED")
        # time 분할에 timestamp 역할이 없으면 기존 역할×분할 교차 검사가 필수 질문(분할 방식 불일치)으로 먼저 멈춘다.
        plan = self.time_plan()
        plan["roles"]["id"] = "식별자"
        error = self.assert_rejected(task, plan, "REQUIRED_QUESTION_UNANSWERED")
        self.assertIn("split.strategy.mismatch", error.subjects)

    def test_rule_change_after_the_agreed_preview_is_rejected(self):
        task = self.diagnosed()
        preview = orchestrator.preview_plan(self.stored(task), self.time_plan(), {})

        for changes in ({"timezone": "UTC"}, {"frequency": "weekdays"}, {"timestamp_format": "%d.%m.%Y"}):
            with self.subTest(changes=changes), self.assertRaises(PlanError) as caught:
                orchestrator.confirm_plan(task["task_id"], self.time_plan(**changes), root=self.store,
                                          expected_digest=preview["digest"])

            self.assertEqual(caught.exception.code, "PLAN_PREVIEW_CHANGED")

        self.assertEqual(set(self.stored(task)["consents"]), {"A"})

    def test_consent_binds_the_time_settings_of_the_confirmed_plan(self):
        task = self.diagnosed()
        task = orchestrator.confirm_plan(task["task_id"], self.plan(), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        content = task["consents"]["C"]["content"]
        self.assertEqual(content["time"], TIME_DEFAULT)
        self.assertEqual(len(content["time_signature"]), 64)
        adapter = task_adapters.adapter("tabular_v2")
        changed = json.loads(json.dumps(task))
        changed["confirmed"]["plan"]["time"] = dict(TIME)
        self.assertNotEqual(adapter.consent_c_content(changed)["time_signature"], content["time_signature"])

    def test_model_cannot_see_or_set_time_settings(self):
        def requester(payload, config):
            self.requests.append(payload)
            return model_reply(ROLES, changes=[{"path": "time.column", "value": "id", "reason": "시간 컬럼"}])

        task = self.diagnosed(requester=requester)
        sent = json.dumps(self.requests, ensure_ascii=False)
        self.assertNotIn("row_order_confirmed", sent)
        self.assertNotIn("timestamp_format", sent)
        self.assertNotIn("time.column", json.loads(self.requests[0]["messages"][-1]["content"])["allowed_values"])
        self.assertEqual(task["draft"]["plan"]["time"], TIME_DEFAULT)
        self.assertFalse([item for item in task["draft"]["changes"] if item["path"].startswith("time.")])

    def test_model_time_strategy_is_cleared_for_the_user_to_choose(self):
        def requester(payload, config):
            self.requests.append(payload)
            return model_reply(TIME_ROLES, changes=[{"path": "split.strategy", "value": "time", "reason": "시간 순서"}])

        task = self.diagnosed(requester=requester)
        self.assertIsNone(task["draft"]["plan"]["split"]["strategy"])
        self.assertEqual(task["draft"]["plan"]["time"], TIME_DEFAULT)
        self.assertTrue(any(note.startswith("모델이 time 분할을 제안했습니다.") for note in task["draft"]["notes"]))

    def test_record_from_before_the_time_section_is_read_with_default_settings(self):
        task = self.diagnosed()
        record = self.stored(task)
        record["draft"]["plan"].pop("time")
        task_store.save_task(record, expected_revision=record["revision"], root=self.store)
        state = orchestrator.review_state(self.stored(task))
        self.assertEqual(state["plan"]["time"], TIME_DEFAULT)
        self.assertNotIn("time", self.stored(task)["draft"]["plan"])
        # 이전 형식(time 구획 없음)으로 제출한 계획도 기본 시간 설정으로 검증·미리보기·진행한다. 표시 출처도 모든 항목에 있다.
        preview = orchestrator.preview_plan(self.stored(task), self.plan(), {})
        self.assertEqual(set(preview["origins"]), {path for path, _ in flatten(preview["plan"])})
        self.assertEqual(preview["origins"]["time.column"], {"source": "base"})
        task = orchestrator.confirm_plan(task["task_id"], self.plan(), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))


if __name__ == "__main__":
    unittest.main()

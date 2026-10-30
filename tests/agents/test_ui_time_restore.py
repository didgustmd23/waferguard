"""최종 계획 미리보기 화면의 시간 결측 복원 절을 실제 Streamlit 화면으로 진행한다(미리보기 → 표 → 적용 → 규칙 변경 시 다시 미리보기).

작업은 합성 CSV fixture로 진단까지 만든 뒤 화면에서 연다. OS 운송·공유 런타임 ID·모델 응답만 모의한다.
"""

from pathlib import Path
import unittest

from src.agents import task_store
from tests.agents.test_time_restore_flow import TimeRestoreFlowFixture
from tests.agents.test_time_split_flow import FULL_CSV


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TimeRestoreScreenTest(TimeRestoreFlowFixture):
    def setUp(self):
        super().setUp()
        self.start_patch("src.agents.task_store.STORE_ROOT", new=self.store)
        self.start_patch("src.ui.runtime_notice.server_permissions", return_value={"elevated": True, "pid": 1, "python": "synthetic"})

    def button(self, app, label):
        return next(item for item in app.button if item.label == label)

    def assert_no_fallback_notice(self, app):
        """정상 완료(빈 결과)를 결과 불명확으로 보고 저장 상태 대조 안내를 내지 않는다."""
        from src.ui import task_submit

        texts = [item.value for kind in (app.info, app.error, app.warning) for item in kind]
        self.assertFalse([text for text in texts if "결과를 화면에서 직접 받지 못했" in text or text == task_submit.UNCLEAR[1]])

    def test_preview_table_apply_and_rule_change_on_the_final_plan_screen(self):
        from streamlit.testing.v1 import AppTest

        task = self.diagnosed()
        key = "task_review_" + task["task_id"][:8] + "0"
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180)
        app.session_state["ui_task_id"] = task["task_id"]
        # 계획 양식 제출 대신 미리 볼 계획을 세션에 둔다(양식 제출과 같은 키). 화면 표시만으로는 실행하지 않는다.
        app.session_state[key + "_draft"] = {"plan": self.time_plan(), "answers": {}}
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value == "#### 시간 결측 복원" for item in app.markdown))
        self.assertTrue(any("복원 미리보기를 실행하세요" in item.value for item in app.warning))
        self.assertFalse([item for item in app.button if item.label == "확인하고 진행"])
        self.assertNotIn("time_restore", task_store.load_task(task["task_id"], root=self.store))

        # 동의 없이 누르면 실행하지 않는다.
        self.button(app, "복원 미리보기 실행").click().run()
        self.assertTrue(any(item.value == "복원 미리보기 범위를 확인하고 체크해 주세요." for item in app.warning))
        self.assertNotIn("time_restore", task_store.load_task(task["task_id"], root=self.store))

        next(item for item in app.checkbox if item.label.endswith("(결정 T)")).check()
        self.button(app, "복원 미리보기 실행").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["time_restore"]["status"], "previewed")
        self.assertTrue(any(item.value.startswith("복원 대상: 전체 6행 중 2행") for item in app.text))
        self.assert_no_fallback_notice(app)
        tables = [item.value for item in app.dataframe if "표에 기록할 값" in item.value.columns]
        self.assertEqual(tables[0]["표에 기록할 값"].tolist(), ["2026-10-20", "2026-10-23"])

        next(item for item in app.checkbox if item.label.endswith("(결정 R)")).check()
        self.button(app, "복원 적용").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        stored = task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["time_restore"]["status"], stored["phase"]), ("applied", "awaiting_review"))
        self.assertTrue(any(item.value.startswith("복원을 적용했습니다: 2행") for item in app.success))
        self.assert_no_fallback_notice(app)
        self.assertFalse([item for item in app.button if item.label in ("복원 적용", "복원 미리보기 실행")])
        # 적용 뒤에는 같은 화면에 입력 출처 안내와 결정 C 동의·확인 버튼이 나온다(확정·분할 흐름은 test_time_split_flow).
        self.assertTrue(any(item.value.startswith("확정 시 입력: 파생 표(복원 2행)") for item in app.text))
        self.assertTrue([item for item in app.button if item.label == "확인하고 진행"])
        self.assertTrue([item for item in app.button if item.label == "다시 미리보기 (새 시도)"])

        # 규칙을 바꾸면 이전 미리보기·적용은 쓰지 않고 다시 미리보기를 요구한다.
        app.session_state[key + "_draft"] = {"plan": self.time_plan(timezone="UTC"), "answers": {}}
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any("이전 미리보기·확인은 쓰지 않습니다" in item.value for item in app.warning))
        self.assertTrue([item for item in app.button if item.label == "복원 미리보기 실행"])
        self.assertFalse([item for item in app.button if item.label == "확인하고 진행"])


class ZeroRowRepreviewScreenTest(TimeRestoreFlowFixture):
    """확정 직전 재확인이 실패하면 안내대로 같은 화면에서 새 미리보기를 실행해 다시 확정할 수 있다."""

    csv_text = FULL_CSV

    def setUp(self):
        super().setUp()
        self.start_patch("src.agents.task_store.STORE_ROOT", new=self.store)
        self.start_patch("src.ui.runtime_notice.server_permissions", return_value={"elevated": True, "pid": 1, "python": "synthetic"})

    button = TimeRestoreScreenTest.button

    def test_failed_confirmation_offers_a_new_preview_on_the_same_screen(self):
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest

        task = self.diagnosed()
        key = "task_review_" + task["task_id"][:8] + "0"
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180)
        app.session_state["ui_task_id"] = task["task_id"]
        app.session_state[key + "_draft"] = {"plan": self.time_plan(), "answers": {}}
        app.run()
        next(item for item in app.checkbox if item.label.endswith("(결정 T)")).check()
        self.button(app, "복원 미리보기 실행").click().run()
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["time_restore"]["preview"]["restored_count"], 0)
        # 코드가 바뀐 뒤(이전 미리보기를 현재 코드로 다시 열 수 없음) 확정하면 거부되고, 같은 화면에 새 미리보기 버튼이 있다.
        with patch("src.agents.tabular_time_service.code_id", return_value="1" * 64),                 patch("src.agents.tabular_time_flow.code_id", return_value="1" * 64):
            app.run()
            next(item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")).check()
            self.button(app, "확인하고 진행").click().run()
            self.assertFalse(app.exception, [item.value for item in app.exception])
            self.assertTrue(any(item.value.startswith("복원 결과 또는 작업용 표를 다시 확인하지 못했습니다") for item in app.error))
            self.assertNotIn("C", task_store.load_task(task["task_id"], root=self.store)["consents"])
            next(item for item in app.checkbox if item.label == "위 규칙으로 새 복원 미리보기 실행에 동의합니다 (결정 T)").check()
            self.button(app, "다시 미리보기 (새 시도)").click().run()
            self.assertFalse(app.exception, [item.value for item in app.exception])
            entry = task_store.load_task(task["task_id"], root=self.store)["time_restore"]
            self.assertEqual((entry["status"], entry["attempt"], len(entry["history"])), ("previewed", 2, 1))
            self.assertTrue([item for item in app.button if item.label == "확인하고 진행"])


if __name__ == "__main__":
    unittest.main()

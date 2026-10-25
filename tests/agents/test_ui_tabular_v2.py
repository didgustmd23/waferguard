"""통합 작업 화면의 v2 표형 흐름을 실제 Streamlit 화면으로 진행한다.

새 작업은 current 세대(CSV→v2)로 만든다. OS transport와 로컬 모델 응답만 모의한다. 화면 표시만으로는 작업·복사·모델 호출이 없어야 한다.
"""

from pathlib import Path
import unittest

from tests.agents import test_task_orchestrator as flow
from tests.agents import test_ui_task as screens


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TabularV2ScreenTest(unittest.TestCase):
    setUp = screens.TaskScreenTest.setUp
    setUp_flow = screens.TaskScreenTest.setUp_flow
    setUp_csv = screens.TaskScreenTest.setUp_csv
    start_patch = screens.TaskScreenTest.start_patch
    transport = screens.TaskScreenTest.transport

    def start(self, source):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.text_input(key="_widget_task_source").set_value(str(source))
        app.text_area(key="task_goal").set_value("label 예측용으로 분할·전처리해 주세요")
        app.checkbox(key="task_consent_a").check()
        next(item for item in app.button if item.label == "작업 시작").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        return app

    def task(self):
        tasks = list(self.store.iterdir())
        self.assertEqual(len(tasks), 1)
        return flow.task_store.load_task(tasks[0].name, root=self.store)

    def test_duplicate_header_choice_consent_stale_resubmission_and_next_action_on_one_screen(self):
        import pandas as pd

        frame = pd.read_csv(self.source)
        frame.insert(1, "zero", frame["x"] * 2)
        frame.insert(2, "zero_", frame["x"] * 3)
        source = self.root / "duplicate.csv"
        frame.to_csv(source, index=False, header=["x", "zero", "zero", "label", "id", "time"])
        self.model_calls.side_effect = lambda payload, config: flow.model_reply(dict(flow.ROLES, zero_1="수치형", zero_2="수치형"))
        app = self.start(source)
        task = self.task()
        self.assertEqual((task["kind"], task["phase"]), ("tabular_v2", "awaiting_input"))
        self.assertTrue(any(item.value == "필요한 결정 — 입력 해석" for item in app.subheader))
        self.assertTrue(any("같은 컬럼명이 여러 번" in item.value for item in app.warning))
        self.assertTrue(any(item.value == "확인된 사유: 영향받는 컬럼 수 2" for item in app.text))
        self.assertTrue(any(item.value.startswith("다음 행동: ") for item in app.text))
        self.assertTrue(any(item.value == "현재: 입력 해석 선택 필요" for item in app.text))
        self.model_calls.assert_not_called()
        self.assertIsNone(task["input_selection"]["materialize_run"])

        # 동의 없이 제출하면 진행하지 않는다.
        header = next(item for item in app.selectbox if item.label == "중복 컬럼명 처리")
        self.assertIsNone(header.value)
        empty = next(item for item in app.selectbox if item.label == "빈 칸 처리")
        self.assertEqual(empty.value, "결측으로 읽기 (기본)")
        next(item for item in app.button if item.label == "고른 해석으로 진행").click().run()
        self.assertTrue(any(item.value == "동의 범위를 확인하고 체크해 주세요." for item in app.warning))
        self.assertEqual(self.task()["revision"], task["revision"])

        # 처리 방법을 고르지 않은(해석이 그대로인) 제출은 거부하고 원인·다음 행동을 같은 화면에 보인다.
        app.checkbox(key="task_input_consent").check()
        next(item for item in app.button if item.label == "고른 해석으로 진행").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value.startswith("제출을 받지 않았습니다.") for item in app.error))
        self.assertTrue(any(item.value.startswith("원인: 화면이 오래됐거나") for item in app.text))
        self.assertEqual((self.task()["revision"], self.model_calls.call_count), (task["revision"], 0))

        # 화면을 그린 뒤 기록이 바뀌면(다른 탭 등) 이 화면에서 본 revision의 제출은 거부한다. 새로 그린 화면에서는 진행한다.
        current = self.task()
        flow.task_store.save_task(current, expected_revision=current["revision"], root=self.store)
        header = next(item for item in app.selectbox if item.label == "중복 컬럼명 처리")
        header.set_value(header.options[0])
        next(item for item in app.button if item.label == "고른 해석으로 진행").click().run()
        self.assertTrue(any(item.value.startswith("제출을 받지 않았습니다.") for item in app.error))
        self.assertEqual(self.model_calls.call_count, 0)
        self.assertEqual(self.task()["phase"], "awaiting_input")

        next(item for item in app.button if item.label == "고른 해석으로 진행").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(self.task()["phase"], "awaiting_review", self.task().get("problem"))
        self.assertEqual(self.model_calls.call_count, 1)
        self.assertTrue(any("계획 확인" in item.value for item in app.subheader))
        self.assertTrue(any("번호를 붙였습니다" in item.value for item in app.text))
        self.assertTrue(any(item.value.startswith("입력 해석: UTF-8") for item in app.text))

    def test_model_refusal_starts_from_default_plan_and_finishes_with_v2_report(self):
        from src.agents.local_ollama import LocalProposalError

        def refuse(payload, config):
            raise LocalProposalError("OLLAMA_CONNECTION_FAILED")

        self.model_calls.side_effect = refuse
        before = self.source.read_bytes()
        app = self.start(self.source)
        task = self.task()
        self.assertEqual((task["kind"], task["phase"], task["draft"]["source"]), ("tabular_v2", "awaiting_review", "default"))
        self.assertTrue(any(item.value.startswith("권장안: 기본 계획 (로컬 LLM 권장안을 사용할 수 없었습니다") for item in app.text))
        # 기존 분할 계약의 클래스 조건을 양식에서 보이고 바꿀 수 있다. 기본은 켜짐(모든 구간에 모든 클래스 요구)이다.
        self.assertTrue(next(item for item in app.checkbox if item.label == "모든 구간에 모든 클래스 요구").value)

        # 기본 계획의 필수 질문(target 등)에 답하고 최종 계획을 확인한다.
        texts = [item.value for item in app.text]
        required = [text.split(".")[0] for text in texts if text.startswith("질문 ") and "[필수]" in text]
        for number in required:
            box = next(item for item in app.selectbox if item.label == number + " 답변")
            wanted = next((option for option in box.options if option.startswith("label")), None)
            box.set_value(wanted or next(option for option in box.options if "(권장)" in option))
        # 기본 계획은 고유값이 많은 문자열 컬럼(id·time)을 범주형으로 둔다. 사용자가 양식에서 새 범주 처리를 고른다.
        unknown = next(item for item in app.selectbox if item.label == "Train에 없던 범주 처리 (unknown_category)")
        unknown.set_value("ignore")
        screens.preview_and_confirm(self, app)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        stored = self.task()
        self.assertEqual(stored["phase"], "awaiting_apply", stored.get("problem"))
        self.assertTrue(any("필요한 결정 — 적용" in item.value for item in app.subheader))

        app.checkbox(key="task_consent_d").check()
        next(item for item in app.button if item.label == "적용하고 마무리").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        stored = self.task()
        self.assertEqual(stored["phase"], "completed", stored.get("problem"))
        self.assertTrue(any(item.value == "최종 결과" for item in app.subheader))
        self.assertTrue(any(item.value.startswith("처리: 완료 · 로컬 LLM 권장안: 사용 못 함") for item in app.text))
        self.assertTrue(any(item.value.startswith("입력 해석: ") for item in app.text))
        self.assertTrue(any("load_handoff_partition" in item.value for item in app.text))
        self.assertEqual(self.calls.count("apply_preparation"), 1)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.model_calls.call_count, 1)


if __name__ == "__main__":
    unittest.main()

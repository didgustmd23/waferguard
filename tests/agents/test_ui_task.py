"""통합 작업 화면(표형): 요청 → 진단 → 로컬 LLM 권장안 → 질문·계획 확인 → 전처리 → 통합 결과를 실제 Streamlit 화면으로 진행한다.

OS transport와 로컬 모델 응답만 모의한다. 화면 표시만으로는 작업·복사·모델 호출이 없어야 한다.
"""

from pathlib import Path
import unittest
from unittest.mock import patch

from tests.agents import test_task_orchestrator as flow
from tests.agents.test_tabular_task_v2 import add_v2_patches


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def preview(app):
    """계획 확인 1단계: 양식·답변을 제출해 답변을 반영한 최종 계획을 미리 본다(실행 없음)."""
    next(item for item in app.button if item.label == "최종 계획 미리보기").click().run()


def preview_and_confirm(test, app):
    """미리 본 최종 계획에 동의하고 확정한다. 미리보기에서 검증 오류가 나면 확정 버튼이 없다."""
    preview(app)
    test.assertFalse(app.exception, [item.value for item in app.exception])
    boxes = [item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")]

    if boxes:
        boxes[0].check()
        next(item for item in app.button if item.label == "확인하고 진행").click().run()


class TaskScreenTest(unittest.TestCase):
    setUp_flow = flow.TaskOrchestratorTest.setUp
    setUp_csv = flow.TaskOrchestratorTest.setUp_csv
    start_patch = flow.TaskOrchestratorTest.start_patch
    transport = flow.TaskOrchestratorTest.transport

    def setUp(self):
        self.setUp_flow()
        self.start_patch("src.agents.task_store.STORE_ROOT", new=self.store)
        self.start_patch("src.ui.runtime_notice.server_permissions",
                         return_value={"elevated": True, "pid": 1, "python": "synthetic"})
        self.model_calls = self.start_patch("src.agents.task_recommend.request_local_proposal",
                                            side_effect=lambda payload, config: flow.model_reply())
        # 화면에서 새로 시작한 CSV 작업은 v2 표형 작업이다. 기록을 직접 만든 테스트는 기존(legacy) 작업을 화면에서 연다.
        add_v2_patches(self, self.store, self.handoff_runs)

    def test_screen_flow_from_request_to_integrated_result(self):
        from streamlit.testing.v1 import AppTest

        before = self.source.read_bytes()
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value == "요청" for item in app.subheader))
        self.assertFalse(self.store.exists() and any(self.store.iterdir()))

        app.text_input(key="_widget_task_source").set_value(str(self.source))
        app.text_area(key="task_goal").set_value("label 예측용으로 분할·전처리해 주세요")
        # 동의 없이 제출하면 작업을 만들지 않는다.
        next(item for item in app.button if item.label == "작업 시작").click().run()
        self.assertFalse(self.store.exists() and any(self.store.iterdir()))
        self.model_calls.assert_not_called()

        app.checkbox(key="task_consent_a").check()
        next(item for item in app.button if item.label == "작업 시작").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(self.model_calls.call_count, 1)
        self.assertTrue(any("계획 확인" in item.value for item in app.subheader))
        self.assertTrue(any("120행" in item.value for item in app.text))
        self.assertFalse(any("120행" in item.value for item in app.markdown))
        # 동의는 최종 계획을 미리 본 뒤에만 나온다. 미리보기만으로는 실행하지 않는다.
        self.assertFalse([item for item in app.button if item.label == "확인하고 진행"])
        stages = list(self.calls)
        preview(app)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertIn("#### 최종 적용 계획 (확정 전 확인)", [item.value for item in app.markdown])
        self.assertTrue(any(item.value.startswith("처리 내용: ") for item in app.text))
        self.assertEqual(self.calls, stages)
        next(item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")).check()
        next(item for item in app.button if item.label == "확인하고 진행").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any("필요한 결정 — 적용" in item.value for item in app.subheader))
        # 적용 대상은 구조화된 표로 보여 주고 원문 보고서는 접어 둔다.
        self.assertIn("#### 특징별 처리", [item.value for item in app.markdown])
        self.assertEqual(self.calls.count("apply_preparation"), 0)

        app.checkbox(key="task_consent_d").check()
        next(item for item in app.button if item.label == "적용하고 마무리").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value == "최종 결과" for item in app.subheader))
        headings = [item.value for item in app.markdown]
        for title in ("#### 핵심 결과와 주요 문제", "#### 컬럼별 처리와 제외 사유", "#### 실제 전후 비교", "#### 그래프",
                      "#### 경고: 정보 손실·미처리·미검증", "#### 참고 사항", "#### 적용 구간과 결과 파일 활용"):
            self.assertIn(title, headings)
        self.assertTrue(app.dataframe)
        # 보고서 그래프 2종(요청·확정 분할 비율, target 분포)은 저장 보고서의 같은 사실로 화면에 그린다.
        texts = [item.value for item in app.text]
        for title in ("■ 요청·확정 분할 비율 (구간별)", "■ target 클래스 분포 (전체 입력, 분할 전)", "■ 분할 EDA 그래프 (검증된 PNG)"):
            self.assertIn(title, texts)
        self.assertEqual(len(app.get("vega_lite_chart")), 2)
        self.assertFalse(any(self.root.name in str(item.value) for item in app.text if "보고서 파일" not in item.value))
        self.assertEqual(self.calls.count("apply_preparation"), 1)
        self.assertEqual(self.model_calls.call_count, 1)
        self.assertEqual(self.source.read_bytes(), before)

        # 화면을 다시 그려도 새 작업·실행·모델 호출이 생기지 않는다.
        stages = list(self.calls)
        app.run()
        self.assertEqual(self.calls, stages)
        self.assertEqual(self.model_calls.call_count, 1)
        self.assertEqual(len(list(self.store.iterdir())), 1)


    def test_time_split_draft_requires_explicit_strategy_and_shows_target_check(self):
        from streamlit.testing.v1 import AppTest

        reply = flow.model_reply(dict(flow.ROLES, time="timestamp"),
                                 changes=[{"path": "split.strategy", "value": "time", "reason": "시간 순서"}])
        self.model_calls.side_effect = lambda payload, config: reply
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.text_input(key="_widget_task_source").set_value(str(self.source))
        # 목표에 target 컬럼 이름이 없다. 기본 초안에는 target도 없다.
        app.text_area(key="task_goal").set_value("불량 예측용으로 분할·전처리해 주세요")
        app.checkbox(key="task_consent_a").check()
        next(item for item in app.button if item.label == "작업 시작").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])

        # v2는 계획 단계에서 time을 고를 수 있지만 시간 설정은 사용자가 직접 정한다. 모델의 time 제안은 random으로 바꾸지 않고 비운다.
        strategy = next(item for item in app.selectbox if item.label == "분할 방식")
        self.assertIsNone(strategy.value)
        self.assertIn("time", strategy.options)
        self.assertEqual(sum(item.value.startswith("현재 로컬 모델의 한계:")
                             and "내용의 타당성을 보장하지 않습니다" in item.value
                             for item in app.text), 1)
        self.assertTrue(any("모델이 time 분할을 제안했습니다." in item.value and "모델은 시간 설정을 정하지 않습니다" in item.value
                            for item in app.text))
        # 분할 방식과 target 미지정은 모두 필수 질문으로 드러난다.
        self.assertTrue(any("[필수]" in item.value and "분할 방식" in item.value for item in app.text))
        self.assertTrue(any("[필수]" in item.value and "target(예측 대상)이 지정되지 않았습니다" in item.value for item in app.text))

        stages = list(self.calls)
        preview(app)
        self.assertTrue(any("답해야 하는 질문이 남아 있습니다." in item.value for item in app.error))
        self.assertFalse([item for item in app.button if item.label == "확인하고 진행"])
        self.assertEqual(self.calls, stages)

    def test_duplicate_header_offers_consented_numbered_copy_on_screen(self):
        import pandas as pd
        from streamlit.testing.v1 import AppTest

        frame = pd.read_csv(self.source)
        frame.insert(1, "zero", frame["x"] * 2)
        frame.insert(2, "zero_", frame["x"] * 3)
        source = self.root / "duplicate.csv"
        frame.to_csv(source, index=False, header=["x", "zero", "zero", "label", "id", "time"])
        self.model_calls.side_effect = lambda payload, config: flow.model_reply(dict(flow.ROLES, zero_1="수치형", zero_2="수치형"))
        # 번호 붙인 복사본 해결 동작은 기존(legacy) CSV 작업의 차단 화면이다. 새 작업의 중복 헤더는 입력 해석 대기로 간다
        # (test_ui_tabular_v2). 저장된 기존 작업을 화면에서 열어 같은 해결 경로가 유지되는지 검사한다.
        task = flow.orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", source, flow.LIMITS, self.model, root=self.store)
        self.assertEqual(task["kind"], "tabular_csv")
        flow.orchestrator.diagnose(task["task_id"], root=self.store)
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.session_state["ui_task_id"] = task["task_id"]
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any("같은 컬럼명이 여러 번" in item.value for item in app.error))
        self.assertTrue(any(item.value == "확인된 사유: 영향받는 컬럼 수 2" for item in app.text))
        self.model_calls.assert_not_called()

        # 동의 없이 누르면 정비하지 않는다.
        next(item for item in app.button if item.label == "번호 붙인 복사본으로 진행").click().run()
        self.model_calls.assert_not_called()
        app.checkbox(key="task_action_consent_header_dedupe").check()
        next(item for item in app.button if item.label == "번호 붙인 복사본으로 진행").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any("계획 확인" in item.value for item in app.subheader))
        self.assertTrue(any("번호를 붙였습니다" in item.value for item in app.text))
        self.assertEqual(self.model_calls.call_count, 1)

    def test_quality_review_shows_findings_and_user_can_exclude_flagged_columns(self):
        import pandas as pd
        from streamlit.testing.v1 import AppTest

        frame = pd.read_csv(self.source)
        frame["const"] = 5
        frame.to_csv(self.source, index=False)
        roles = dict(flow.ROLES, const="수치형")
        self.model_calls.side_effect = lambda payload, config: flow.model_reply(roles)
        task = flow.orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", self.source, flow.LIMITS, self.model, root=self.store)
        task = flow.orchestrator.diagnose(task["task_id"], root=self.store)
        task = flow.orchestrator.confirm_plan(task["task_id"], flow.plan(roles), root=self.store)
        self.assertEqual(task["problem"]["code"], "QUALITY_REVIEW")
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.session_state["ui_task_id"] = task["task_id"]
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value == "확인된 지적 1건" for item in app.text))
        flagged = next(item for item in app.checkbox if item.label == "품질 검사가 지적한 특징 1개를 제외로 표시합니다")
        # 기본은 바꾸지 않는다. 사용자가 고를 때만 제외로 표시한다.
        self.assertFalse(flagged.value)
        flagged.check()
        preview_and_confirm(self, app)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        stored = flow.task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["phase"], stored["confirmed"]["plan"]["roles"]["const"]), ("awaiting_apply", "제외"))

    def test_candidate_failure_with_detail_code_opens_plan_review_with_confirmed_cause(self):
        import pandas as pd
        from streamlit.testing.v1 import AppTest

        frame = pd.read_csv(self.source)
        frame["code"] = ["v" + str(index % 40) for index in range(len(frame))]
        frame.to_csv(self.source, index=False)
        roles = dict(flow.ROLES, code="범주형")
        preparation = dict(flow.PREPARATION, max_categories=8)
        self.model_calls.side_effect = lambda payload, config: flow.model_reply(roles)
        task = flow.orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", self.source, flow.LIMITS, self.model, root=self.store)
        task = flow.orchestrator.diagnose(task["task_id"], root=self.store)
        task = flow.orchestrator.confirm_plan(task["task_id"], flow.plan(roles, preparation=preparation), root=self.store)
        self.assertEqual(task["problem"]["code"], "STEP_FAILED")
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.session_state["ui_task_id"] = task["task_id"]
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value == "필요한 결정 — 계획 확인" for item in app.subheader))
        self.assertTrue(any(item.value == "확인된 원인 (worker 세부 코드)" for item in app.warning))
        self.assertTrue(any(item.value == "관련 항목: code (로컬 진단 기준 후보)" for item in app.text))

    def test_binning_and_scaling_answers_conflict_is_explained_and_user_picks_one(self):
        """작업 9b600512…의 답변 조합 재현: target 필수 답변 + 구간화 quantile·범주 결측 처리·스케일링 standard.

        검증은 그대로 거부하고, 충돌한 질문 번호와 값을 보여 준다. 다른 답변은 바꾸지 않으며, 사용자가 한쪽을 none으로 고르면 진행한다.
        """
        from streamlit.testing.v1 import AppTest

        roles = dict(flow.ROLES, label="수치형")
        self.model_calls.side_effect = lambda payload, config: flow.model_reply(roles, questions=[
            {"path": "preparation.binning", "text": "분포가 좁은 수치 컬럼을 구간화할까요?", "options": ["quantile", "uniform", "none"]},
            {"path": "preparation.categorical_imputation", "text": "범주 결측을 어떻게 처리할까요?",
             "options": ["missing_category", "most_frequent"]},
            {"path": "preparation.scaling", "text": "수치 스케일링 방법을 고를까요?", "options": ["standard", "minmax", "none"]}])
        task = flow.orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", self.source, flow.LIMITS, self.model, root=self.store)
        task = flow.orchestrator.diagnose(task["task_id"], root=self.store)
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.session_state["ui_task_id"] = task["task_id"]
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        texts = [item.value for item in app.text]
        numbers = {path: next(text.split(".")[0] for text in texts if text.startswith("질문 ") and needle in text)
                   for path, needle in (("target", "target(예측 대상)"), ("preparation.binning", "구간화"),
                                        ("preparation.scaling", "스케일링"))}
        picks = {numbers["target"]: "label", numbers["preparation.binning"]: "quantile",
                 numbers["preparation.scaling"]: "standard"}
        categorical = next(text.split(".")[0] for text in texts if text.startswith("질문 ") and "범주 결측" in text)
        picks[categorical] = "missing_category"

        def answer(choices):
            for number, value in choices.items():
                box = next(item for item in app.selectbox if item.label == number + " 답변")
                box.set_value(next(option for option in box.options if option.split(" (")[0] == value))

        answer(picks)
        preview(app)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value.startswith("수치 범주화(binning)와 스케일링(scaling)은 함께 적용할 수 없습니다") for item in app.error))
        texts = [item.value for item in app.text]
        self.assertIn(numbers["preparation.binning"] + " 답변: preparation.binning = quantile", texts)
        self.assertIn(numbers["preparation.scaling"] + " 답변: preparation.scaling = standard", texts)
        stored = flow.task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["phase"], stored["steps"], stored["confirmed"]), ("awaiting_review", {}, None))
        # 다른 답변은 그대로 남아 있다. 사용자가 스케일링만 none으로 고르면 진행한다.
        self.assertEqual(next(item for item in app.selectbox if item.label == numbers["target"] + " 답변").value.split(" (")[0], "label")
        answer({numbers["preparation.scaling"]: "none"})
        preview_and_confirm(self, app)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        stored = flow.task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual(stored["phase"], "awaiting_apply", stored.get("problem"))
        prepared = stored["confirmed"]["plan"]
        self.assertEqual((prepared["roles"]["label"], prepared["preparation"]["binning"], prepared["preparation"]["scaling"],
                          prepared["preparation"]["categorical_imputation"]), ("target", "quantile", "none", "missing_category"))

    def open_task(self, roles, changes=(), questions=()):
        from streamlit.testing.v1 import AppTest

        self.model_calls.side_effect = lambda payload, config: flow.model_reply(roles, changes=list(changes), questions=list(questions))
        task = flow.orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", self.source, flow.LIMITS, self.model, root=self.store)
        task = flow.orchestrator.diagnose(task["task_id"], root=self.store)
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.session_state["ui_task_id"] = task["task_id"]
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        return task, app

    def test_option_guide_binning_off_without_question_and_apply_warning_with_revise_path(self):
        """정상 0/1 특징 + 모델 권장 clipping·구간화: 사전 경고, 질문 없이 구간화 끄기, 적용 전 정보 손실 경고, 계획 수정 경로."""
        import pandas as pd

        frame = pd.read_csv(self.source)
        frame["flag"] = [1 if index % 10 == 0 else 0 for index in range(len(frame))]
        frame.to_csv(self.source, index=False)
        roles = dict(flow.ROLES, flag="수치형")
        task, app = self.open_task(roles, changes=[
            {"path": "preparation.outlier", "value": "iqr_clip", "reason": "극단값 영향을 줄입니다"},
            {"path": "preparation.binning", "value": "quantile", "reason": "분포를 단순화합니다"}])
        texts = [item.value for item in app.text]
        self.assertIn("■ 수치 구간화", texts)
        self.assertIn("모델 의견(확인된 사실 아님): 분포를 단순화합니다", texts)
        self.assertTrue(any(item.value.startswith("사전 경고 ") for item in app.warning))
        self.assertTrue(any("flag" in text and "상수" in text for text in texts))
        # 모델 문장·컬럼명은 마크다운으로 해석하는 위치에 두지 않는다.
        self.assertFalse(any("flag" in item.value or "단순화" in item.value for item in list(app.markdown) + list(app.warning)))

        binning = next(item for item in app.selectbox if item.label == "수치 구간화 (binning)")
        self.assertEqual((binning.value, "none" in binning.options or any("[none]" in option for option in binning.options)), ("quantile", True))
        binning.set_value("none")
        preview_and_confirm(self, app)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        stored = flow.task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual(stored["phase"], "awaiting_apply", stored.get("problem"))
        self.assertEqual((stored["confirmed"]["plan"]["preparation"]["binning"], stored["confirmed"]["plan"]["preparation"]["outlier"]),
                         ("none", "iqr_clip"))

        # 적용 전 경고: Train 후보 계산의 정보 손실. 설정은 바꾸지 않고 계획 수정 경로를 보인다.
        self.assertTrue(any(item.value.startswith("처리 영향 경고 ") for item in app.warning))
        self.assertTrue(any("flag" in item.value and "정보 손실" in item.value for item in app.text))
        self.assertEqual(self.calls.count("apply_preparation"), 0)
        next(item for item in app.button if item.label == "적용하지 않고 계획 수정").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value.startswith("적용하지 않고 계획을 고치는 중입니다") for item in app.text))
        outlier = next(item for item in app.selectbox if item.label == "극단값 자르기(clipping) (outlier)")
        self.assertEqual(outlier.value, "iqr_clip")
        outlier.set_value("none")
        preview_and_confirm(self, app)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        stored = flow.task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["phase"], stored["attempt"], stored["confirmed"]["plan"]["preparation"]["outlier"]),
                         ("awaiting_apply", 2, "none"))
        self.assertEqual(self.calls.count("apply_preparation"), 0)

    def test_form_value_and_answer_conflict_is_chosen_by_user_on_screen(self):
        task, app = self.open_task(flow.ROLES, questions=[
            {"path": "preparation.binning", "text": "수치 컬럼을 구간화할까요?", "options": ["quantile", "none"]}])
        self.assertEqual(sum(item.value.startswith("현재 로컬 모델의 한계:")
                             and "다른 로컬·API 모델의 품질은 별도 평가" in item.value for item in app.text), 1)
        number = next(item.value.split(".")[0] for item in app.text if item.value.startswith("질문 ") and "구간화할까요" in item.value)
        box = next(item for item in app.selectbox if item.label == number + " 답변")
        box.set_value(next(option for option in box.options if option.split(" (")[0] == "quantile"))
        next(item for item in app.selectbox if item.label == "수치 구간화 (binning)").set_value("uniform")
        preview(app)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any("자동으로 고르지 않습니다" in item.value for item in app.error))
        self.assertTrue(any(item.value.startswith("충돌 1: preparation.binning") for item in app.text))
        self.assertFalse([item for item in app.button if item.label == "확인하고 진행"])
        # 고르지 않으면 진행하지 않는다.
        next(item for item in app.button if item.label == "고른 값으로 다시 미리보기").click().run()
        self.assertTrue(any(item.value == "모든 충돌 항목에서 쓸 값을 고르세요." for item in app.warning))
        next(item for item in app.radio if item.label == "충돌 1에 쓸 값").set_value("질문 답변")
        next(item for item in app.button if item.label == "고른 값으로 다시 미리보기").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        next(item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")).check()
        next(item for item in app.button if item.label == "확인하고 진행").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        stored = flow.task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["phase"], stored["confirmed"]["plan"]["preparation"]["binning"]), ("awaiting_apply", "quantile"),
                         stored.get("problem"))

    def test_unsaved_completion_is_recovered_from_screen(self):
        from streamlit.testing.v1 import AppTest
        import tests.agents.test_task_orchestrator as base

        self.model_calls.side_effect = lambda payload, config: flow.model_reply()
        task = flow.orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", self.source, flow.LIMITS, self.model, root=self.store)
        task = flow.orchestrator.diagnose(task["task_id"], root=self.store)
        task = flow.orchestrator.confirm_plan(task["task_id"], flow.plan(), root=self.store)

        with base.TaskOrchestratorTest.fail_completed_save(None), self.assertRaises(ValueError):
            flow.orchestrator.approve_apply(task["task_id"], root=self.store)

        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.session_state["ui_task_id"] = task["task_id"]
        app.run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value == "필요한 결정 — 완료 기록 복구" for item in app.subheader))
        next(item for item in app.button if item.label == "검증 후 완료 기록 복구").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(flow.task_store.load_task(task["task_id"], root=self.store)["phase"], "completed")
        self.assertTrue(any(item.value.startswith("완료 기록 복구됨") for item in app.info))

    def test_record_read_failure_shows_confirmed_cause(self):
        from streamlit.testing.v1 import AppTest

        from src.agents import task_store

        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=60).run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        app.session_state["ui_task_id"] = "0" * 32

        with patch("src.agents.task_store.load_task", side_effect=task_store.TaskRecordError("RECORD_TOO_LARGE", "x")):
            app.run()

        self.assertTrue(any("크기 상한(512 KiB)" in item.value for item in app.error))
        self.assertFalse(any("새 작업으로 시작하세요" in item.value for item in app.error))

    def test_goal_guide_is_shown_without_filling_the_goal_and_typed_goal_is_sent_unchanged(self):
        from streamlit.testing.v1 import AppTest

        from src.ui import task_view

        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        goal = app.text_area(key="task_goal")
        # 안내·예시는 표시만 한다. 입력란은 비어 있고 특정 데이터셋 컬럼명을 기본값처럼 채우지 않는다.
        self.assertEqual(goal.value, "")
        self.assertNotIn("Survived", goal.placeholder)
        self.assertIn(task_view.GOAL_NOTICE, [item.value for item in app.caption])
        self.assertTrue(any(item.label == "목표 작성 안내 (선택 사항)" for item in app.expander))
        texts = [item.value for item in app.text]
        self.assertIn(task_view.GOAL_EXAMPLE, texts)
        self.assertTrue(all("- " + item in texts for item in task_view.GOAL_ITEMS))

        typed = "  생존 여부 예측. 대상 컬럼은 label, id는 식별자로만 둡니다.\n그룹은 모릅니다.  "
        app.text_input(key="_widget_task_source").set_value(str(self.source))
        goal.set_value(typed)
        app.checkbox(key="task_consent_a").check()
        real = flow.orchestrator.start_task

        with patch("src.agents.task_orchestrator.start_task", side_effect=real) as start:
            next(item for item in app.button if item.label == "작업 시작").click().run()

        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(start.call_args.args[0], typed)
        # 화면은 적은 문장을 바꾸지 않고 전달하며, 기록에도 그대로 저장된다(기존 저장 형식).
        self.assertEqual(flow.task_store.load_task(next(self.store.iterdir()).name, root=self.store)["goal"], typed)

    def test_non_admin_server_cannot_start_task(self):
        from streamlit.testing.v1 import AppTest

        with patch("src.ui.runtime_notice.server_permissions",
                   return_value={"elevated": False, "pid": 1, "python": "synthetic"}):
            app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=60).run()
            self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
            app.text_input(key="_widget_task_source").set_value(str(self.source))
            app.text_area(key="task_goal").set_value("label 예측")
            app.checkbox(key="task_consent_a").check()
            start = next(item for item in app.button if item.label == "작업 시작")
            # 관리자 권한이 없으면 제출 버튼 자체가 비활성이라 브라우저에서도 누를 수 없다.
            self.assertTrue(start.disabled)

        self.assertFalse(self.store.exists() and any(self.store.iterdir()))
        self.model_calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()

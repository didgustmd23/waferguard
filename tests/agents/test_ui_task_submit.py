"""통합 작업 실행 제출 잠금. 처리 중 재실행 요청·중복 클릭·실패·권한 거부에서 실행이 끊기거나 중복되지 않아야 한다.

단위 테스트는 Streamlit의 동작(화면 갱신·세션 상태 접근 때 대기 중인 재실행 요청을 BaseException으로 처리)을 가짜 st로
모의한다. 화면 테스트는 실제 Streamlit AppTest에서 실행 도중 재실행 요청을 넣는다. OS transport와 로컬 모델 응답만 모의한다.
"""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.ui import task_submit


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Rerun(BaseException):
    """Streamlit RerunException 대역(BaseException)."""


class FakeSession(dict):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def get(self, *args):
        self.owner.yield_point()
        return super().get(*args)

    def __getitem__(self, key):
        self.owner.yield_point()
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.owner.yield_point()
        super().__setitem__(key, value)

    def pop(self, *args):
        self.owner.yield_point()
        return super().pop(*args)


class FakeSt:
    """화면 갱신·세션 상태 접근마다 대기 중인 재실행 요청이 있으면 Rerun을 낸다(Streamlit의 yield point)."""

    def __init__(self):
        self.pending = False
        self.shown = []
        self.session_state = FakeSession(self)

    def yield_point(self):
        if self.pending:
            self.pending = False
            raise Rerun()

    def rerun(self):
        raise Rerun()

    def _show(self, kind, text):
        self.yield_point()
        self.shown.append((kind, text))

    def info(self, text):
        self._show("info", text)

    def error(self, text):
        self._show("error", text)

    def warning(self, text):
        self._show("warning", text)

    @contextmanager
    def spinner(self, label, show_time=False):
        self.yield_point()
        yield
        self.yield_point()

    @contextmanager
    def container(self):
        self.yield_point()
        yield


class TaskSubmitUnitTest(unittest.TestCase):
    def setUp(self):
        self.st = FakeSt()
        self.calls = []

    def click(self, action="confirm", task_id="t", bound=None):
        """클릭 callback → 같은 재실행의 reconcile(app.py) 순서를 따른다."""
        task_submit.reserve(self.st, action, task_id, task_submit.generation(self.st), bound)
        task_submit.reconcile(self.st)
        self.assertTrue(task_submit.reserved(self.st, action, task_id))

    def execute(self, call, verify=None):
        """버튼 아래 실행. 실행 뒤에는 항상 다시 그린다(Rerun)."""
        with self.assertRaises(Rerun):
            task_submit.execute(self.st, call, "처리 중", verify)

    def notice(self):
        self.st.shown.clear()
        task_submit.show_notice(self.st)
        return self.st.shown

    def test_rerun_requested_during_call_does_not_interrupt_it_and_result_is_applied_next_run(self):
        def call(progress):
            # 사용자가 처리 중 다른 위젯을 눌렀다. 실행 안에는 yield point가 없으므로 끝까지 진행한다.
            self.st.pending = True
            self.calls.append("orchestrator")
            return {"link": "t1", "notice": [("info", "완료")], "pop": ["draft"]}

        self.st.session_state["draft"] = {"plan": 1}
        self.click()
        self.execute(call)
        state = task_submit.submit_state(self.st)
        self.assertEqual(self.calls, ["orchestrator"])
        self.assertEqual(state.request["state"], "done")
        self.assertEqual(self.st.shown[0], ("info", task_submit.PROCESSING))
        self.assertTrue(task_submit.locked(self.st))

        task_submit.reconcile(self.st)
        self.assertFalse(task_submit.locked(self.st))
        self.assertEqual((dict.get(self.st.session_state, "ui_task_id"), "draft" in dict(self.st.session_state)), ("t1", False))
        self.assertEqual(state.generation, 1)
        self.assertEqual(self.notice(), [("info", "완료")])

    def test_interrupted_result_application_is_retried_without_running_again(self):
        self.click()
        self.execute(lambda progress: self.calls.append("orchestrator") or {"link": "t1"})

        # 결과 반영의 세션 쓰기 직전에 재실행이 오면 요청을 지우지 않는다. 다음 재실행에서 같은 결과를 다시 반영한다.
        self.st.pending = True
        with self.assertRaises(Rerun):
            task_submit.reconcile(self.st)

        self.assertTrue(task_submit.locked(self.st))
        task_submit.reconcile(self.st)
        self.assertEqual((self.calls, dict.get(self.st.session_state, "ui_task_id")), (["orchestrator"], "t1"))
        self.assertFalse(task_submit.locked(self.st))

    def test_duplicate_and_stale_clicks_reserve_only_once(self):
        seen = task_submit.generation(self.st)
        task_submit.reserve(self.st, "confirm", "t", seen)
        first = task_submit.submit_state(self.st).request
        task_submit.reserve(self.st, "confirm", "t", seen)
        task_submit.reserve(self.st, "apply", "t", seen)
        self.assertIs(task_submit.submit_state(self.st).request, first)
        self.assertFalse(task_submit.idle(self.st, seen))
        task_submit.reconcile(self.st)
        self.execute(lambda progress: self.calls.append("run") or {})
        task_submit.reconcile(self.st)

        # 실행 전에 그린 화면에서 늦게 도착한 클릭은 다시 예약하지 않는다.
        task_submit.reserve(self.st, "confirm", "t", seen)
        self.assertFalse(task_submit.locked(self.st))
        self.assertFalse(task_submit.idle(self.st, seen))
        self.assertTrue(task_submit.idle(self.st, task_submit.generation(self.st)))
        self.assertEqual(self.calls, ["run"])

    def test_same_button_clicked_again_before_execution_is_taken_over_by_the_new_run_once(self):
        self.click(bound="digest-shown")
        # 첫 클릭의 재실행이 실행 지점 전에 중지됐다(Streamlit fastReruns). 같은 화면의 같은 버튼을 다시 눌렀다.
        task_submit.reserve(self.st, "confirm", "t", 0, "digest-shown")
        task_submit.reconcile(self.st)
        self.assertTrue(task_submit.reserved(self.st, "confirm", "t"))
        self.execute(lambda progress: self.calls.append("run") or {})
        task_submit.reconcile(self.st)
        self.assertEqual(self.calls, ["run"])
        self.assertFalse(task_submit.locked(self.st))

    def test_stopped_run_does_not_execute_after_the_new_run_took_the_reservation_over(self):
        self.click(bound="digest-shown")
        state = task_submit.submit_state(self.st)
        request = state.request
        # 다른 재실행이 같은 버튼 클릭으로 예약을 이어받았다(아직 그 재실행의 reconcile 전).
        task_submit.reserve(self.st, "confirm", "t", 0, "digest-shown")
        # 중지 요청을 받은 이전 재실행이 마지막 yield point를 지나 실행 지점에 왔다. 실행하지 않는다.
        self.execute(lambda progress: self.calls.append("old run") or {})
        self.assertEqual(self.calls, [])
        self.assertIs(state.request, request)
        self.assertEqual(request["state"], "reserved")

    def test_other_widget_before_execution_releases_without_running(self):
        self.click(bound="digest-shown")
        # 다른 조작으로 새 재실행이 시작됐다. 그 사이 바뀐 값으로 다시 준비하지 않고 실행 없이 풀며 다시 누르게 한다.
        task_submit.reconcile(self.st)
        self.assertFalse(task_submit.locked(self.st))
        self.assertFalse(task_submit.reserved(self.st, "confirm", "t"))
        self.assertEqual(self.notice(), [task_submit.STALE])
        self.assertEqual(self.calls, [])

    def test_bound_value_is_the_one_drawn_on_the_clicked_screen(self):
        self.st.session_state["shown"] = "digest-later"
        self.click(bound="digest-shown")
        self.assertEqual(task_submit.bound(self.st), "digest-shown")

    def test_reservation_without_rendered_button_is_released_without_call(self):
        task_submit.reserve(self.st, "confirm", "t", 0)
        task_submit.reconcile(self.st)

        with self.assertRaises(Rerun):
            task_submit.finish(self.st)

        self.assertFalse(task_submit.locked(self.st))
        self.assertEqual(task_submit.generation(self.st), 1)

    def test_overlapping_run_keeps_lock_and_waits_until_the_running_thread_finishes(self):
        started, release = threading.Event(), threading.Event()

        def call(progress):
            started.set()
            release.wait(10)
            self.calls.append("orchestrator")
            return {"notice": [("info", "완료")]}

        self.click(task_id="t")
        worker = threading.Thread(target=lambda: self.assertRaises(Rerun, task_submit.execute, self.st, call, "처리 중"))
        worker.start()
        self.assertTrue(started.wait(10))

        # 처리 중 위젯 조작으로 새 재실행이 시작됐다. 아직 실행 중이므로 확인 불가로 정리하거나 잠금을 풀지 않는다.
        task_submit.reconcile(self.st)
        self.assertTrue(task_submit.locked(self.st))
        task_submit.reserve(self.st, "confirm", "t", task_submit.generation(self.st))
        self.assertEqual(task_submit.submit_state(self.st).request["state"], "running")

        # 새 재실행은 실행 중인 버튼·단계 자리에서 끝날 때까지 기다린 뒤 다시 그린다.
        threading.Timer(0.2, release.set).start()
        with self.assertRaises(Rerun):
            task_submit.wait_if_running(self.st, "t")

        worker.join(10)
        self.assertIn(("info", task_submit.WAITING), self.st.shown)
        task_submit.reconcile(self.st)
        self.assertEqual(self.calls, ["orchestrator"])
        self.assertFalse(task_submit.locked(self.st))
        self.assertEqual(self.notice(), [("info", "완료")])

    def test_base_exception_without_stored_completion_reports_unclear_and_is_not_swallowed(self):
        class Stop(BaseException):
            pass

        def call(progress):
            progress["link"] = "new-task"
            raise Stop()

        self.click(action="start", task_id=None)
        with self.assertRaises(Stop):
            task_submit.execute(self.st, call, "처리 중", lambda progress: None)

        self.assertEqual(task_submit.submit_state(self.st).request["state"], "aborted")
        task_submit.reconcile(self.st)
        self.assertFalse(task_submit.locked(self.st))
        # 결과를 모르는 작업 시작도 만든 작업에 화면을 연결해 저장 상태를 볼 수 있게 한다.
        self.assertEqual(dict.get(self.st.session_state, "ui_task_id"), "new-task")
        self.assertEqual(self.notice(), [task_submit.UNCLEAR])

    def test_base_exception_after_stored_completion_shows_confirmed_stage_instead_of_unclear(self):
        class Stop(BaseException):
            pass

        def call(progress):
            raise Stop()

        self.click()
        with self.assertRaises(Stop):
            task_submit.execute(self.st, call, "처리 중", lambda progress: "저장된 작업 기록으로 확인: 적용 확인 필요")

        task_submit.reconcile(self.st)
        self.assertEqual(self.notice(), [("info", "저장된 작업 기록으로 확인: 적용 확인 필요")])

    def test_normal_completion_with_empty_result_is_not_reported_as_unclear(self):
        verified = []

        def verify(progress):
            verified.append(progress)
            return "저장된 작업 기록으로 확인"

        # 알릴 내용이 없는 정상 완료({})는 결과를 받은 것이다. 저장 상태 대조·대체 안내를 쓰지 않는다.
        self.click()
        self.execute(lambda progress: {}, verify)
        self.assertEqual(task_submit.submit_state(self.st).request["state"], "done")
        task_submit.reconcile(self.st)
        self.assertFalse(task_submit.locked(self.st))
        self.assertEqual(self.notice(), [])
        self.assertEqual(verified, [])

    def test_completion_without_a_dict_result_still_uses_the_stored_state_check(self):
        # 결과가 dict가 아니면(형식 오류) 정상 결과로 보지 않고 기존 대체 경로로 정리한다. 자동 재실행은 없다.
        self.click()
        self.execute(lambda progress: None, lambda progress: None)
        task_submit.reconcile(self.st)
        self.assertEqual(self.notice(), [task_submit.UNCLEAR])
        self.assertFalse(task_submit.locked(self.st))

    def test_decline_releases_lock_and_redraws_with_notice(self):
        task_submit.reserve(self.st, "confirm", "t", 0)

        with self.assertRaises(Rerun):
            task_submit.decline(self.st, [("warning", "동의하세요")])

        self.assertFalse(task_submit.locked(self.st))
        self.assertEqual(self.notice(), [("warning", "동의하세요")])
        self.assertEqual(self.notice(), [])


class StoredStateVerificationTest(unittest.TestCase):
    """결과를 받지 못한 실행의 완료 판정(task_view._verify_stored). 대기·완료·문제 단계이고 기록이 진행했을 때만 확인한다."""

    def verify(self, record, task_id="t", revision=3, progress=None):
        from src.ui import task_view

        with patch("src.agents.task_store.load_task", return_value=record) as load:
            result = task_view._verify_stored(task_id, revision)(progress or {})

        return result, load

    def test_resting_phase_with_advanced_revision_is_confirmed_with_stage(self):
        result, _ = self.verify({"phase": "awaiting_apply", "revision": 5})
        self.assertIn("현재 단계: 적용 확인 필요", result)

    def test_problem_phase_is_reported_as_stage_not_success(self):
        result, _ = self.verify({"phase": "needs_attention", "revision": 5})
        self.assertIn("현재 단계: 확인이 필요한 문제", result)
        self.assertNotIn("완료", result.split("현재 단계")[1])

    def test_in_progress_or_unchanged_record_is_not_confirmed(self):
        for record in ({"phase": "processing", "revision": 9}, {"phase": "awaiting_review", "revision": 3}):
            with self.subTest(record=record):
                self.assertIsNone(self.verify(record)[0])

    def test_start_uses_task_id_left_by_the_run(self):
        result, load = self.verify({"phase": "awaiting_review", "revision": 4}, task_id=None, revision=None,
                                   progress={"link": "new-task"})
        load.assert_called_once_with("new-task")
        self.assertIn("현재 단계: 계획 확인 필요", result)
        self.assertIsNone(self.verify({"phase": "diagnosing", "revision": 4}, task_id=None, revision=None,
                                      progress={"link": "new-task"})[0])
        self.assertIsNone(self.verify({"phase": "awaiting_review", "revision": 4}, task_id=None, revision=None)[0])


class PipelineTransportCleanupTest(unittest.TestCase):
    def test_profile_cleanup_runs_even_if_progress_subscriber_raises_base_exception(self):
        from src.agents import module_pipeline_transport as transport
        from src.agents.operation_progress import observe_progress

        class Interrupt(BaseException):
            pass

        def subscriber(event):
            if event.stage == "cleanup":
                raise Interrupt()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(task_id="synthetic", request_id="synthetic", output_directory=str(root / "output"),
                                      scratch_directory=str(root / "scratch"), limits=None)

            with patch.object(transport, "AppContainer") as container, \
                    patch.object(transport, "code_manifest", side_effect=ValueError("synthetic")), \
                    observe_progress(subscriber), self.assertRaises(Interrupt):
                transport.run_pipeline_worker(request, root, threading.Event())

            container.return_value.close.assert_called_once_with()


def request_rerun():
    """실행 도중 사용자가 다른 위젯을 눌러 재실행 요청이 도착한 상황."""
    from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    ctx = get_script_run_ctx()
    ctx.script_requests.request_rerun(RerunData(query_string=ctx.query_string, page_script_hash=ctx.page_script_hash))


class TaskScreenLockTest(unittest.TestCase):
    from tests.agents import test_task_orchestrator as flow

    setUp_flow = flow.TaskOrchestratorTest.setUp
    setUp_csv = flow.TaskOrchestratorTest.setUp_csv
    start_patch = flow.TaskOrchestratorTest.start_patch
    transport = flow.TaskOrchestratorTest.transport

    def setUp(self):
        from tests.agents.test_tabular_task_v2 import add_v2_patches

        flow = self.flow
        self.setUp_flow()
        self.start_patch("src.agents.task_store.STORE_ROOT", new=self.store)
        self.permissions = self.start_patch("src.ui.runtime_notice.server_permissions",
                                            return_value={"elevated": True, "pid": 1, "python": "synthetic"})
        self.model_calls = self.start_patch("src.agents.task_recommend.request_local_proposal",
                                            side_effect=lambda payload, config: flow.model_reply())
        add_v2_patches(self, self.store, self.handoff_runs)

    def app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()

    def task(self):
        return self.flow.task_store.load_task(next(self.store.iterdir()).name, root=self.store)

    def start(self, app):
        app.text_input(key="_widget_task_source").set_value(str(self.source))
        app.text_area(key="task_goal").set_value("label 예측용으로 분할·전처리해 주세요")
        app.checkbox(key="task_consent_a").check()
        self.button(app, "작업 시작").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])

    def review(self, app):
        """최종 계획을 미리 보고 동의한다(확정 직전 상태)."""
        self.start(app)
        self.button(app, "최종 계획 미리보기").click().run()
        next(item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")).check()

    @staticmethod
    def button(app, label):
        return next(item for item in app.button if item.label == label)

    def assert_no_interrupt(self, app):
        self.assertFalse(any("중단되었습니다" in item.value for item in app.warning))
        self.assertFalse([item for item in app.button if item.label == "새 작업 시작" and not item.disabled
                          and self.task()["phase"] in {"diagnosing", "processing", "applying"}])

    def test_rerun_request_during_start_creates_one_task_and_keeps_it_connected(self):
        real = self.flow.orchestrator.start_task

        def start_task(*args, **kwargs):
            request_rerun()
            return real(*args, **kwargs)

        app = self.app()
        with patch("src.agents.task_orchestrator.start_task", side_effect=start_task) as spy:
            self.start(app)

        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(list(self.store.iterdir())), 1)
        self.assertEqual(app.session_state["ui_task_id"], self.task()["task_id"])
        self.assertEqual(self.task()["phase"], "awaiting_review", self.task().get("problem"))
        self.assertTrue(any("계획 확인" in item.value for item in app.subheader))
        self.assert_no_interrupt(app)

    def test_rerun_request_and_progress_signal_during_confirm_do_not_interrupt_processing(self):
        from src.agents.operation_progress import report_progress

        real = self.flow.orchestrator.confirm_plan

        def confirm_plan(*args, **kwargs):
            # 처리 중 클릭으로 재실행 요청이 왔고, transport가 진행 신호를 보낸다. 화면 갱신이 없으므로 끊기지 않아야 한다.
            request_rerun()
            report_progress("worker_wait")
            return real(*args, **kwargs)

        app = self.app()
        self.review(app)

        with patch("src.agents.task_orchestrator.confirm_plan", side_effect=confirm_plan) as spy:
            self.button(app, "확인하고 진행").click().run()

        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(self.task()["phase"], "awaiting_apply", self.task().get("problem"))
        self.assertTrue(any("필요한 결정 — 적용" in item.value for item in app.subheader))
        self.assertFalse(self.button(app, "적용하고 마무리").disabled)
        self.assertIsNone(app.session_state[task_submit.KEY].request)
        self.assert_no_interrupt(app)

    def test_rerun_request_during_apply_completes_once_without_interrupt(self):
        real = self.flow.orchestrator.approve_apply

        def approve_apply(*args, **kwargs):
            request_rerun()
            return real(*args, **kwargs)

        app = self.app()
        self.review(app)
        self.button(app, "확인하고 진행").click().run()
        app.checkbox(key="task_consent_d").check()

        with patch("src.agents.task_orchestrator.approve_apply", side_effect=approve_apply) as spy:
            self.button(app, "적용하고 마무리").click().run()

        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(self.task()["phase"], "completed", self.task().get("problem"))
        self.assertTrue(any(item.value == "최종 결과" for item in app.subheader))
        self.assertEqual(self.calls.count("apply_preparation"), 1)
        self.assert_no_interrupt(app)

    @staticmethod
    def leaves(app):
        """화면 순서대로 잎 요소(블록 제외)를 모은다."""
        found = []

        def walk(node):
            for child in getattr(node, "children", {}).values():
                if getattr(child, "children", None):
                    walk(child)
                else:
                    found.append(child)

        walk(app._tree)
        return found

    def assert_processing_below(self, app, label):
        """처리 중 안내가 실행한 버튼 바로 다음 요소로 그려졌는지."""
        leaves = self.leaves(app)
        index = next(number for number, item in enumerate(leaves) if getattr(item, "label", None) == label)
        self.assertEqual(getattr(leaves[index + 1], "value", None) if index + 1 < len(leaves) else None, task_submit.PROCESSING)
        notices = [item for item in leaves if isinstance(getattr(item, "value", None), str) and item.value == task_submit.PROCESSING]
        self.assertEqual(len(notices), 1)
        self.assertIs(notices[0], leaves[index + 1])

    def test_processing_notice_is_under_the_start_button_and_task_is_connected_after(self):
        app = self.app()

        # 실행 직후 다시 그리기를 막아 실행 중이던 화면을 그대로 본다.
        with patch("streamlit.rerun"):
            self.start(app)

        self.assert_processing_below(app, "작업 시작")
        self.assertTrue(self.button(app, "작업 시작").disabled)
        self.assertTrue(app.radio(key="ui_task_screen").disabled)
        self.assertEqual(len(list(self.store.iterdir())), 1)

        app.run()
        self.assertEqual(app.session_state["ui_task_id"], self.task()["task_id"])
        self.assertTrue(any("계획 확인" in item.value for item in app.subheader))
        self.assertFalse(any(item.value == task_submit.UNCLEAR[1] for item in app.error))
        self.assertFalse(app.radio(key="ui_task_screen").disabled)

    def test_screen_is_locked_with_notice_under_the_confirm_button_and_runs_once(self):
        app = self.app()
        self.review(app)
        self.assertFalse(self.button(app, "확인하고 진행").disabled)
        real = self.flow.orchestrator.confirm_plan

        with patch("streamlit.rerun"), patch("src.agents.task_orchestrator.confirm_plan", side_effect=real) as spy:
            self.button(app, "확인하고 진행").click().run()

        # 실행 중이던 화면: 안내는 버튼 바로 아래, 실행 버튼·화면 이동은 잠금. 실행에 쓰는 동의 값은 잠그지 않는다.
        self.assertEqual(spy.call_count, 1)
        self.assert_processing_below(app, "확인하고 진행")
        self.assertTrue(self.button(app, "확인하고 진행").disabled)
        self.assertTrue(self.button(app, "최종 계획 미리보기").disabled)
        self.assertTrue(next(item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")).value)
        self.assertTrue(app.radio(key="ui_task_screen").disabled)
        self.assertTrue(app.checkbox(key="ui_show_advanced_tools").disabled)
        self.assertFalse([item for item in app.button if item.label == "새 작업 시작" and not item.disabled])

        with patch("src.agents.task_orchestrator.confirm_plan", side_effect=real) as again:
            app.run()

        again.assert_not_called()
        self.assertEqual(self.task()["phase"], "awaiting_apply", self.task().get("problem"))
        self.assertFalse(any(item.value == task_submit.UNCLEAR[1] for item in app.error))
        self.assertFalse(app.radio(key="ui_task_screen").disabled)
        self.assertFalse(self.button(app, "적용하고 마무리").disabled)

    def test_processing_notice_is_under_the_apply_button(self):
        app = self.app()
        self.review(app)
        self.button(app, "확인하고 진행").click().run()
        app.checkbox(key="task_consent_d").check()

        with patch("streamlit.rerun"):
            self.button(app, "적용하고 마무리").click().run()

        self.assert_processing_below(app, "적용하고 마무리")
        app.run()
        self.assertEqual(self.task()["phase"], "completed", self.task().get("problem"))
        self.assertTrue(any(item.value == "최종 결과" for item in app.subheader))

    def test_failure_and_permission_denial_release_lock_and_keep_task(self):
        app = self.app()
        self.review(app)
        before = self.task()

        with patch("src.agents.task_orchestrator.confirm_plan", side_effect=RuntimeError("synthetic")) as spy:
            self.button(app, "확인하고 진행").click().run()

        self.assertEqual(spy.call_count, 1)
        self.assertTrue(any(item.value.startswith("처리를 시작하지 못했습니다.") for item in app.error))
        self.assertFalse(self.button(app, "확인하고 진행").disabled)
        self.assertFalse(app.radio(key="ui_task_screen").disabled)
        self.assertEqual((self.task()["phase"], self.task()["revision"]), (before["phase"], before["revision"]))

        from src.ui.runtime_notice import EXECUTION_PERMISSION_ERROR

        self.permissions.return_value = {"elevated": False, "pid": 1, "python": "synthetic"}
        with patch("src.agents.task_orchestrator.confirm_plan") as spy:
            self.button(app, "확인하고 진행").click().run()

        spy.assert_not_called()
        self.assertIn(EXECUTION_PERMISSION_ERROR, [item.value for item in app.error])
        self.assertFalse(self.button(app, "확인하고 진행").disabled)
        self.assertEqual(self.task()["revision"], before["revision"])

        # 동의를 풀고 누르면 실행하지 않고 안내한다. 잠금은 남지 않는다.
        self.permissions.return_value = {"elevated": True, "pid": 1, "python": "synthetic"}
        next(item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")).uncheck()
        with patch("src.agents.task_orchestrator.confirm_plan") as spy:
            self.button(app, "확인하고 진행").click().run()

        spy.assert_not_called()
        self.assertIn("최종 계획과 진행 범위를 확인하고 체크해 주세요.", [item.value for item in app.warning])
        self.assertFalse(self.button(app, "확인하고 진행").disabled)


if __name__ == "__main__":
    unittest.main()

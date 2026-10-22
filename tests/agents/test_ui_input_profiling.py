"""1번 화면 입력 진단 → 승인 실행 → 결과 표시 → 역할 지정·저장·연결과 LLM 전달 차단을 검사한다.

합성 CSV만 사용한다. OS transport(AppContainer·Job)만 모의 처리하고 worker 계산·부모 검증은 실제 코드를 쓴다.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.agents.input_snapshot import create_snapshot
from src.agents.profile_store import build_profile, save_profile
from src.ui import commentary_view
from src.ui.approval_flow import ApprovalFlow
from src.ui.commentary_service import build_packet
from src.ui.profiling_service import (UiInputProfiling, blocked_reason, create_profiling, linked_profile, link_record,
                                      load_profile_result, missing_over)
from src.ui.profiling_view import BLOCKED_MESSAGES, _bulk_numeric, current_profiling
from src.ui.result_service import load_results
from src.ui.state import default_settings
from tests.agents.test_input_profile import CSV_BYTES
from tests.agents.test_input_profile_pipeline import _transport


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"wall_seconds": 60, "memory_mib": 2048, "processes": 2}


class InputProfilingServiceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "synthetic.csv"
        source.write_bytes(CSV_BYTES)
        self.snapshot_dir = self.root / "snapshot"
        create_snapshot(source, self.snapshot_dir, max_input_bytes=4096)
        (self.root / "runs").mkdir()
        patcher = patch("src.agents.pipeline_approval_cli._directory",
                        side_effect=lambda name, create_parent=False: self.root / "runs" / name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def completed(self):
        profiling = create_profiling("sel01", str(self.snapshot_dir.resolve()), LIMITS, confirmed=True)
        flow = ApprovalFlow(profiling, workflow=False)

        with patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=_transport) as launch:
            flow.run("start")
            launch.assert_not_called()
            self.assertTrue(flow.ready)
            self.assertIs(flow.summary["diagnostic_only"], True)
            self.assertIsNone(flow.summary["roles"])
            self.assertIs(flow.summary["llm_transfer"], False)
            flow.run("approve")
            self.assertEqual(launch.call_count, 1)

        self.assertEqual(flow.state["status"], "completed")
        return profiling

    def test_creation_requires_confirmation_and_starts_nothing(self):
        with self.assertRaises(ValueError):
            create_profiling("sel01", str(self.snapshot_dir.resolve()), LIMITS)

        with patch("src.agents.module_pipeline.run_pipeline_worker") as launch:
            profiling = create_profiling("sel01", str(self.snapshot_dir.resolve()), LIMITS, confirmed=True)

        launch.assert_not_called()
        self.assertTrue(profiling.name.startswith("input-profile-"))
        self.assertEqual(profiling.read_state().status, "created")
        self.assertIsNone(profiling.approval_id)

        with self.assertRaises(ValueError):
            load_profile_result(profiling)

    def test_approved_run_result_is_validated_and_has_no_raw_values(self):
        profiling = self.completed()
        result = load_profile_result(profiling)
        self.assertIs(result["diagnostic_only"], True)
        self.assertIs(result["llm_transfer"], False)
        self.assertEqual(result["summary"]["row_count"], 8)
        rows = {row["column"]: row for row in result["columns"]}
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows["int_col"]["mean"], 9.75)
        self.assertEqual(rows["int_col"]["outlier_candidate_count"], 1)
        self.assertEqual(rows["all_inf"]["finite_count"], 0)
        self.assertIsNone(rows["all_inf"]["mean"])
        self.assertEqual(rows["text_col"]["blank_string_count"], 2)
        self.assertNotIn("CANARY", json.dumps(result, ensure_ascii=False))
        # 결측률 0.25(float_col)·0.625(both_inf)만 0.2 초과
        self.assertEqual(missing_over(result, 0.2), 2)

        # 일반 결과 화면 경로는 진단 run을 열지 않는다(프로파일 없는 계약은 일반 검증기에 들어가지 않음).
        with self.assertRaisesRegex(ValueError, "DIAGNOSTIC_RESULT_NOT_GENERAL"):
            load_results(profiling)

    def blocked_run(self, payload):
        source = self.root / "blocked.csv"
        source.write_bytes(payload)
        snapshot = self.root / ("blocked_snapshot_" + str(len(payload)))
        create_snapshot(source, snapshot, max_input_bytes=4096)
        profiling = create_profiling("sel02", str(snapshot.resolve()), LIMITS, confirmed=True)
        flow = ApprovalFlow(profiling, workflow=False)

        with patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=_transport):
            flow.run("start")
            flow.run("approve")

        self.assertEqual(flow.state["status"], "blocked_input")
        return profiling

    def test_duplicate_column_names_keep_block_and_show_fixed_reason(self):
        """중복 컬럼 차단은 그대로이며 사유는 고정 코드·개수만 반환한다. 이름 변경·결과 표시는 없다."""
        profiling = self.blocked_run(b"zero,SECRET_NAME,zero,zero\n1,2,3,4\n5,6,7,8\n")
        reason = blocked_reason(profiling)
        self.assertEqual(reason, {"code": "DUPLICATE_COLUMN_NAMES", "duplicated_names": 1, "affected_columns": 3})
        self.assertNotIn("zero", json.dumps(reason))

        with self.assertRaises(ValueError):
            load_profile_result(profiling)

        self.assertEqual(BLOCKED_MESSAGES["DUPLICATE_COLUMN_NAMES"].count("허용하지 않으면 컬럼명을 바꾸지 않습니다"), 1)

    def test_snapshot_change_after_block_is_not_reported_as_other_reason(self):
        """재검증 실패는 중복 여부를 단정하지 않는 별도 사유로 표시하고 세션에 저장하지 않는다."""
        from unittest.mock import MagicMock
        from src.ui.profiling_view import _render_blocked

        profiling = self.blocked_run(b"zero,a,zero" + bytes([10]) + b"1,2,3" + bytes([10]))
        (Path(profiling.snapshot_directory) / "input.bin").write_bytes(b"x,y" + bytes([10]) + b"1,2" + bytes([10]))

        with self.assertRaises(ValueError):
            blocked_reason(profiling)

        fake = MagicMock()
        fake.session_state = {}
        _render_blocked(fake, profiling)
        self.assertEqual(fake.error.call_args.args[0], BLOCKED_MESSAGES["REASON_UNVERIFIED"])
        self.assertNotIn("ui_input_profile_blocked", fake.session_state)
        fake.button.assert_called_once()

    def test_header_read_is_bounded(self):
        from src.ui.profiling_service import MAX_HEADER_CHARS, read_header

        path = self.root / "wide.csv"
        path.write_text(",".join("c" + str(index) for index in range(MAX_HEADER_CHARS // 4)) + chr(10) + "1" + chr(10),
                        encoding="utf-8")

        with self.assertRaises(ValueError):
            read_header(path)

        path.write_text("a,b" + chr(10) + "1,2" + chr(10), encoding="utf-8")
        self.assertEqual(read_header(path), ["a", "b"])

    def test_recovery_buttons_only_for_finished_states(self):
        from src.ui.profiling_view import recovery_action

        self.assertEqual(recovery_action("blocked_input"), "choose_other")

        for status in ("failed_runtime", "security_locked", "cancelled"):
            self.assertEqual(recovery_action(status), "restart")

        for status in ("created", "running", "awaiting_approval", "completed", "unknown"):
            self.assertIsNone(recovery_action(status))

    def test_empty_column_name_is_reported_without_auto_fill(self):
        profiling = self.blocked_run(b"a,,b\n1,2,3\n4,5,6\n")
        self.assertEqual(blocked_reason(profiling), {"code": "EMPTY_COLUMN_NAME"})

    def test_completed_run_has_no_blocked_reason(self):
        self.assertIsNone(blocked_reason(self.completed()))

    def test_commentary_is_explicitly_blocked(self):
        """정책이 자동 해설을 허용해도 진단 작업·결과는 패킷 생성·전송까지 가지 않는다."""
        profiling = self.completed()
        result = load_profile_result(profiling)
        fake = SimpleNamespace(session_state={"ui_settings": default_settings()})

        with patch.object(commentary_view, "disclosure_decision", return_value="AUTO_ALLOWED"), \
                patch.object(commentary_view, "load_policy") as policy, \
                patch.object(commentary_view, "build_packet") as packet, \
                patch.object(commentary_view, "send_once") as send:
            commentary_view.after_completion(fake, profiling, workflow=False)

        policy.assert_not_called()
        packet.assert_not_called()
        send.assert_not_called()
        self.assertTrue(commentary_view.diagnostic_blocked(profiling))

        with self.assertRaisesRegex(ValueError, "COMMENTARY_DIAGNOSTIC_BLOCKED"):
            build_packet(dict(result, status="completed", files=[]), {"tables": True, "graphs": True})

    def saved(self, purpose):
        store = self.root / "profiles"
        patcher = patch("src.agents.profile_store.STORE_ROOT", store)
        patcher.start()
        self.addCleanup(patcher.stop)
        split = {"strategy": "random"} if purpose == "split" else None
        profile = build_profile(("a", "b"), {"a": "수치형", "b": "제외"}, purpose=purpose, profile_id="ui_" + purpose, split=split)
        return save_profile(profile)

    def test_link_follows_selection_purpose_and_saved_content(self):
        saved = self.saved("eda")
        state = {"ui_linked_profile": link_record({"source_selection_id": "sel01", "result_id": "r1"}, saved)}
        self.assertIs(state["ui_linked_profile"]["execution_approved"], False)
        self.assertEqual(linked_profile(state, "sel01", "eda_review_v1"), saved["local_file"])
        self.assertEqual(linked_profile(state, "sel02", "eda_review_v1"), "")
        self.assertEqual(linked_profile(state, "sel01", "split_review_v1"), "")
        self.assertEqual(linked_profile({}, "sel01", "eda_review_v1"), "")

        # 저장 후 다른 내용으로 바뀐 파일은 기본값으로 연결하지 않는다.
        path = Path(saved["local_file"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value["max_plot_features"] = 3
        path.write_text(json.dumps(value), encoding="utf-8")
        self.assertEqual(linked_profile(state, "sel01", "eda_review_v1"), "")
        path.unlink()
        self.assertEqual(linked_profile(state, "sel01", "eda_review_v1"), "")

    def test_split_review_form_uses_only_matching_split_profile(self):
        from src.ui.workflow_view import linked_split_profile

        selection = {"mode": "기존 CSV snapshot", "selection_id": "sel01"}
        saved = self.saved("split")
        state = {"ui_source_mode": "기존 CSV snapshot", "ui_source_selection": selection,
                 "ui_linked_profile": link_record({"source_selection_id": "sel01", "result_id": "r"}, saved)}
        self.assertEqual(linked_split_profile(SimpleNamespace(session_state=state)), saved["local_file"])
        state["ui_linked_profile"]["workflow_id"] = "eda_review_v1"
        self.assertEqual(linked_split_profile(SimpleNamespace(session_state=state)), "")
        self.assertEqual(linked_split_profile(SimpleNamespace(session_state={})), "")

    def test_bulk_numeric_only_on_click_and_preserves_user_roles(self):
        result = {"result_id": "r1", "columns": [
            {"column": "a", "dtype_group": "numeric"}, {"column": "b", "dtype_group": "numeric"},
            {"column": "c", "dtype_group": "text"}, {"column": "d", "dtype_group": "boolean"}]}
        fake = SimpleNamespace(session_state={})
        from src.ui.profiling_view import _roles

        self.assertEqual(set(_roles(fake, result)["roles"].values()), {"제외"})
        fake.session_state["ui_profile_roles"]["roles"]["b"] = "target"
        fake.session_state["ui_profile_preview"] = {"result_id": "r1"}
        _bulk_numeric(fake, result)
        self.assertEqual(fake.session_state["ui_profile_roles"]["roles"], {"a": "수치형", "b": "target", "c": "제외", "d": "제외"})
        self.assertNotIn("ui_profile_preview", fake.session_state)

    def test_bulk_exclude_only_changes_feature_roles_and_reports_counts(self):
        """상수·전체 결측 특징만 제외로 바꾸고, 사용자가 지정한 target·식별자·group·timestamp는 그대로 둔다."""
        from src.ui.profiling_view import EXCLUDE_NOTICE, _bulk_exclude_uninformative, _roles, uninformative_columns

        result = {"result_id": "r1", "columns": [
            {"column": "const_num", "dtype_group": "numeric", "observed_unique": 1},
            {"column": "const_cat", "dtype_group": "text", "observed_unique": 1},
            {"column": "all_missing", "dtype_group": "numeric", "observed_unique": 0},
            {"column": "const_target", "dtype_group": "numeric", "observed_unique": 1},
            {"column": "const_id", "dtype_group": "text", "observed_unique": 1},
            {"column": "const_group", "dtype_group": "text", "observed_unique": 1},
            {"column": "const_time", "dtype_group": "text", "observed_unique": 1},
            {"column": "already_excluded", "dtype_group": "numeric", "observed_unique": 1},
            {"column": "useful", "dtype_group": "numeric", "observed_unique": 5},
        ]}
        fake = SimpleNamespace(session_state={"ui_profile_preview": {"result_id": "r1"}})
        roles = _roles(fake, result)["roles"]
        roles.update(const_num="수치형", const_cat="범주형", all_missing="수치형", const_target="target",
                     const_id="식별자", const_group="group", const_time="timestamp", useful="수치형")
        self.assertEqual(len(uninformative_columns(result)), 8)

        # 렌더링만으로는 바뀌지 않는다. 클릭 callback을 호출할 때만 적용한다.
        self.assertEqual(roles["const_num"], "수치형")
        _bulk_exclude_uninformative(fake, result)
        self.assertEqual(fake.session_state["ui_profile_roles"]["roles"], {
            "const_num": "제외", "const_cat": "제외", "all_missing": "제외", "const_target": "target",
            "const_id": "식별자", "const_group": "group", "const_time": "timestamp",
            "already_excluded": "제외", "useful": "수치형"})
        self.assertEqual(fake.session_state["ui_profile_bulk_notice"],
                         {"result_id": "r1", "targets": 8, "changed": 3, "kept": 4, "before": 1, "after": 4})
        self.assertNotIn("ui_profile_preview", fake.session_state)
        self.assertEqual(fake.session_state["ui_profile_roles"]["revision"], 1)
        self.assertIn("데이터 파일은 변경하지 않습니다", EXCLUDE_NOTICE)

        # 적용 후에도 개별 수정 가능: 초안 역할은 일반 dict로 남고 다시 바꿀 수 있다.
        fake.session_state["ui_profile_roles"]["roles"]["const_num"] = "수치형"
        self.assertEqual(_roles(fake, result)["roles"]["const_num"], "수치형")

    def test_selection_change_discards_profiling_state(self):
        profiling = create_profiling("sel01", str(self.snapshot_dir.resolve()), LIMITS, confirmed=True)
        fake = SimpleNamespace(session_state={"ui_input_profiling": profiling, "ui_profile_roles": {}, "ui_profile_preview": {}})
        directory = str(self.snapshot_dir.resolve())
        self.assertIs(current_profiling(fake, {"selection_id": "sel01"}, directory), profiling)
        self.assertIsNone(current_profiling(fake, {"selection_id": "sel02"}, directory))
        self.assertEqual(set(fake.session_state) & {"ui_input_profiling", "ui_profile_roles", "ui_profile_preview"}, set())


class InputProfilingAppTest(unittest.TestCase):
    """실제 Streamlit 화면으로 진단 실행부터 프로파일 저장·3번 입력란 연결까지 진행한다."""

    def test_screen_one_flow_saves_and_links_profile(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = root / "synthetic.csv"
            source.write_bytes(CSV_BYTES)
            snapshot = root / "snapshot"
            create_snapshot(source, snapshot, max_input_bytes=4096)
            (root / "runs").mkdir()
            store = root / "profiles"

            with patch("src.agents.pipeline_approval_cli._directory",
                       side_effect=lambda name, create_parent=False: root / "runs" / name), \
                    patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=_transport) as launch, \
                    patch("src.agents.profile_store.STORE_ROOT", store), \
                    patch("src.ui.runtime_notice.server_permissions",
                          return_value={"elevated": True, "pid": 1, "python": "synthetic"}), \
                    patch("src.agents.local_ollama.request_local_proposal") as model, \
                    patch("src.ui.commentary_service.send_once") as send:
                app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=60).run()
                app.selectbox(key="ui_source_mode_widget").set_value("기존 CSV snapshot").run()
                app.text_input(key="_widget_source_snapshot_path").set_value(str(snapshot))
                next(item for item in app.button if item.label == "로컬 데이터 선택 확인").click().run()
                app.toggle(key="ui_profiling_open").set_value(True).run()
                self.assertEqual(list((root / "runs").iterdir()), [])

                # 동의 없이 제출하면 작업을 만들지 않는다.
                next(item for item in app.button if item.label == "입력 진단 작업 준비").click().run()
                self.assertEqual(list((root / "runs").iterdir()), [])
                next(item for item in app.checkbox if item.label.startswith("선택한 snapshot의 입력 진단")).check()
                next(item for item in app.button if item.label == "입력 진단 작업 준비").click().run()
                self.assertEqual(len(list((root / "runs").iterdir())), 1)
                launch.assert_not_called()

                app.button(key="approval_flow_start").click().run()
                launch.assert_not_called()
                self.assertFalse(any(item.label == "결과 자료 민감도" for item in app.selectbox))
                next(item for item in app.checkbox if item.label.startswith("표시한 범위를 승인")).check()
                next(item for item in app.button if item.label == "승인하고 실행").click().run()
                self.assertEqual(launch.call_count, 1)
                self.assertFalse(app.exception, [item.value for item in app.exception])
                self.assertTrue(any("입력 진단 결과" == item.value for item in app.subheader))
                self.assertFalse(any("CANARY" in str(item.value) for item in app.dataframe))

                # 역할은 사용자가 누를 때만 일괄 지정한다.
                app.button(key="ui_profile_bulk_numeric").click().run()
                next(item for item in app.button if item.label == "프로파일 검증 미리보기").click().run()
                self.assertTrue(any("계약 검사를 통과" in item.value for item in app.success))
                next(item for item in app.checkbox if item.label.startswith("새 로컬 프로파일 파일만")).check()
                next(item for item in app.button if item.label == "프로파일 저장").click().run()

                saved = store / "ui_profile01.json"
                value = json.loads(saved.read_text(encoding="utf-8"))
                self.assertEqual(value["roles"]["numeric"], ["int_col", "float_col", "pos_inf", "neg_inf", "both_inf", "all_inf", "edge"])
                self.assertEqual(value["roles"]["categorical"], [])
                self.assertEqual(app.session_state["ui_linked_profile"]["path"], str(saved))

                # 3번 입력 결합 화면의 프로파일 기본값이 저장 경로로 연결된다(LLM 제안 저장 전이라 화면은 잠김).
                from src.ui.input_view import prepare_selected_input
                fake = SimpleNamespace(session_state={key: app.session_state[key] for key in (
                    "ui_linked_profile", "ui_source_selection", "ui_source_mode")}, text=lambda *_: None)
                self.assertEqual(prepare_selected_input(fake)[1], str(saved))

                # 진단·저장 이후 대화 전송 미리보기에도 컬럼명·프로파일 경로가 들어가지 않는다(모델 호출 없음).
                app.text_area[0].set_value("품질 검사 후 결측률을 확인해 주세요.")
                next(item for item in app.button if item.label == "요청 초안 보관").click().run()
                app.button(key="ui_preview_request").click().run()
                pending = json.dumps(app.session_state["ui_dialogue"].pending, ensure_ascii=False)
                self.assertIn("결측률", pending)

                for leaked in ("int_col", "pos_inf", "ui_profile01", "CANARY",
                               json.dumps(str(store))[1:-1], json.dumps(str(snapshot))[1:-1]):
                    self.assertNotIn(leaked, pending)

            model.assert_not_called()
            send.assert_not_called()


    def test_duplicate_block_shows_reason_and_choose_other_input_recovers(self):
        """차단 사유를 화면에 고정 문구로 보이고, 버튼으로 입력 선택만 해제해 새 CSV로 다시 진행한다."""
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            snapshots = {}

            for name, payload in (("dup", b"zero,SECRET_NAME,zero\n1,2,3\n4,5,6\n"), ("ok", CSV_BYTES)):
                source = root / (name + ".csv")
                source.write_bytes(payload)
                snapshots[name] = root / ("snapshot_" + name)
                create_snapshot(source, snapshots[name], max_input_bytes=4096)

            (root / "runs").mkdir()

            with patch("src.agents.pipeline_approval_cli._directory",
                       side_effect=lambda name, create_parent=False: root / "runs" / name),                     patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=_transport) as launch,                     patch("src.ui.runtime_notice.server_permissions",
                          return_value={"elevated": True, "pid": 1, "python": "synthetic"}):
                app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=60).run()
                app.selectbox(key="ui_source_mode_widget").set_value("기존 CSV snapshot").run()

                def select(name):
                    app.text_input(key="_widget_source_snapshot_path").set_value(str(snapshots[name]))
                    next(item for item in app.button if item.label == "로컬 데이터 선택 확인").click().run()

                def run_diagnostic():
                    next(item for item in app.checkbox if item.label.startswith("선택한 snapshot의 입력 진단")).check()
                    next(item for item in app.button if item.label == "입력 진단 작업 준비").click().run()
                    app.button(key="approval_flow_start").click().run()
                    next(item for item in app.checkbox if item.label.startswith("표시한 범위를 승인")).check()
                    next(item for item in app.button if item.label == "승인하고 실행").click().run()

                select("dup")
                app.toggle(key="ui_profiling_open").set_value(True).run()
                run_diagnostic()
                self.assertFalse(app.exception, [item.value for item in app.exception])
                errors = " ".join(item.value for item in app.error)
                self.assertIn("중복된 컬럼명", errors)
                self.assertIn("허용하지 않으면 컬럼명을 바꾸지 않습니다", errors)
                self.assertTrue(any("해당 컬럼 2개" in item.value for item in app.caption))
                shown = errors + " ".join(item.value for item in app.caption)
                self.assertNotIn("SECRET_NAME", shown)
                self.assertNotIn("zero", shown)
                self.assertFalse(any(item.value == "입력 진단 결과" for item in app.subheader))

                app.button(key="ui_profiling_choose_other").click().run()
                self.assertNotIn("ui_source_selection", app.session_state)
                self.assertEqual(app.text_input(key="_widget_source_snapshot_path").value, "")
                self.assertTrue(snapshots["dup"].exists())

                select("ok")
                app.toggle(key="ui_profiling_open").set_value(True).run()
                run_diagnostic()
                self.assertEqual(launch.call_count, 2)
                self.assertTrue(any(item.value == "입력 진단 결과" for item in app.subheader))
                self.assertEqual(len(list((root / "runs").iterdir())), 2)


    def test_bulk_exclude_button_on_screen(self):
        """실제 화면에서 클릭 시에만 적용하고 전후 개수·안내를 표시하며, 저장 JSON에서 제외되고 파일은 그대로다."""
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            payload = b"pid,const,empty,x,y" + bytes([10]) + bytes([10]).join(
                b"%d,5,,%d.5,%d" % (index, index, index * 2) for index in range(1, 7)) + bytes([10])
            source = root / "c.csv"
            source.write_bytes(payload)
            snapshot = root / "snapshot"
            create_snapshot(source, snapshot, max_input_bytes=4096)
            (root / "runs").mkdir()
            store = root / "profiles"

            with patch("src.agents.pipeline_approval_cli._directory",
                       side_effect=lambda name, create_parent=False: root / "runs" / name), \
                    patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=_transport), \
                    patch("src.agents.profile_store.STORE_ROOT", store), \
                    patch("src.ui.runtime_notice.server_permissions",
                          return_value={"elevated": True, "pid": 1, "python": "synthetic"}):
                app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=60).run()
                app.selectbox(key="ui_source_mode_widget").set_value("기존 CSV snapshot").run()
                app.text_input(key="_widget_source_snapshot_path").set_value(str(snapshot))
                next(item for item in app.button if item.label == "로컬 데이터 선택 확인").click().run()
                app.toggle(key="ui_profiling_open").set_value(True).run()
                next(item for item in app.checkbox if item.label.startswith("선택한 snapshot의 입력 진단")).check()
                next(item for item in app.button if item.label == "입력 진단 작업 준비").click().run()
                app.button(key="approval_flow_start").click().run()
                next(item for item in app.checkbox if item.label.startswith("표시한 범위를 승인")).check()
                next(item for item in app.button if item.label == "승인하고 실행").click().run()

                self.assertFalse(any("상수·전체 결측 컬럼" in item.value for item in app.info))
                app.button(key="ui_profile_bulk_numeric").click().run()
                app.button(key="ui_profile_bulk_exclude").click().run()
                notice = next(item.value for item in app.info if "상수·전체 결측 컬럼" in item.value)
                self.assertIn("2개 중 2개를 제외로 바꿨습니다", notice)
                self.assertIn("적용 전 0개 → 적용 후 2개", notice)
                self.assertIn("분석 대상에서 제외하며 데이터 파일은 변경하지 않습니다", notice)

                next(item for item in app.button if item.label == "프로파일 검증 미리보기").click().run()
                next(item for item in app.checkbox if item.label.startswith("새 로컬 프로파일 파일만")).check()
                next(item for item in app.button if item.label == "프로파일 저장").click().run()
                value = json.loads((store / "ui_profile01.json").read_text(encoding="utf-8"))
                self.assertEqual(value["roles"]["numeric"], ["pid", "x", "y"])
                self.assertEqual((snapshot / "input.bin").read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()

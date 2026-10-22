"""중복 컬럼명 자동 정비(사용자 허용 시)의 규칙·격리 worker·부모 검증·파생 입력 등록을 합성 CSV로 검사한다.

OS transport(AppContainer·Job)만 모의 처리한다. worker 계산·파일 생성·부모 검증은 실제 코드를 실행한다.
"""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.agents.input_profile_pipeline import DiagnosticInputContract, InputProfileTool, bind_input_profile
from src.agents.input_snapshot import Snapshot, create_snapshot, verify_snapshot
from src.agents.module_pipeline_contract import POLICY_VERSION, bind_pipeline
from src.agents.module_pipeline_results import read_header_dedupe_result
from src.agents.module_pipeline_worker import execute_request
from src.agents.reviewed_csv_pipeline_contract import bind_reviewed_csv_pipeline
from src.agents.reviewed_module_contract import demo_profile
from src.agents.state import LocalStateStore
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.input_profile import (DEDUPE_INPUT, DEDUPE_MAP, dedupe_names, header_dedupe_pipeline,
                                             parse_header, split_header)
from src.ui.approval_flow import ApprovalFlow
from src.ui.profiling_service import (create_profiling, dedupe_message, effective_snapshot, load_profile_result,
                                      register_derived_input)
from tests.agents.test_input_profile_pipeline import _transport


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIMITS = WorkerLimits(wall_seconds=60, memory_mib=2048, processes=2)
LF, CRLF = bytes((10,)), bytes((13, 10))
# 실제 사례와 같은 구조: 28개 컬럼 중 19개가 zero
NAMES = ["Passengerid", "Age", "Fare", "Sex", "sibsp"] + ["zero"] * 19 + ["Parch", "Pclass", "Embarked", "Survived"]
HEADER = ",".join(NAMES).encode()
ROWS = [",".join(str((row * 7 + column) % 5) for column in range(len(NAMES))).encode() for row in range(6)]
DUP_CSV = LF.join([HEADER, *ROWS]) + LF
MESSAGE = "zero라는 컬럼명이 19번 사용되어 zero_1~zero_19로 변경했습니다. 데이터 값과 컬럼 순서는 유지했고, 원본은 변경하지 않았습니다."


class DedupeRuleTest(unittest.TestCase):
    def test_numbering_keeps_unique_names_order_and_count(self):
        renamed = dedupe_names(NAMES)
        self.assertEqual(len(renamed), len(NAMES))
        self.assertEqual(renamed[5:24], [f"zero_{index}" for index in range(1, 20)])
        self.assertEqual(renamed[:5] + renamed[24:], NAMES[:5] + NAMES[24:])
        self.assertEqual(len(set(renamed)), len(renamed))

    def test_existing_names_never_collide(self):
        self.assertEqual(dedupe_names(["zero", "zero_2", "zero", "zero"]), ["zero__1", "zero_2", "zero__2", "zero__3"])
        self.assertEqual(dedupe_names(["a", "b", "a", "b"]), ["a_1", "b_1", "a_2", "b_2"])
        with self.assertRaises(ValueError):
            dedupe_names(["x", "x", "x_1", "x__1", "x___1", "x____1"])

    def test_empty_names_are_out_of_scope(self):
        for names in (["a", "", "a"], ["a", " ", "b"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                dedupe_names(names)

    def test_header_record_split_handles_quotes_bom_and_crlf(self):
        bom = bytes((0xEF, 0xBB, 0xBF))
        data = bom + b'a,"b' + LF + b'c",a' + CRLF + b"1,2,3" + CRLF
        has_bom, header, terminator = split_header(data)
        self.assertTrue(has_bom)
        self.assertEqual(terminator, "crlf")
        self.assertEqual(parse_header(header), ["a", "b" + LF.decode() + "c", "a"])

        with self.assertRaises(ValueError):
            split_header(b"a,b,c")


class DedupeWorkerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.count = 0

    def staged(self, payload):
        self.count += 1
        source = self.root / f"source{self.count}.csv"
        source.write_bytes(payload)
        snapshot_dir = self.root / f"snapshot{self.count}"
        snapshot = create_snapshot(source, snapshot_dir, max_input_bytes=1024 * 1024)
        states = LocalStateStore(self.root / "runs" / f"run{self.count}")
        states.create("synthetic_operator", policy_version=POLICY_VERSION)
        tool = InputProfileTool(states, DiagnosticInputContract(str(snapshot_dir.resolve()), snapshot, LIMITS),
                                header_dedupe_pipeline())
        request = tool._request(f"{self.count:032x}")
        return tool, request, tool._stage(request), snapshot_dir, snapshot

    def run_worker(self, payload):
        tool, request, root, snapshot_dir, snapshot = self.staged(payload)
        reply = execute_request(request, root)
        return tool, reply, root / "output" / "result" / "pipeline", snapshot_dir, snapshot

    def test_duplicate_names_create_derived_input_with_identical_data_bytes(self):
        tool, reply, pipeline, snapshot_dir, snapshot = self.run_worker(DUP_CSV)
        self.assertEqual(reply.status, "succeeded")
        child = pipeline / "header_dedupe"
        derived = (child / DEDUPE_INPUT).read_bytes()
        mapping = json.loads((child / DEDUPE_MAP).read_text(encoding="utf-8"))

        self.assertEqual(derived.split(LF, 1)[1], DUP_CSV.split(LF, 1)[1])
        self.assertEqual(parse_header(derived.split(LF, 1)[0] + LF), dedupe_names(NAMES))
        self.assertEqual(mapping["original_sha256"], snapshot.sha256)
        self.assertEqual(mapping["changed_positions"], list(range(5, 24)))
        self.assertEqual(mapping["columns"][5], {"position": 5, "original": "zero", "renamed": "zero_1"})
        self.assertEqual(dedupe_message(mapping), MESSAGE)
        self.assertEqual((snapshot_dir / "input.bin").read_bytes(), DUP_CSV)

        result = read_header_dedupe_result(pipeline, tool.module_plan, tool.contract.snapshot_directory, snapshot)
        self.assertTrue(result["changed"])
        self.assertEqual(result["derived_sha256"], hashlib.sha256(derived).hexdigest())

    def test_crlf_bom_and_quoted_names_are_preserved(self):
        bom = bytes((0xEF, 0xBB, 0xBF))
        payload = bom + b'id,"a,b",id,"x' + LF + b'y"' + CRLF + b'1,"v,w",2,"m' + LF + b'n"' + CRLF + b"3,u,4,k" + CRLF
        tool, reply, pipeline, _, snapshot = self.run_worker(payload)
        self.assertEqual(reply.status, "succeeded")
        derived = (pipeline / "header_dedupe" / DEDUPE_INPUT).read_bytes()
        self.assertTrue(derived.startswith(bom + b'id_1,"a,b",id_2,"x' + LF + b'y"' + CRLF))
        self.assertTrue(derived.endswith(payload[payload.index(CRLF) + 2:]))
        self.assertIsNotNone(read_header_dedupe_result(pipeline, tool.module_plan, tool.contract.snapshot_directory, snapshot))

    def test_no_duplicates_creates_no_derived_file(self):
        tool, reply, pipeline, _, snapshot = self.run_worker(b"a,b" + LF + b"1,2" + LF)
        self.assertEqual(reply.status, "succeeded")
        self.assertEqual({path.name for path in (pipeline / "header_dedupe").iterdir()}, {DEDUPE_MAP})
        result = read_header_dedupe_result(pipeline, tool.module_plan, tool.contract.snapshot_directory, snapshot)
        self.assertFalse(result["changed"])

    def test_empty_names_and_broken_header_are_blocked_not_fixed(self):
        for payload in (b"a,,a" + LF + b"1,2,3" + LF, b'a,"b,a' + LF + b"1,2" + LF):
            with self.subTest(payload=payload):
                _, reply, pipeline, _, _ = self.run_worker(payload)
                self.assertEqual(reply.status, "blocked_input")
                self.assertFalse((pipeline / "header_dedupe" / DEDUPE_INPUT).exists())

    def test_quote_parity_mismatch_is_blocked_not_reinterpreted(self):
        """헤더 끝 탐색과 csv 해석이 어긋나는 입력(보안 검토 재현 사례)은 정비하지 않고 차단한다."""
        payload = bytes.fromhex("612c612c6222632c22780a79220a312c322c332c340a362c372c382c390a")

        with self.assertRaises(ValueError):
            parse_header(split_header(payload)[1])

        tool, reply, pipeline, _, snapshot = self.run_worker(payload)
        self.assertEqual(reply.status, "blocked_input")
        self.assertFalse((pipeline / "header_dedupe" / DEDUPE_INPUT).exists())

        # 인용되지 않은 필드 안 따옴표로 헤더 끝이 뒤로 밀리는 경우도 한 레코드가 아니어서 거부한다.
        with self.assertRaises(ValueError):
            parse_header(split_header(b'a,a,b"c,d' + LF + b'1,2,3,4' + LF + b'"' + LF)[1])

    def test_parent_rejects_tampered_dedupe_results(self):
        tool, reply, pipeline, _, snapshot = self.run_worker(DUP_CSV)
        self.assertEqual(reply.status, "succeeded")

        def mapping_rename(directory):
            path = directory / "header_dedupe" / DEDUPE_MAP
            value = json.loads(path.read_text(encoding="utf-8"))
            value["columns"][5]["renamed"] = "zero_99"
            path.write_text(json.dumps(value), encoding="utf-8")

        def data_byte(directory):
            path = directory / "header_dedupe" / DEDUPE_INPUT
            data = bytearray(path.read_bytes())
            data[-2] = ord("4") if data[-2] != ord("4") else ord("3")
            path.write_bytes(bytes(data))

        def header_swap(directory):
            path = directory / "header_dedupe" / DEDUPE_INPUT
            path.write_bytes(path.read_bytes().replace(b"zero_1,zero_2", b"zero_2,zero_1", 1))

        cases = {
            "mapping_rename": mapping_rename,
            "data_byte": data_byte,
            "header_swap": header_swap,
            "extra_file": lambda directory: (directory / "header_dedupe" / "extra.csv").write_text("x"),
            "derived_missing": lambda directory: (directory / "header_dedupe" / DEDUPE_INPUT).unlink(),
            "appended_row": lambda directory: (directory / "header_dedupe" / DEDUPE_INPUT).open("ab").write(b"9" + LF),
        }

        for name, change in cases.items():
            with self.subTest(name):
                directory = self.root / ("tampered_" + name)
                shutil.copytree(pipeline, directory)
                change(directory)
                self.assertIsNone(read_header_dedupe_result(directory, tool.module_plan, tool.contract.snapshot_directory, snapshot))

        # 원본 snapshot이 바뀌면 같은 결과도 받아들이지 않는다.
        self.assertIsNone(read_header_dedupe_result(pipeline, tool.module_plan, tool.contract.snapshot_directory,
                                                    Snapshot(snapshot.size_bytes, "0" * 64, snapshot.max_input_bytes)))

    def test_copied_worker_runs_dedupe_with_allowlisted_files_only(self):
        """정비 계획 상수·헤더 규칙이 복사 허용 목록만으로 별도 -I -S 프로세스에서 동작한다."""
        import subprocess
        import sys
        from src.agents.module_pipeline_transport import ENTRY, WORKER_FILES

        tool, request, root, _, snapshot = self.staged(DUP_CSV)

        for name in WORKER_FILES:
            target = root / "tools" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / name, target)

        shutil.copyfile(PROJECT_ROOT / ENTRY, root / "tools" / "worker.py")
        script = (
            "import sys;sys.path[:0]=sys.argv[1:3];from pathlib import Path;"
            "from src.agents.worker_contract import WorkerRequest;import worker;"
            "root=Path(sys.argv[3]);r=WorkerRequest.from_json((root/'input/request.json').read_text());"
            "assert worker.execute_request(r,root).status=='succeeded'"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script, str(root / "tools"),
             str(Path(sys.prefix) / "Lib/site-packages"), str(root)],
            cwd=self.root, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIsNotNone(read_header_dedupe_result(root / "output" / "result" / "pipeline", tool.module_plan,
                                                       tool.contract.snapshot_directory, snapshot))

    def test_general_path_duplicate_rejection_is_unchanged(self):
        source = self.root / "general.csv"
        source.write_bytes(DUP_CSV)
        snapshot_dir = self.root / "general_snapshot"
        snapshot = create_snapshot(source, snapshot_dir, max_input_bytes=1024 * 1024)

        with self.assertRaises(ValueError):
            bind_pipeline(header_dedupe_pipeline(), demo_profile())

        with self.assertRaises(ValueError):
            bind_reviewed_csv_pipeline(header_dedupe_pipeline(), snapshot_dir, snapshot, demo_profile(), LIMITS)

        changed = header_dedupe_pipeline()
        changed["steps"][0]["settings"] = {"separator": "-"}

        with self.assertRaises(ValueError):
            bind_input_profile(changed, snapshot_dir, snapshot, LIMITS)

        # 정비 옵션 없이 진단하면 기존처럼 중복 헤더로 차단한다.
        from src.preprocessing.input_profile import input_profile_pipeline

        states = LocalStateStore(self.root / "runs" / "general")
        states.create("synthetic_operator", policy_version=POLICY_VERSION)
        tool = InputProfileTool(states, DiagnosticInputContract(str(snapshot_dir.resolve()), snapshot, LIMITS),
                                input_profile_pipeline())
        request = tool._request("e" * 32)
        self.assertEqual(execute_request(request, tool._stage(request)).status, "blocked_input")


class DerivedInputFlowTest(unittest.TestCase):
    """승인 → 정비 → 파생 입력 등록 → 새 승인으로 파생 입력 진단까지 서비스 계층으로 진행한다."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "dup.csv"
        source.write_bytes(DUP_CSV)
        self.snapshot_dir = self.root / "snapshot"
        create_snapshot(source, self.snapshot_dir, max_input_bytes=1024 * 1024)
        (self.root / "runs").mkdir()
        patcher = patch("src.agents.pipeline_approval_cli._directory",
                        side_effect=lambda name, create_parent=False: self.root / "runs" / name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.limits = {"wall_seconds": 60, "memory_mib": 2048, "processes": 2}

    def approve(self, profiling):
        flow = ApprovalFlow(profiling, workflow=False)

        with patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=_transport) as launch:
            flow.run("start")
            launch.assert_not_called()
            summary = flow.summary
            flow.run("approve")

        return flow.state["status"], summary

    def test_dedupe_then_diagnose_derived_input_with_new_approval(self):
        dedupe = create_profiling("sel01", str(self.snapshot_dir.resolve()), self.limits, confirmed=True, header_dedupe=True)
        status, summary = self.approve(dedupe)
        self.assertEqual(status, "completed")
        self.assertIn("원본과 기존 snapshot은 변경하지 않습니다", summary["input_change"])

        with self.assertRaises(ValueError):
            load_profile_result(dedupe)

        derived = register_derived_input(dedupe, root=self.root / "derived")
        self.assertTrue(derived["changed"])
        self.assertEqual(derived["message"], MESSAGE)
        record = json.loads(Path(derived["record_path"]).read_text(encoding="utf-8"))
        self.assertEqual(record["original"]["snapshot"]["sha256"], hashlib.sha256(DUP_CSV).hexdigest())
        self.assertEqual(record["original"]["snapshot_directory"], str(self.snapshot_dir.resolve()))
        self.assertEqual(record["mapping"]["changed_positions"], list(range(5, 24)))
        self.assertTrue(verify_snapshot(Path(derived["directory"]), Snapshot(**record["derived"]["snapshot"])))
        self.assertNotEqual(record["derived"]["snapshot"]["sha256"], record["original"]["snapshot"]["sha256"])
        self.assertEqual((self.snapshot_dir / "input.bin").read_bytes(), DUP_CSV)

        state = {"ui_derived_input": derived}
        original = str(self.snapshot_dir.resolve())
        self.assertEqual(effective_snapshot(state, "sel01", original), derived["directory"])
        self.assertEqual(effective_snapshot(state, "sel02", original), original)

        diagnostic = create_profiling("sel01", derived["directory"], self.limits, confirmed=True)
        self.assertNotEqual(diagnostic.binding_id, dedupe.binding_id)
        status, summary = self.approve(diagnostic)
        self.assertEqual(status, "completed")
        self.assertEqual(summary["snapshot_sha256"], record["derived"]["snapshot"]["sha256"])
        result = load_profile_result(diagnostic)
        self.assertEqual([row["column"] for row in result["columns"]], dedupe_names(NAMES))

        # 생성 파일은 덮어쓰지 않는다. 같은 정비 결과를 다시 등록해도 새 이름으로만 만든다.
        again = register_derived_input(dedupe, root=self.root / "derived")
        self.assertNotEqual(again["directory"], derived["directory"])

    def test_derived_input_is_used_only_while_record_matches(self):
        dedupe = create_profiling("sel01", str(self.snapshot_dir.resolve()), self.limits, confirmed=True, header_dedupe=True)
        self.assertEqual(self.approve(dedupe)[0], "completed")
        derived = register_derived_input(dedupe, root=self.root / "derived")
        state = {"ui_derived_input": derived}
        original = str(self.snapshot_dir.resolve())
        self.assertEqual(effective_snapshot(state, "sel01", original), derived["directory"])

        record_path = Path(derived["record_path"])
        record = record_path.read_text(encoding="utf-8")
        tampered = json.loads(record)
        tampered["original"]["snapshot_directory"] = str(self.root / "other")
        record_path.write_text(json.dumps(tampered), encoding="utf-8")

        with self.assertRaises(ValueError):
            effective_snapshot(state, "sel01", original)

        record_path.write_text(record, encoding="utf-8")
        manifest = Path(derived["directory"]) / "snapshot.json"
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["sha256"] = "0" * 64
        manifest.write_text(json.dumps(value), encoding="utf-8")

        with self.assertRaises(ValueError):
            effective_snapshot(state, "sel01", original)

        record_path.unlink()

        with self.assertRaises(ValueError):
            effective_snapshot(state, "sel01", original)

    def test_registration_failure_is_not_retried_on_rerender(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from src.ui import profiling_view

        fake = MagicMock()
        fake.session_state = {}
        profiling = SimpleNamespace(name="input-dedupe-1", purpose="header_dedupe")

        with patch.object(profiling_view, "register_derived_input", side_effect=ValueError("fail")) as register:
            profiling_view._register(fake, profiling)

        register.assert_called_once()
        self.assertEqual(fake.session_state["ui_input_profile_register_failed"], "input-dedupe-1")
        self.assertNotIn("ui_derived_input", fake.session_state)
        self.assertIn("ui_input_profile_register_failed", profiling_view.SESSION_KEYS)

    def test_registration_failure_leaves_no_record(self):
        dedupe = create_profiling("sel01", str(self.snapshot_dir.resolve()), self.limits, confirmed=True, header_dedupe=True)
        self.assertEqual(self.approve(dedupe)[0], "completed")
        real = create_snapshot

        def tampered(source, destination, *, max_input_bytes):
            copy = self.root / "tampered.csv"
            copy.write_bytes(Path(source).read_bytes() + b"9" + LF)
            return real(copy, destination, max_input_bytes=max_input_bytes + 2)

        with patch("src.ui.profiling_service.create_snapshot", side_effect=tampered):
            with self.assertRaises(ValueError):
                register_derived_input(dedupe, root=self.root / "derived")

        self.assertEqual(list((self.root / "derived").glob("*.json")), [])


class DedupeAppTest(unittest.TestCase):
    """실제 화면: 옵션 없이 차단 → 자동 정비로 다시 준비 → 정비 승인 → 안내 문구 → 파생 입력 진단 승인 → 결과."""

    def test_screen_flow_from_block_to_derived_diagnostic(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = root / "dup.csv"
            source.write_bytes(DUP_CSV)
            snapshot = root / "snapshot"
            create_snapshot(source, snapshot, max_input_bytes=1024 * 1024)
            (root / "runs").mkdir()

            with patch("src.agents.pipeline_approval_cli._directory",
                       side_effect=lambda name, create_parent=False: root / "runs" / name), \
                    patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=_transport) as launch, \
                    patch("src.ui.profiling_service.SNAPSHOT_ROOT", root / "derived"), \
                    patch("src.ui.runtime_notice.server_permissions",
                          return_value={"elevated": True, "pid": 1, "python": "synthetic"}):
                app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=60).run()
                app.checkbox(key="ui_show_advanced_tools").check().run()
                app.selectbox(key="ui_workflow_mode").select("LLM 제안").run()
                app.selectbox(key="ui_source_mode_widget").set_value("기존 CSV snapshot").run()
                app.text_input(key="_widget_source_snapshot_path").set_value(str(snapshot))
                next(item for item in app.button if item.label == "로컬 데이터 선택 확인").click().run()
                app.toggle(key="ui_profiling_open").set_value(True).run()

                def prepare_and_approve():
                    next(item for item in app.checkbox if item.label.startswith("선택한 snapshot의 입력 진단")).check()
                    next(item for item in app.button if item.label == "입력 진단 작업 준비").click().run()
                    app.button(key="approval_flow_start").click().run()
                    next(item for item in app.checkbox if item.label.startswith("표시한 범위를 승인")).check()
                    next(item for item in app.button if item.label == "승인하고 실행").click().run()

                # 옵션을 끄면 기존처럼 차단한다.
                prepare_and_approve()
                self.assertFalse(app.exception, [item.value for item in app.exception])
                self.assertTrue(any("입력 진단이 차단되었습니다" in item.value for item in app.info))
                self.assertFalse(any("검증된 결과를 확인하세요" in item.value for item in app.info))
                self.assertFalse((root / "derived").exists())

                app.button(key="ui_profiling_dedupe_retry").click().run()
                self.assertTrue(app.checkbox(key="ui_profile_dedupe_option").value)
                prepare_and_approve()
                self.assertFalse(app.exception, [item.value for item in app.exception])
                self.assertTrue(any(item.value == MESSAGE for item in app.success))
                self.assertEqual((snapshot / "input.bin").read_bytes(), DUP_CSV)

                # 파생 입력으로 새 진단 작업·새 승인. 정비 옵션은 다시 묻지 않는다.
                self.assertFalse(any(item.key == "ui_profile_dedupe_option" for item in app.checkbox))
                prepare_and_approve()
                self.assertEqual(launch.call_count, 3)
                self.assertTrue(any(item.value == "입력 진단 결과" for item in app.subheader))
                self.assertTrue(any("입력 진단이 완료되었습니다" in item.value for item in app.info))
                self.assertEqual(len(list((root / "runs").iterdir())), 3)


if __name__ == "__main__":
    unittest.main()

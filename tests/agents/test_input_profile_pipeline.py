"""입력 진단 계획의 승인·worker·부모 검증 연결과 일반 경로 비완화를 합성 입력으로 검사한다.

OS transport(AppContainer·Job)만 모의 처리한다. worker 계산·파일 저장·부모 검증은 실제 코드를 실행한다.
"""

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import pkgutil
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import src.agents
from src.agents.eda_input_contract import bind_eda_input
from src.agents.input_profile_pipeline import (
    DiagnosticInputContract, InputProfileTool, bind_input_profile,
)
from src.agents.input_snapshot import create_snapshot
from src.agents.module_pipeline import ModulePipelineTool
from src.agents.module_pipeline_contract import POLICY_VERSION, bind_pipeline
from src.agents.module_pipeline_results import read_input_profile_result
from src.agents.module_pipeline_transport import CONTROL_FILES, ENTRY, WORKER_FILES
from src.agents.module_pipeline_worker import execute_request
from src.agents.pipeline_approval_cli import _build_binding, create, create_input_profile, next_step, record_approval, preview_approval, status
from src.agents.pipeline_binding_registry import build_pipeline_execution, get_pipeline_binder, registered_pipeline_adapters
from src.agents.reviewed_csv_pipeline_contract import bind_reviewed_csv_pipeline
from src.agents.reviewed_module_contract import demo_profile
from src.agents.snapshot_eda import SnapshotEdaTool
from src.agents.state import LocalStateStore
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.input_profile import input_profile_pipeline
from tests.agents.test_input_profile import CSV_BYTES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "configs/agents/profiles/tabular_example.json"
PIPELINE_PATH = PROJECT_ROOT / "configs/agents/pipelines/tabular_eda_example.json"
LIMITS = WorkerLimits(wall_seconds=60, memory_mib=2048, processes=2)
HEADER = ("int_col", "float_col", "pos_inf", "neg_inf", "both_inf", "all_inf", "text_col", "bool_col", "mixed", "edge")


def _transport(request, root, signal):
    """실제 worker를 같은 프로세스에서 실행하고 Job 종료 증거만 대체한다."""
    reply = execute_request(request, root)
    (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")

    return {"request_id": request.request_id, "profile_cleanup": True, "launch": {
        "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
        "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                      "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                      "configured_process_limit": request.limits.processes},
    }}


class InputProfilePipelineTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "synthetic.csv"
        source.write_bytes(CSV_BYTES)
        self.snapshot_dir = self.root / "snapshot"
        self.snapshot = create_snapshot(source, self.snapshot_dir, max_input_bytes=4096)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir()

    def run_directory(self, name, *, create_parent=False):
        return self.runs_dir / name

    def cli(self, function, *args, **kwargs):
        with patch("src.agents.pipeline_approval_cli._directory", side_effect=self.run_directory):
            return function(*args, **kwargs)

    def tool(self):
        self.count = getattr(self, "count", 0) + 1
        states = LocalStateStore(self.root / "tool_runs" / f"diagnostic{self.count}")
        states.create("synthetic_operator", policy_version=POLICY_VERSION)
        contract = DiagnosticInputContract(str(self.snapshot_dir.resolve()), self.snapshot, LIMITS)

        return InputProfileTool(states, contract, input_profile_pipeline())

    def staged(self, tool):
        request = tool._request(f"{self.count:032x}")
        return request, tool._stage(request)

    def rewrite(self, request, root, change):
        """worker 설정을 바꾸고 요청 config_id도 맞춰 worker 자체의 계획 검사만 남긴다."""
        settings = root / "input" / "eda.json"
        packet = json.loads(settings.read_text(encoding="utf-8"))
        change(packet)
        settings.write_text(json.dumps(packet, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        config_id = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        request = replace(request, config_id=config_id)
        (root / "input" / "request.json").write_text(request.to_json(), encoding="utf-8")
        return request

    def test_cli_approval_isolated_execution_and_parent_validation(self):
        """승인 전 worker 미실행, 승인 1회 소비 후 실행, 결과는 진단 검증기를 통과한다."""
        created = self.cli(create_input_profile, "diagnostic-01", self.snapshot_dir, LIMITS, confirmed=True)
        self.assertEqual(created["status"], "created")
        self.assertFalse(created["worker_started"])
        selection = json.loads((self.runs_dir / "diagnostic-01" / "selection.json").read_text(encoding="utf-8"))
        self.assertIsNone(selection["profile"])
        self.assertEqual(selection["pipeline"], input_profile_pipeline())

        with patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=_transport) as launch:
            self.assertEqual(self.cli(next_step, "diagnostic-01")["code"], "STEP_COMPLETED")
            self.assertEqual(self.cli(next_step, "diagnostic-01")["code"], "APPROVAL_REQUIRED")
            preview = self.cli(preview_approval, "diagnostic-01")
            summary = preview["summary"]
            self.assertEqual(summary["adapter_id"], "tabular_csv_diagnostic_v1")
            self.assertIs(summary["diagnostic_only"], True)
            self.assertIsNone(summary["roles"])
            self.assertIs(summary["llm_transfer"], False)
            self.assertEqual([item["module_id"] for item in summary["modules"]], ["input_profile"])
            launch.assert_not_called()

            approval = self.cli(record_approval, "diagnostic-01", preview, confirmed=True)
            launch.assert_not_called()
            result = self.cli(next_step, "diagnostic-01", approval["approval_id"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(self.cli(status, "diagnostic-01")["approval_status"], "consumed")

            # 같은 승인을 다시 써도 worker를 다시 시작하지 않는다.
            self.cli(next_step, "diagnostic-01", approval["approval_id"])
            self.assertEqual(launch.call_count, 1)

        outputs = list((self.runs_dir / "diagnostic-01").rglob("input_profile.csv"))
        self.assertEqual(len(outputs), 1)
        pipeline = outputs[0].parent.parent
        self.assertEqual({path.name for path in pipeline.iterdir()}, {"pipeline_result.json", "input_profile"})
        self.assertNotIn("CANARY", outputs[0].read_text(encoding="utf-8-sig"))

    def test_approval_invalidated_by_code_or_snapshot_change(self):
        self.cli(create_input_profile, "diagnostic-02", self.snapshot_dir, LIMITS, confirmed=True)

        with patch("src.agents.input_profile_pipeline.code_manifest", return_value={"changed": "code"}):
            with self.assertRaises(ValueError):
                self.cli(status, "diagnostic-02")

        (self.snapshot_dir / "input.bin").write_bytes(CSV_BYTES + b"9,9,9,9,9,9,x,True,9,9\n")

        with self.assertRaises(ValueError):
            self.cli(status, "diagnostic-02")

        with self.assertRaises(ValueError):
            self.cli(create_input_profile, "diagnostic-03", self.snapshot_dir, LIMITS)

    def test_general_path_checks_are_not_relaxed(self):
        general = json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
        diagnostic = input_profile_pipeline()

        # 진단 계획에 프로파일을 붙이거나 일반 바인더·결합기에 넣으면 기존 검사가 거부한다.
        with self.assertRaises(ValueError):
            bind_reviewed_csv_pipeline(diagnostic, self.snapshot_dir, self.snapshot, demo_profile(), LIMITS)

        with self.assertRaises(ValueError):
            bind_pipeline(diagnostic, demo_profile())

        # 일반 계획·다른 단계를 섞은 계획·설정을 바꾼 계획은 진단 바인더가 거부한다.
        mixed = input_profile_pipeline()
        mixed["steps"].append(dict(general["steps"][0], step_id="quality"))
        changed = input_profile_pipeline()
        changed["steps"][0]["settings"] = {}
        renamed = input_profile_pipeline()
        renamed["pipeline_id"] = "other"

        for config in (general, mixed, changed, renamed):
            with self.assertRaises(ValueError):
                bind_input_profile(config, self.snapshot_dir, self.snapshot, LIMITS)

        # 선택 기록 경로: 일반 계획에 profile null, 진단 계획에 프로파일
        base = {"snapshot_directory": str(self.snapshot_dir.resolve()), "snapshot": self.snapshot.__dict__, "limits": LIMITS.__dict__}

        with self.assertRaises(ValueError):
            _build_binding(dict(base, profile=None, pipeline=general))

        with self.assertRaises(ValueError):
            _build_binding(dict(base, profile=demo_profile().to_dict(), pipeline=diagnostic))

        # 진단 adapter ID는 프로파일 바인더 경로로 열리지 않고, 실행기는 자기 바인딩 타입만 받는다.
        self.assertEqual(registered_pipeline_adapters(), ("tabular_csv_v1",))

        with self.assertRaises(ValueError):
            get_pipeline_binder("tabular_csv_diagnostic_v1")

        states = LocalStateStore(self.root / "mixed_runs" / "a")
        states.create("synthetic_operator", policy_version="x")
        reviewed = bind_reviewed_csv_pipeline(general, self.snapshot_dir, self.snapshot, demo_profile(), LIMITS)
        diagnostic_binding = bind_input_profile(diagnostic, self.snapshot_dir, self.snapshot, LIMITS)

        with self.assertRaises(TypeError):
            build_pipeline_execution("tabular_csv_diagnostic_v1", states, reviewed)

        with self.assertRaises(TypeError):
            build_pipeline_execution("tabular_csv_v1", states, diagnostic_binding)

        # 진단 입력 계약은 일반 EDA 도구에, 일반 계약은 진단 도구에 넣을 수 없다.
        with self.assertRaises(TypeError):
            SnapshotEdaTool(states, diagnostic_binding.input_contract)

        with self.assertRaises(TypeError):
            ModulePipelineTool(states, diagnostic_binding.input_contract, general)

        with self.assertRaises(TypeError):
            InputProfileTool(states, bind_eda_input(self.snapshot_dir, self.snapshot, demo_profile(), LIMITS), diagnostic)

        # 일반 CLI create는 여전히 프로파일 파일을 요구하고 기존 결합을 거친다.
        self.assertEqual(self.cli(create, "general-01", self.snapshot_dir, PROFILE_PATH, PIPELINE_PATH, LIMITS, confirmed=True)["status"], "created")

    def test_worker_compares_full_plan_and_recomputes_signature(self):
        """부모 검사를 우회한 설정도 worker가 복사된 고정 계획과 직접 대조해 거부한다."""
        general = json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
        cases = {
            "general_plan_without_profile": lambda packet: packet.update(pipeline=general),
            "extra_step": lambda packet: packet["pipeline"]["steps"].append(dict(packet["pipeline"]["steps"][0], step_id="again")),
            "changed_id": lambda packet: packet["pipeline"].update(pipeline_id="other"),
            "settings_added": lambda packet: packet["pipeline"]["steps"][0].update(settings={}),
            "wrong_signature": lambda packet: packet.update(pipeline_signature="0" * 64),
            "profile_with_diagnostic": lambda packet: packet.update(profile=demo_profile().to_dict()),
        }

        for name, change in cases.items():
            with self.subTest(name):
                tool = self.tool()
                request, root = self.staged(tool)
                request = self.rewrite(request, root, change)
                self.assertEqual(execute_request(request, root).status, "blocked_input")
                self.assertFalse((root / "output" / "result" / "pipeline").exists())

        tool = self.tool()
        request, root = self.staged(tool)
        self.assertEqual(execute_request(request, root).status, "succeeded")
        self.assertIsNotNone(read_input_profile_result(root / "output" / "result" / "pipeline", tool.module_plan, HEADER))

    def test_parent_rejects_tampered_results(self):
        tool = self.tool()
        request, root = self.staged(tool)
        self.assertEqual(execute_request(request, root).status, "succeeded")
        original = root / "output" / "result" / "pipeline"
        table = original / "input_profile" / "input_profile.csv"
        text = table.read_text(encoding="utf-8-sig")

        def edited(old, new):
            self.assertEqual(text.count(old), 1, old)
            return lambda directory: (directory / "input_profile" / "input_profile.csv").write_text(text.replace(old, new), encoding="utf-8-sig")

        def summary(directory):
            path = directory / "input_profile" / "input_summary.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["row_count"] = 9
            path.write_text(json.dumps(value), encoding="utf-8")

        cases = {
            # pos_inf 행의 finite_count 7→6: 유한값+무한대 = 비결측 항등식 위반
            "finite_identity": edited("pos_inf,numeric,0,0.0,8,,1,0,7,", "pos_inf,numeric,0,0.0,8,,1,0,6,"),
            # 비수치 컬럼에 통계 기입
            "text_statistics": edited("bool_col,boolean,0,0.0,2,,,,,", "bool_col,boolean,0,0.0,2,,,,1,"),
            # 분위수 순서 위반(int_col q25 2.75 → 5.0)
            "quantile_order": edited(",2.75,4.5,6.25,50.0,", ",5.0,4.5,6.25,50.0,"),
            # IQR 경계가 공식과 다름
            "iqr_bound": edited(",-2.5,11.5,1", ",-3.0,11.5,1"),
            "extra_file": lambda directory: (directory / "input_profile" / "extra.csv").write_text("x", encoding="utf-8"),
            "summary_rows": summary,
            # 장악된 worker가 컬럼명 칸에 원본 값을 넣거나 행 순서를 바꾸는 경우
            "column_renamed": edited("\nint_col,numeric,", "\nSECRET_LOT_A123,numeric,"),
            "rows_reordered": lambda directory: (directory / "input_profile" / "input_profile.csv").write_text(
                "\n".join([text.splitlines()[0], *reversed(text.splitlines()[1:])]) + "\n", encoding="utf-8-sig"),
            "negative_std": edited(",4.0,1.8708286933869707,", ",4.0,-1.8708286933869707,"),
            "padded_number": edited("float_col,numeric,2,0.25,", "float_col,numeric,2, 0.25,"),
        }

        for name, change in cases.items():
            with self.subTest(name):
                directory = self.root / ("tampered_" + name)
                shutil.copytree(original, directory)
                change(directory)
                self.assertIsNone(read_input_profile_result(directory, tool.module_plan, HEADER))

        self.assertIsNotNone(read_input_profile_result(original, tool.module_plan, HEADER))
        self.assertIsNotNone(tool._read_result(original))

        # 부모 대조 기준은 worker 결과가 아니라 승인된 snapshot 헤더다.
        self.assertIsNone(read_input_profile_result(original, tool.module_plan, HEADER[::-1]))

    def test_copied_worker_runs_diagnostic_with_allowlisted_files_only(self):
        """고정 계획 상수·진단 계산이 복사 허용 목록만으로 별도 프로세스에서 import된다."""
        self.assertIn("src/preprocessing/input_profile.py", WORKER_FILES)
        self.assertIn("src/agents/input_profile_pipeline.py", CONTROL_FILES)
        self.assertNotIn("src/agents/input_profile_pipeline.py", WORKER_FILES)
        tool = self.tool()
        request, root = self.staged(tool)
        tools = root / "tools"

        for name in WORKER_FILES:
            target = tools / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / name, target)

        shutil.copyfile(PROJECT_ROOT / ENTRY, tools / "worker.py")
        script = (
            "import sys;sys.path[:0]=sys.argv[1:3];from pathlib import Path;"
            "from src.agents.worker_contract import WorkerRequest;import worker;"
            "root=Path(sys.argv[3]);r=WorkerRequest.from_json((root/'input/request.json').read_text());"
            "assert worker.execute_request(r,root).status=='succeeded';"
            "assert 'src.agents.input_profile_pipeline' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script, str(tools), str(Path(sys.prefix) / "Lib/site-packages"), str(root)],
            cwd=self.root, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIsNotNone(read_input_profile_result(root / "output" / "result" / "pipeline", tool.module_plan, HEADER))

    def test_approval_boundary_files_invalidate_existing_selection(self):
        """승인 선택·경로 선택·실행 직전 대조 코드가 바뀌면 같은 승인 run을 열지 않는다."""
        import src.agents.module_pipeline_transport as transport

        boundary = (
            "src/agents/pipeline_approval_cli.py", "src/agents/pipeline_binding_registry.py",
            "src/agents/pipeline_approval.py", "src/agents/pipeline_csv_execution.py",
        )
        self.assertTrue(set(boundary).issubset(CONTROL_FILES))
        self.assertFalse(set(boundary) & set(WORKER_FILES))
        self.cli(create_input_profile, "diagnostic-04", self.snapshot_dir, LIMITS, confirmed=True)
        original = transport._fingerprint

        for name in boundary:
            with self.subTest(name):
                def changed(path, target=name):
                    record = dict(original(path))

                    if Path(path).as_posix().endswith(target):
                        record["sha256"] = "0" * 64

                    return record

                with patch.object(transport, "_fingerprint", side_effect=changed):
                    with self.assertRaises(ValueError):
                        self.cli(status, "diagnostic-04")

        self.assertEqual(self.cli(status, "diagnostic-04")["stage"], "preflight")

    def test_diagnostic_files_enter_only_registered_tabular_consumers(self):
        """공유 진단 의존성은 승인된 표형 소비자에만 결합하고 WM 등으로 퍼지지 않는다."""
        new = {"src/preprocessing/input_profile.py", "src/agents/input_profile_pipeline.py"}
        members = []

        for module in pkgutil.iter_modules(src.agents.__path__):
            if module.name.endswith("_transport"):
                transport = importlib.import_module("src.agents." + module.name)

                if new & set(getattr(transport, "CONTROL_FILES", ())):
                    members.append(module.name)

        self.assertEqual(members, ["module_pipeline_transport", "partition_export_transport", "snapshot_split_transport", "tabular_input_transport"])


if __name__ == "__main__":
    unittest.main()

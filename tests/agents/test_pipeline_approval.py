"""형식 중립 승인 흐름과 첫 CSV 바인더의 경계를 확인한다."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from src.agents.input_snapshot import create_snapshot
from src.agents.pipeline_approval_cli import approve, create, next_step, status
from src.agents.pipeline_binding_registry import (
    get_pipeline_binder,
    registered_pipeline_adapters,
)
from src.agents.reviewed_module_contract import demo_profile
from src.agents.synthetic_eda_data import CSV_BYTES
from src.agents.worker_contract import WorkerLimits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "configs/agents/profiles/tabular_example.json"
PIPELINE_PATH = PROJECT_ROOT / "configs/agents/pipelines/tabular_eda_example.json"


class PipelineApprovalTest(unittest.TestCase):
    """CSV는 첫 어댑터로 쓰되 승인 흐름은 형식 중립적으로 유지한다."""

    def setUp(self):
        # 제한된 Windows 실행 환경에서 외부 TEMP 권한에 의존하지 않도록
        # 임시 fixture를 워크스페이스 루트 아래에 생성한다.
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

        # 입력은 저장소의 SECOM/WM 원본이 아닌 고정된 합성 CSV만 사용한다.
        source = self.root / "synthetic.csv"
        source.write_bytes(CSV_BYTES)
        self.snapshot_dir = self.root / "snapshot"
        create_snapshot(source, self.snapshot_dir, max_input_bytes=4096)

        # 같은 형식의 CLI 흐름을 매 테스트마다 별도 run 폴더에서 격리한다.
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir()

    def run_directory(self, name, *, create_parent=False):
        """CLI의 기본 로그 경로 대신 임시 작업 폴더를 사용한다."""
        parent = self.runs_dir

        if create_parent:
            parent.mkdir(exist_ok=True)

        return parent / name

    def cli_call(self, function, *args, **kwargs):
        """각 CLI 호출이 동일한 격리 임시 경로를 사용하도록 연결한다."""
        with patch(
            "src.agents.pipeline_approval_cli._directory",
            side_effect=self.run_directory,
        ):
            return function(*args, **kwargs)

    def create_run(self, name="pipeline-test01"):
        """고정된 프로파일·모듈 계획으로 승인 run을 준비한다."""
        return self.cli_call(
            create,
            name,
            self.snapshot_dir,
            PROFILE_PATH,
            PIPELINE_PATH,
            WorkerLimits(wall_seconds=60, memory_mib=2048, processes=2),
            confirmed=True,
        )

    def test_registry_rejects_unregistered_input_format(self):
        """미등록 파일 형식을 CSV 등 다른 어댑터로 추정하지 않는다."""
        self.assertEqual(registered_pipeline_adapters(), ("tabular_csv_v1",))
        self.assertTrue(callable(get_pipeline_binder("tabular_csv_v1")))

        with self.assertRaises(ValueError):
            get_pipeline_binder("parquet_v1")

    def test_binding_uses_format_neutral_approval_stage(self):
        """CSV 바인더도 공통 run_pipeline 승인 단계와 출처 계약을 제공한다."""
        from src.agents.input_snapshot import Snapshot

        snapshot_info = json.loads((self.snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
        binding = get_pipeline_binder("tabular_csv_v1")(
            json.loads(PIPELINE_PATH.read_text(encoding="utf-8")),
            self.snapshot_dir,
        Snapshot(
            size_bytes=snapshot_info["size_bytes"],
            sha256=snapshot_info["sha256"],
            max_input_bytes=snapshot_info["max_input_bytes"],
        ),
            demo_profile(),
            WorkerLimits(wall_seconds=60, memory_mib=2048, processes=2),
        )

        plan = binding.execution_plan()
        summary = binding.approval_summary()

        self.assertEqual(plan.steps[-1].stage, "run_pipeline")
        self.assertEqual(binding.source_id, plan.source_id)
        self.assertEqual(summary["adapter_id"], "tabular_csv_v1")
        self.assertEqual(summary["source_id"], binding.source_id)
        self.assertEqual(binding.to_dict()["schema_version"], binding.policy_version)

    def test_approval_is_consumed_only_when_worker_stage_runs(self):
        """승인 저장 후에도 worker를 호출하거나 승인을 소비하지 않는다."""
        created = self.create_run()
        self.assertEqual(created["status"], "created")
        self.assertFalse(created["worker_started"])

        from src.agents.pipeline_approval_cli import _build_binding, _json_value
        selection = json.loads(
            (self.run_directory("pipeline-test01") / "selection.json").read_text(encoding="utf-8")
        )
        self.assertEqual(_json_value(_build_binding(selection).to_dict()), selection["binding"])

        first = self.cli_call(next_step, "pipeline-test01")
        second = self.cli_call(next_step, "pipeline-test01")
        self.assertEqual(first["code"], "STEP_COMPLETED")
        self.assertEqual(second["code"], "APPROVAL_REQUIRED")

        with (
            patch("src.agents.pipeline_approval_cli.os.isatty", return_value=True),
            patch("src.agents.pipeline_approval_cli.secrets.token_hex", return_value="a1b2c3d4"),
            patch("builtins.input", return_value="PIPELINE-A1B2C3D4"),
            patch("builtins.print"),
            patch("src.agents.pipeline_approval_cli._directory", side_effect=self.run_directory),
        ):
            approved = approve("pipeline-test01")

        self.assertEqual(approved["status"], "approval_recorded")
        self.assertFalse(approved["worker_started"])

        result = self.cli_call(next_step, "pipeline-test01")
        current = self.cli_call(status, "pipeline-test01")
        self.assertEqual(result["code"], "APPROVAL_REQUIRED")
        self.assertEqual(current["approval_status"], "pending")

        from src.agents.contracts import ToolResult

        calls = []

        class FakeExecution:
            worker_started = False

            def execute(self, output, arguments):
                calls.append((output, arguments))
                self.worker_started = True
                (output / "worker-result.json").write_text("{}", encoding="utf-8")
                return ToolResult("passed")

            def validate(self, output, arguments):
                return (output / "worker-result.json").is_file()

        with patch(
            "src.agents.pipeline_approval.build_pipeline_execution",
            return_value=FakeExecution(),
        ):
            executed = self.cli_call(
                next_step, "pipeline-test01", approved["approval_id"],
            )

        self.assertEqual(executed["code"], "STEP_COMPLETED")
        self.assertTrue(executed["worker_started"])
        self.assertEqual(len(calls), 1)

        current = self.cli_call(status, "pipeline-test01")
        self.assertEqual(current["status"], "completed")
        self.assertEqual(current["approval_status"], "consumed")

    def test_csv_execution_binding_matches_json_roundtrip(self):
        """JSON 복원 시 tuple이 list로 바뀌어도 같은 바인딩으로 인식한다."""
        self.create_run("pipeline-json-roundtrip01")

        from src.agents.pipeline_approval_cli import _open

        _directory_path, _selection, run = self.cli_call(
            _open,
            "pipeline-json-roundtrip01",
        )
        run.pipeline_execution._check_binding(run.arguments)

    def test_modified_pipeline_invalidates_saved_binding(self):
        """run 생성 후 모듈 설정이 바뀌면 저장된 계약과 달라져 거부한다."""
        self.create_run("pipeline-change01")
        directory = self.run_directory("pipeline-change01")
        selection_path = directory / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["pipeline"]["pipeline_id"] = "modified_pipeline"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.cli_call(status, "pipeline-change01")


if __name__ == "__main__":
    unittest.main()

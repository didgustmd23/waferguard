"""대화 CLI→실제 제안 저장·복원→미승인 작업 생성의 합성 통합 검사."""

from contextlib import closing, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.agents import langchain_local, llm_dialogue_cli, llm_proposal_cli
from src.agents import llm_execution_candidate as candidate
from src.agents import llm_proposal_store as store
from src.agents.input_snapshot import create_snapshot
from src.agents.synthetic_eda_data import CSV_BYTES
from src.agents.worker_contract import WorkerLimits
from tests.agents.test_llm_dialogue import clarification
from tests.agents.test_llm_proposal import valid_content


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DialogueHandoffTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.saved = self.root / "proposals"
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.drafts = self.root / "candidates"
        self.limits = WorkerLimits(60, 2048, 2)
        source = self.root / "synthetic.csv"
        source.write_bytes(CSV_BYTES)
        self.snapshot = self.root / "snapshot"
        create_snapshot(source, self.snapshot, max_input_bytes=4096)
        self.profile = self.root / "profile.json"
        self.profile.write_bytes((PROJECT_ROOT / "configs/agents/profiles/tabular_example.json").read_bytes())

    def dialogue_and_save(self, confirmation="SAVE"):
        arguments = ["dialogue", "--example", "clarify", "--save-name", "dialogue01"]
        answers = ["SEND", "품질 검사 후 결측률을 확인해 주세요.", "SEND", confirmation]

        with patch("sys.argv", arguments), redirect_stdout(StringIO()) as output:
            with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True):
                with patch("builtins.input", side_effect=answers):
                    with patch.object(langchain_local, "request_local_proposal",
                                      side_effect=[clarification(), valid_content()]) as model:
                        with patch.object(llm_proposal_cli, "save_proposal",
                                          side_effect=lambda name, result: store.save_proposal(name, result, root=self.saved)):
                            self.assertEqual(llm_dialogue_cli.main(), 0)

        self.assertEqual(model.call_count, 2)
        return output.getvalue()

    def preview(self):
        with patch.object(candidate, "load_proposal",
                          side_effect=lambda name: store.load_proposal(name, root=self.saved)):
            return candidate.preview_candidate("dialogue01", self.snapshot, self.profile, self.limits)

    def prepare(self, selection_id):
        with patch.object(candidate, "load_proposal",
                          side_effect=lambda name: store.load_proposal(name, root=self.saved)):
            with patch("src.agents.pipeline_approval_cli._directory",
                       side_effect=lambda name, **kwargs: self.runs / name):
                return candidate.create_candidate(
                    "dialoguerun01", "dialogue01", self.snapshot, self.profile, self.limits,
                    expected_selection_id=selection_id, confirmed=True, root=self.drafts,
                )

    def test_dialogue_to_existing_run_preserves_signature_and_has_no_approval(self):
        output = self.dialogue_and_save()
        self.assertIn("proposal_saved", output)

        # 후속 단계에서는 모델을 호출하지 않는다. 저장·복원·준비는 기존 함수를 사용한다.
        with patch.object(langchain_local, "request_local_proposal", side_effect=AssertionError("모델 재호출 금지")):
            restored = store.load_proposal("dialogue01", root=self.saved)
            preview = self.preview()
            result = self.prepare(preview["selection_id"])

        self.assertFalse(restored["model_called"])
        self.assertEqual(result["status"], "execution_candidate_created")
        self.assertFalse(result["execution_approval_issued"])
        self.assertFalse(result["worker_started"])
        run = self.runs / "dialoguerun01"
        state = json.loads((run / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "created")
        self.assertFalse((run / "approval.json").exists())

        with closing(
            sqlite3.connect((run / "approvals.sqlite3").as_uri() + "?mode=ro", uri=True)
        ) as db, db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM approvals").fetchone()[0], 0)

        envelope = json.loads((self.saved / "dialogue01.json").read_text(encoding="utf-8"))
        self.assertEqual(envelope["plan_signature"], restored["plan_signature"])
        self.assertEqual(set(envelope), {
            "schema_version", "proposal", "proposal_sha256", "plan_signature",
            "proposal_only", "execution_approved", "worker_started",
        })
        self.assertFalse(self.runs.parent.joinpath("workers").exists())

    def test_save_decline_does_not_create_stored_proposal_or_run(self):
        self.dialogue_and_save("취소")
        self.assertFalse(self.saved.exists())
        self.assertEqual(list(self.runs.iterdir()), [])
        self.assertFalse(self.drafts.exists())

    def test_stored_proposal_tampering_blocks_preparation(self):
        self.dialogue_and_save()
        preview = self.preview()
        path = self.saved / "dialogue01.json"
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["proposal"]["pipeline"]["pipeline_id"] = "changed"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.prepare(preview["selection_id"])

        self.assertEqual(list(self.runs.iterdir()), [])
        self.assertFalse(self.drafts.exists())

    def test_review_stops_dialogue_plan_before_analysis_and_never_reruns(self):
        """모델 제안도 품질 중단을 우회하지 않는다. OS 운송만 합성 실행으로 대체한다."""
        from src.agents.pipeline_approval_cli import _open
        from src.agents.module_pipeline_worker import execute_request
        from src.agents.module_pipeline_results import read_pipeline_result

        self.dialogue_and_save()
        self.profile.write_bytes(
            (PROJECT_ROOT / "configs/agents/profiles/dialogue_review_example.json").read_bytes()
        )
        preview = self.preview()
        self.assertEqual(preview["effective_execution"]["quality_rules"]["missing_ratio_threshold"], 0.0)
        self.prepare(preview["selection_id"])

        def synthetic_transport(request, root, signal):
            # 테스트 fixture만 직접 계산한다. 실제 AppContainer·ACL·프로세스는 시작하지 않는다.
            reply = execute_request(request, root)
            (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
            return {"request_id": request.request_id, "profile_cleanup": True, "launch": {
                "status": "exited", "exit_code": 0,
                "resources": {"status": "succeeded", "cleanup_confirmed": True,
                              "active_processes": 0,
                              "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                              "configured_process_limit": request.limits.processes},
            }}

        with patch("src.agents.pipeline_approval_cli._directory", side_effect=lambda name: self.runs / name):
            _directory, _selection, run = _open("dialoguerun01")

        self.assertEqual(run.next().code, "STEP_COMPLETED")
        self.assertEqual(run.next().code, "APPROVAL_REQUIRED")
        scope = run.executor.scope_for("run_pipeline")
        approval = run.approvals.record_confirmed_approval(scope)

        with patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=synthetic_transport) as transport:
            with patch("src.preprocessing.eda.analyze_missing", side_effect=AssertionError("품질 검토 후 분석 금지")):
                result = run.next(approval_id=approval.approval_id)
                self.assertEqual(result.status, "blocked_input")
                self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
                task = run.approvals.get_task(scope)
                output = run.executor._output(task)
                self.assertFalse((output / "AGENT_COMPLETE.json").exists())
                self.assertFalse((output / "pipeline/missing").exists())
                report = read_pipeline_result(output / "pipeline", run.binding.plan,
                                              run.binding.input_contract.profile)
                self.assertEqual(report["status"], "review_required")
                self.assertEqual(report["completed_steps"], ["quality"])
                self.assertEqual(run.approvals.get(approval.approval_id).status, "consumed")
                self.assertEqual(run.states.load().status, "blocked_input")
                run.next()
                transport.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""입력 해석 대기·새 A 개정·중복 제출 방지를 실제 입력 도구와 합성 어댑터로 검사한다."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.agents import task_adapters, task_consent, task_orchestrator as orchestrator, task_store
from src.agents import tabular_input_flow as flow
from src.agents import tabular_input_service as service
from src.agents.llm_config import load_local_config
from tests.agents import test_tabular_input as inputs


class _Adapter:
    """작업 3 입력 준비만 실제로 수행한다. 이후 진단·권장안은 아직 미구현인 다음 단계의 대역이다."""
    KIND = "tabular_input_test"

    @staticmethod
    def consent_a_content(record):
        return flow.consent_content(record)

    @staticmethod
    def diagnose(record, directory, context):
        return _Adapter.finish(flow.prepare_input(record, directory, context))

    @staticmethod
    def check_input_selection(record, selection, directory):
        return flow.check_selection(record, selection, directory)

    @staticmethod
    def input_consent_content(record, selection):
        return flow.consent_content(record, selection)

    @staticmethod
    def resume_input(record, directory, context, selection):
        return _Adapter.finish(flow.prepare_input(record, directory, context, selection=selection))

    @staticmethod
    def check_source(record):
        flow.check_source(record)

    @staticmethod
    def finish(record):
        if record["phase"] != "awaiting_input":
            flow.check_source(record)
            record["diagnostic"] = {"status": "completed", "run_name": "synthetic_future_diagnostic"}
        return record


class TaskInputInterpretationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input.csv"
        self.source.write_bytes(b"x,x\n1,2\n3,4\n")
        self.store = self.root / "tasks"
        self.models = []
        self.start_patch(patch.dict(task_adapters.ADAPTERS, {_Adapter.KIND: _Adapter}))
        self.start_patch(patch.object(task_adapters, "select", return_value=_Adapter.KIND))
        self.start_patch(patch.object(service, "run_input_worker", side_effect=inputs.transport))
        self.start_patch(patch.object(orchestrator, "_recommend", side_effect=self.recommend))
        self.new_task()

    def new_task(self):
        self.record = orchestrator.start_task("합성 입력", self.source, {"wall_seconds": 60, "memory_mib": 2048, "processes": 2},
                                             asdict(load_local_config()), root=self.store)

    def start_patch(self, item):
        result = item.start()
        self.addCleanup(item.stop)
        return result

    def recommend(self, record, adapter, requester, root):
        self.models.append(record["task_id"])
        record["phase"] = "awaiting_review"
        return task_store.save_task(record, expected_revision=record["revision"], root=root)

    def diagnose(self):
        self.record = orchestrator.diagnose(self.record["task_id"], root=self.store)
        return self.record

    def resume(self, selection=None, **options):
        return orchestrator.resume_input(self.record["task_id"], selection or {"header_repair": "number_duplicates"},
                                         expected_revision=options.get("revision", self.record["revision"]),
                                         expected_inspection_id=options.get("inspection_id", self.record["input_selection"]["inspection_id"]),
                                         root=self.store)

    def test_waiting_stops_model_and_materialize_until_new_selection(self):
        record = self.diagnose()
        self.assertEqual(record["phase"], "awaiting_input", record.get("problem"))
        self.assertEqual(self.models, [])
        self.assertNotIn("C", record["consents"])
        self.assertNotIn("D", record["consents"])
        self.assertIsNone(record["input_selection"].get("materialize_run"))
        old_a = record["consents"]["A"]["consent_id"]
        record = self.resume()
        self.assertEqual(record["phase"], "awaiting_review", record.get("problem"))
        self.assertEqual(len(self.models), 1)
        self.assertNotEqual(record["consents"]["A"]["consent_id"], old_a)
        self.assertTrue(record["input_selection"]["history"])
        self.assertEqual(record["input_selection"]["status"], "materialized")

    def test_normal_default_uses_one_user_consent_for_two_distinct_executions(self):
        self.source.write_bytes(b"x,label\n1,a\n2,b\n")
        self.new_task()
        record = self.diagnose()
        self.assertEqual(record["phase"], "awaiting_review", record.get("problem"))
        selected = record["input_selection"]
        self.assertNotEqual(selected["inspect_run"], selected["materialize_run"])
        self.assertEqual(selected["attempt"], 1)
        self.assertEqual(selected["history"], [])
        self.assertEqual(set(record["consents"]), {"A"})

    def test_invalid_stale_or_unchanged_selection_creates_no_new_consent_or_run(self):
        self.diagnose()
        before = self.record["consents"]["A"]["consent_id"]
        for selection, options in (({"converters": "os.system"}, {}), ({"header_repair": "reject"}, {}),
                                   ({"header_repair": "number_duplicates"}, {"revision": 0}),
                                   ({"header_repair": "number_duplicates"}, {"inspection_id": "a" * 64})):
            with self.subTest(selection=selection, options=options), self.assertRaises(ValueError):
                self.resume(selection, **options)
            current = task_store.load_task(self.record["task_id"], root=self.store)
            self.assertEqual(current["consents"]["A"]["consent_id"], before)
            self.assertEqual(current["input_selection"]["attempt"], 1)

    def test_snapshot_or_committed_inspection_change_blocks_before_new_a(self):
        self.diagnose()
        before = self.record["consents"]["A"]["consent_id"]
        Path(self.record["snapshot"]["directory"], "input.bin").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.resume()
        current = task_store.load_task(self.record["task_id"], root=self.store)
        self.assertEqual(current["consents"]["A"]["consent_id"], before)
        self.assertEqual(self.models, [])

    def test_expired_consent_and_other_user_cannot_resume(self):
        self.diagnose()
        record = task_store.load_task(self.record["task_id"], root=self.store)
        record["consents"]["A"]["expires_at"] = "2000-01-01T00:00:00+00:00"
        self.record = task_store.save_task(record, expected_revision=record["revision"], root=self.store)
        with self.assertRaises(ValueError):
            self.resume()
        with patch.object(task_store, "_current_actor", return_value="other_user"), self.assertRaises(ValueError):
            self.resume()

    def test_changed_committed_inspection_is_rejected_before_new_a_or_run(self):
        self.diagnose()
        before = self.record["consents"]["A"]["consent_id"]
        directory = task_store.task_directory(self.record["task_id"], root=self.store)
        item = service.TabularInputService("inspect", root=directory / "input_runs")
        folder, _ = item.verified(self.record["input_selection"]["inspect_run"])
        (folder / "inspection.json").write_bytes(b"{}")
        with self.assertRaises(ValueError):
            self.resume()
        record = task_store.load_task(self.record["task_id"], root=self.store)
        self.assertEqual(record["consents"]["A"]["consent_id"], before)
        self.assertEqual(record["input_selection"]["attempt"], 1)
        self.assertFalse(list((directory / "input_runs").rglob("*-i2")))

    def test_materialize_save_failure_never_automatically_reexecutes(self):
        self.source.write_bytes(b"x,label\n1,a\n2,b\n")
        self.new_task()
        real_save = task_store.save_task
        failed = False

        def failing_save(record, **options):
            nonlocal failed
            if not failed and (record.get("input_selection") or {}).get("status") == "materialized":
                failed = True
                raise OSError("합성 저장 실패")
            return real_save(record, **options)

        with patch.object(task_store, "save_task", side_effect=failing_save):
            record = self.diagnose()
        self.assertTrue(failed)
        self.assertEqual(record["phase"], "needs_attention")
        self.assertEqual(self.models, [])
        with self.assertRaises(ValueError):
            orchestrator.diagnose(record["task_id"], root=self.store)
        materialize = service.TabularInputService("materialize", root=task_store.task_directory(record["task_id"], root=self.store) / "input_runs")
        self.assertEqual(materialize.status(record["input_selection"]["materialize_run"])["status"], "completed")

    def test_large_input_selection_uses_existing_parts_integrity_checks(self):
        self.source.write_text(",".join(f"c{i}" for i in range(600)) + "\n" + ",".join("1" for _ in range(600)) + "\n", encoding="utf-8")
        self.new_task()
        record = self.diagnose()
        self.assertEqual(record["phase"], "awaiting_review", record.get("problem"))
        directory = task_store.task_directory(record["task_id"], root=self.store)
        stored = json.loads((directory / "task.json").read_text(encoding="utf-8"))
        self.assertIn("$part", stored["input_selection"])
        self.assertEqual(len(task_store.load_task(record["task_id"], root=self.store)["input_selection"]["inspection"]["columns"]), 600)

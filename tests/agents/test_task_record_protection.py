"""공통 작업 기록(task_store)의 보호 생성·잠금 구간 보유·이전 작업 처리, 위임 근거 기록, CSV 후속 출처 소비 경계를 확인한다.

권한은 임시 폴더에 실제 icacls·보안 조회로 설정·확인한다. AppContainer 실행·로컬 모델은 기존 fixture처럼 모의한다.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.agents import audit, task_consent, task_store
from src.agents import protected_store as store
from src.agents.protected_store import StoreUnavailable


AUTHENTICATED_USERS = "S-1-5-11"


def icacls(path, *arguments):
    executable = Path(os.environ["SystemRoot"]) / "System32" / "icacls.exe"
    return subprocess.run([str(executable), str(path), *arguments, "/Q"], capture_output=True, check=False).returncode


class TaskStoreProtectionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "tasks"
        self.record = task_store.create_task("goal", "source.csv", "tabular_v2", root=self.root)
        self.task_id = self.record["task_id"]
        self.directory = self.root / self.task_id

    def lock(self):
        with task_store.task_lock(self.task_id, root=self.root) as directory:
            return directory

    def assert_unprotected(self):
        with self.assertRaises(task_store.TaskRecordError) as caught:
            self.lock()
        self.assertEqual(caught.exception.code, "RECORD_UNPROTECTED")

    def test_new_task_folder_and_record_are_protected_from_creation(self):
        trusted = set(store.trusted_principals())
        info = store._security(self.directory)
        self.assertTrue(info["protected"])
        self.assertFalse(any(flags & 0x10 for _, flags, _, _ in info["dacl"]))
        self.assertEqual({sid for kind, _, _, sid in info["dacl"] if kind == 0}, trusted)
        self.assertTrue(store.check_object(self.directory / "task.json", store.trusted_principals()))
        # 분리 저장(parts)·저장 실행안 폴더도 같은 ACE만 상속한다.
        large = dict(self.record, draft={"value": "x" * (task_store.PART_INLINE_BYTES + 10)})
        task_store.save_task(large, expected_revision=0, root=self.root)
        self.assertTrue(store.check_tree(self.directory / "parts", store.trusted_principals()))
        self.assertEqual(self.lock(), self.directory)

    def test_other_principal_rights_on_record_files_are_rejected(self):
        large = dict(self.record, draft={"value": "x" * (task_store.PART_INLINE_BYTES + 10)})
        task_store.save_task(large, expected_revision=0, root=self.root)
        (self.directory / "plans").mkdir()
        (self.directory / "plans" / "plan.json").write_text("{}", encoding="utf-8")
        (self.directory / "snapshot").mkdir()
        targets = [self.directory / "task.json", next((self.directory / "parts").iterdir()), self.directory / "plans" / "plan.json",
                   self.directory / "snapshot"]
        for target in targets:
            for right in ("(M)", "(D)", "(WDAC)"):
                with self.subTest(target=target.name, right=right):
                    flags = "(OI)(CI)" if target.is_dir() else ""
                    if icacls(target, "/grant", f"*{AUTHENTICATED_USERS}:{flags}{right}") != 0:
                        self.skipTest("이 환경에서 시험용 ACL을 바꿀 수 없습니다")
                    before = store._security(target)
                    self.assert_unprotected()
                    # 권한을 고치지 않는다.
                    self.assertEqual(store._security(target), before)
                    icacls(target, "/remove:g", f"*{AUTHENTICATED_USERS}")
        self.assertEqual(self.lock(), self.directory)

    def test_forged_consent_with_matching_hash_is_stopped_at_protection_boundary(self):
        # 내용과 해시를 함께 맞춘 위조 동의는 해시 검사를 통과하므로, 다른 주체의 쓰기 권한 자체로 거부해야 한다.
        content = {"allowed": [["module", "stage"]], "limits": {}}
        record = task_store.load_task(self.task_id, root=self.root)
        record["consents"]["A"] = task_consent.grant("A", self.task_id, content)
        task_consent.check(record["consents"]["A"], "A", self.task_id)
        task_store.save_task(record, expected_revision=0, root=self.root)
        icacls(self.directory / "task.json", "/grant", f"*{AUTHENTICATED_USERS}:(M)")
        self.assert_unprotected()

    def test_folder_inheritance_or_broad_ace_is_rejected(self):
        icacls(self.directory, "/inheritance:e")
        self.assert_unprotected()

    def test_legacy_unprotected_task_is_view_only(self):
        legacy = self.root / ("a" * 32)
        legacy.mkdir()
        record = dict(self.record, task_id="a" * 32)
        (legacy / "task.json").write_text(json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        before = {path.name: path.read_bytes() for path in legacy.iterdir()}
        # 읽기 전용 조회는 기존처럼 된다. 승인·진행·복구 잠금은 열지 않는다.
        self.assertEqual(task_store.load_task("a" * 32, root=self.root)["task_id"], "a" * 32)
        with self.assertRaises(task_store.TaskRecordError) as caught:
            with task_store.task_lock("a" * 32, root=self.root):
                pass
        self.assertEqual(caught.exception.code, "RECORD_UNPROTECTED")
        # 승인 없는 보고서 재조립용 잠금은 허용한다(보유는 유지).
        with task_store.task_lock("a" * 32, root=self.root, protected=False):
            pass
        self.assertEqual({path.name: path.read_bytes() for path in legacy.iterdir() if path.name != "executor.lock"},
                         {name: data for name, data in before.items()})

    def test_task_folder_and_store_are_held_during_lock_and_released_after(self):
        with task_store.task_lock(self.task_id, root=self.root):
            for path in (self.directory, self.root):
                with self.subTest(path=path.name), self.assertRaises(OSError):
                    os.rename(path, str(path) + "_moved")
        with self.assertRaises(RuntimeError):
            with task_store.task_lock(self.task_id, root=self.root):
                raise RuntimeError("boom")
        os.rename(self.directory, self.root / "moved")

    def test_creation_failure_and_hard_link_are_classified(self):
        with patch.object(task_store, "_create_protected", side_effect=OSError(5, "RUNTIME_ACL_SETUP_FAILED")):
            with self.assertRaises(task_store.TaskRecordError) as caught:
                task_store.create_task("goal", "source.csv", "tabular_v2", root=self.root)
        self.assertEqual(caught.exception.code, "RECORD_PROTECT_FAILED")
        # 작업 폴더 직속 파일이 하드 링크면 보호 경계 밖으로 본다.
        os.link(self.directory / "task.json", self.root.parent / "linked.json")
        self.assert_unprotected()

    def test_hold_failure_is_classified(self):
        with patch.object(task_store, "_hold", side_effect=OSError(5, "denied")):
            with self.assertRaises(task_store.TaskRecordError) as caught:
                self.lock()
        self.assertEqual(caught.exception.code, "RECORD_UNREADABLE")


class DelegationEvidenceTest(unittest.TestCase):
    def test_authorizer_returns_consent_evidence(self):
        task_id = "b" * 32
        content = {"allowed": [["module", "stage"]], "limits": {"memory_mib": 1}}
        task = {"task_id": task_id, "attempt": 2, "consents": {"C": task_consent.grant("C", task_id, content)}}
        evidence = task_consent.authorizer(task, "C", "module", "stage", expected={})(
            {"summary": {"stage": "stage", "limits": {"memory_mib": 1}}})
        self.assertEqual(evidence, {"decision": "C", "consent_id": task["consents"]["C"]["consent_id"],
                                    "consent_sha256": task["consents"]["C"]["content_sha256"], "task_id": task_id, "attempt": 2})
        # 다른 작업의 동의는 거부한다.
        with self.assertRaises(ValueError):
            task_consent.authorizer(dict(task, task_id="c" * 32), "C", "module", "stage", expected={})

    def test_event_evidence_has_fixed_shape(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        good = {"decision": "A", "consent_id": "1" * 32, "consent_sha256": "2" * 64, "task_id": "3" * 32, "attempt": 0,
                "scope_sha256": "4" * 64}
        audit.append_event(directory / "events.jsonl", "approval_requested", "DELEGATED_APPROVAL_RECORDED", stage="ingest",
                           revision=1, evidence=good)
        self.assertEqual(json.loads((directory / "events.jsonl").read_text(encoding="utf-8"))["evidence"], good)
        for bad in (dict(good, decision="B"), dict(good, consent_id="x"), dict(good, attempt=-1), dict(good, extra="raw"),
                    {key: value for key, value in good.items() if key != "scope_sha256"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                audit.append_event(directory / "events.jsonl", "approval_requested", "DELEGATED_APPROVAL_RECORDED",
                                   stage="ingest", revision=1, evidence=bad)

    def test_event_failure_issues_no_approval(self):
        from tests.agents.test_tabular_input import TabularInputTest
        from src.agents import tabular_input_service as service
        fixture = TabularInputTest("test_inspect_and_materialize_consume_separate_approvals")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.create(fixture.inspect, "inspect1")
        for _ in range(3):
            state = fixture.inspect.status("inspect1")
            if state["status"] == "awaiting_approval":
                break
            fixture.inspect.next("inspect1", expected_stage=state["stage"], expected_revision=state["revision"])
        task_id = "d" * 32
        task = {"task_id": task_id, "attempt": 1, "consents": {"A": task_consent.grant(
            "A", task_id, {"allowed": [["tabular_inspect", "ingest"]], "limits": __import__("dataclasses").asdict(fixture.limits)})}}
        authorize = task_consent.authorizer(task, "A", "tabular_inspect", "ingest", expected={})
        # 이벤트 기록이 실패하면 승인 DB에 발급하지 않는다(자동 재시도·중복 발급 없음).
        with patch("src.agents.delegated_approval.append_event", side_effect=OSError("disk")), self.assertRaises(OSError):
            fixture.inspect.record_delegated_approval("inspect1", fixture.inspect.preview_approval("inspect1"), authorize=authorize)
        self.assertEqual(fixture.inspect.status("inspect1")["actions"], [{"id": "approve"}])
        issued = fixture.inspect.record_delegated_approval("inspect1", fixture.inspect.preview_approval("inspect1"),
                                                           authorize=authorize)
        self.assertTrue(issued["approval_id"])
        events = [json.loads(line) for line in (fixture.root / "runs/inspect/inspect1/events.jsonl").read_text(
            encoding="utf-8").splitlines()]
        recorded = [item["evidence"] for item in events if item["code"] == "DELEGATED_APPROVAL_RECORDED"]
        self.assertEqual(len(recorded), 1)
        self.assertEqual((recorded[0]["consent_id"], recorded[0]["decision"], recorded[0]["task_id"], recorded[0]["attempt"]),
                         (task["consents"]["A"]["consent_id"], "A", task_id, 1))
        self.assertEqual(recorded[0]["scope_sha256"], issued["scope_sha256"])


class FlowAndSourceTest(unittest.TestCase):
    """실제 v2 흐름(모의 운송·모델): 보호된 작업 폴더, 위임 근거 연결, 후속 출처 소비 경계."""

    def setUp(self):
        from tests.agents.test_tabular_task_v2 import TabularTaskV2Test
        self.fixture = TabularTaskV2Test("test_current_csv_runs_v2_from_input_to_report_with_typed_handoff")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_delegation_events_link_consent_and_task_folder_is_protected(self):
        task = self.fixture.diagnosed()
        directory = self.fixture.store / task["task_id"]
        self.assertTrue(store._security(directory)["protected"])
        events = [json.loads(line) for path in (directory / "input_runs").glob("*/*/events.jsonl")
                  for line in path.read_text(encoding="utf-8").splitlines()]
        evidence = [item["evidence"] for item in events if item["code"] == "DELEGATED_APPROVAL_RECORDED"]
        # 입력 검사·생성(input_runs)과 진단(pipeline_approval_cli) 위임 모두 같은 동의 A 근거를 남긴다.
        diagnostic = [json.loads(line) for path in (self.fixture.root / "diag_runs").glob("*/events.jsonl")
                      for line in path.read_text(encoding="utf-8").splitlines()]
        evidence += [item["evidence"] for item in diagnostic if "evidence" in item]
        self.assertEqual(len(evidence), 3)
        for item in evidence:
            self.assertEqual((item["decision"], item["consent_id"], item["consent_sha256"], item["task_id"]),
                             ("A", task["consents"]["A"]["consent_id"], task["consents"]["A"]["content_sha256"], task["task_id"]))

    def test_downstream_source_check_requires_protected_input_store(self):
        from src.agents import tabular_input_flow as flow
        task = self.fixture.diagnosed()
        directory = self.fixture.store / task["task_id"]
        record = task_store.load_task(task["task_id"], root=self.fixture.store)
        flow.check_source(record)
        run = directory / "input_runs" / "materialize" / record["input_selection"]["materialize_run"]
        # 생성 실행 기록에 다른 주체의 쓰기 권한이 생기면 출처 소비를 거부한다(해시·DB가 같아도).
        if icacls(run / "approvals.sqlite3", "/grant", f"*{AUTHENTICATED_USERS}:(M)") != 0:
            self.skipTest("이 환경에서 시험용 ACL을 바꿀 수 없습니다")
        from src.agents.task_plan import TaskProblem
        # 후속 확인의 저장소 오류는 입력 오류와 구분한 고정 문제로 보인다.
        with self.assertRaises(TaskProblem) as caught:
            flow.check_source(record)
        self.assertEqual(caught.exception.code, "INPUT_STORE_PERMISSION_INVALID")
        icacls(run / "approvals.sqlite3", "/remove:g", f"*{AUTHENTICATED_USERS}")
        flow.check_source(record)

    def test_source_check_holds_input_store_and_rejects_changed_reference(self):
        from src.agents import tabular_input_flow as flow
        from src.agents.split_evidence import CommittedArtifact
        task = self.fixture.diagnosed()
        directory = self.fixture.store / task["task_id"]
        record = task_store.load_task(task["task_id"], root=self.fixture.store)
        attempts = []
        real = CommittedArtifact.check

        def renaming(artifact):
            target = directory / "input_runs"
            try:
                os.rename(target, str(target) + "_swapped")
                attempts.append(target.name)
            except OSError:
                pass
            return real(artifact)

        with patch.object(CommittedArtifact, "check", renaming):
            flow.check_source(record)
        self.assertEqual(attempts, [])
        # 작업 기록의 출처 참조를 다른 실행으로 바꾸면 거부한다.
        changed = json.loads(json.dumps(record))
        changed["input_selection"]["materialize_run"] = "other"
        with self.assertRaises(ValueError):
            flow.check_source(changed)

    def test_unprotected_input_store_is_rejected_for_source_contract(self):
        task = self.fixture.diagnosed()
        directory = self.fixture.store / task["task_id"]
        record = task_store.load_task(task["task_id"], root=self.fixture.store)
        from src.agents.tabular_source_contract import TabularSourceContract
        source = TabularSourceContract.from_dict(record["input_selection"]["source"])
        icacls(directory / "input_runs", "/inheritance:e")
        with self.assertRaises(StoreUnavailable):
            source.check_input()


if __name__ == "__main__":
    unittest.main()

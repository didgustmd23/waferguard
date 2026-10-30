"""표형 v2 시간 결측 복원 미리보기·적용을 실제 실행기·승인 DB·worker 계산·부모 검증으로 검사한다.

OS 운송(AppContainer·Job)만 모의한다(입력 도구 테스트와 같은 방식). 합성 CSV와 고정 날짜만 쓰고 실제 데이터·모델은 쓰지 않는다.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.agents import tabular_input_service as input_service
from src.agents import tabular_time_service as time_service
from src.agents import tabular_time_worker as time_worker
from src.agents.input_snapshot import create_snapshot
from src.agents.tabular_time_results import read_application, read_preview
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.tabular_bundle import _write, iter_bundle_rows, load_bundle, read_bundle_provenance, read_bundle_schema
from src.preprocessing.tabular_loaders import CsvReadSpec
from tests.agents import test_tabular_input as input_fixture


RUNTIME_ID = input_fixture.RUNTIME_ID
RULES = {"column": "t", "timestamp_format": "%Y-%m-%d", "timezone": "Asia/Seoul", "frequency": "daily", "interval_minutes": None,
         "row_order_confirmed": True}
# 원본 시간값(결측이 아닌 값). 작업 기록·오류·실행 기록에 나오면 안 된다.
ORIGINAL = ("2026-10-19", "2026-10-21", "2026-10-22", "2026-10-24")
TIME_CSV = ("x,label,t\n1,a,2026-10-19\n2,b,\n3,a,2026-10-21\n4,b,2026-10-22\n5,a,\n6,b,2026-10-24\n")


def time_transport(request, root, cancel, *, runtime_id):
    """OS 운송만 모의하고 시간 worker 계산·결과 파일은 실제로 만든다. 실제 운송처럼 승인 packet의 런타임 ID를 기록한다."""
    reply = time_worker.execute_request(request, root, runtime_id)
    (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
    outcome = {"request_id": request.request_id, "profile_cleanup": True,
               "runtime": {"id": runtime_id, "prepared": False, "timings": {}},
               "launch": {"status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
                          "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                                        "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                                        "configured_process_limit": request.limits.processes}}}
    (Path(root) / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
    return outcome


def files_digest(folder):
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(folder).rglob("*")) if path.is_file()}


class TimeServiceFixture(unittest.TestCase):
    """생성된 표 출처 하나에 미리보기·적용 서비스를 실제 실행기로 진행하는 준비. 테스트 메서드는 없다."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target, side_effect in (("src.agents.tabular_input_service.run_input_worker", input_fixture.transport),
                                    ("src.agents.tabular_time_service.run_time_worker", time_transport)):
            patcher = patch(target, side_effect=side_effect)
            patcher.start()
            self.addCleanup(patcher.stop)

    def source(self, text=TIME_CSV, name="s"):
        """합성 CSV를 입력 도구(검사·생성)로 작업용 표 출처 계약까지 만든다."""
        folder = self.root / name
        folder.mkdir()
        (folder / "source.csv").write_text(text, encoding="utf-8")
        snapshot = create_snapshot(folder / "source.csv", folder / "snapshot", max_input_bytes=65536)
        arguments = {"snapshot_directory": str(folder / "snapshot"), "snapshot": asdict(snapshot), "spec": CsvReadSpec().to_dict(),
                     "limits": asdict(WorkerLimits(60, 2048, 2)), "runtime_id": RUNTIME_ID, "max_output_bytes": 8 * 1024 * 1024}
        inspect = input_service.TabularInputService("inspect", root=folder / "input_runs")
        materialize = input_service.TabularInputService("materialize", root=folder / "input_runs")
        inspect.create("i1", **arguments)
        self.assertEqual(input_fixture.drive(inspect, "i1")["status"], "completed")
        materialize.create("m1", inspection_run="i1", **arguments)
        self.assertEqual(input_fixture.drive(materialize, "m1")["status"], "completed")
        return materialize.source_contract("m1"), folder

    def run_preview(self, source, folder, rules=RULES, name="p1"):
        service = time_service.TabularTimeService("preview", root=folder / "time_runs")
        service.create(name, source=source, rules=rules, limits=asdict(source.limits), runtime_id=RUNTIME_ID)
        self.assertEqual(input_fixture.drive(service, name)["status"], "completed")
        return service, service.verified(name)

    def run_apply(self, source, folder, preview_run="p1", name="a1"):
        service = time_service.TabularTimeService("apply", root=folder / "time_runs")
        service.create(name, source=source, rules=RULES, limits=asdict(source.limits), runtime_id=RUNTIME_ID, preview_run=preview_run)
        return service, input_fixture.drive(service, name)


class TimeServiceTest(TimeServiceFixture):
    def test_preview_then_apply_creates_a_verified_derived_table_and_keeps_the_parent(self):
        source, folder = self.source()
        parent_before = files_digest(source.bundle_directory)
        csv_before = (folder / "source.csv").read_bytes()
        _, (_, result) = self.run_preview(source, folder)
        preview = result["preview"]
        self.assertEqual((preview["status"], preview["rows"], preview["restored_count"], preview["row_count"]),
                         ("ready", [1, 4], 2, 6))
        self.assertEqual(preview["times"], ["2026-10-20T00:00:00+09:00", "2026-10-23T00:00:00+09:00"])
        service, state = self.run_apply(source, folder)
        self.assertEqual(state["status"], "completed")
        derived = service.derived_source("a1")
        self.assertTrue(derived.check_input())
        self.assertEqual(derived.parent.source_id, source.source_id)
        self.assertTrue(derived.source_id.startswith("tabular_time_source:"))
        self.assertEqual(time_service.TimeRestoredSource.from_dict(json.loads(json.dumps(derived.to_dict()))), derived)
        frame = load_bundle(derived.bundle_directory, expected=derived.bundle)
        original = load_bundle(source.bundle_directory, expected=source.bundle)
        self.assertEqual(frame["t"].tolist(), ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24"])
        self.assertEqual(frame[["x", "label"]].to_dict(), original[["x", "label"]].to_dict())
        self.assertEqual(str(frame["t"].dtype), str(original["t"].dtype))
        provenance = read_bundle_provenance(derived.bundle_directory, expected=derived.bundle)
        self.assertEqual((provenance["parent_source_id"], provenance["restored_count"], provenance["preview_sha256"]),
                         (source.source_id, 2, result["preview_sha256"]))
        # 원래 생성 표와 원본 CSV는 바뀌지 않는다.
        self.assertEqual(files_digest(source.bundle_directory), parent_before)
        self.assertEqual((folder / "source.csv").read_bytes(), csv_before)
        # 완료된 적용 실행에는 새 승인을 발급하지 않고, 소비된 승인은 커밋된 작업에 묶여 다시 쓰지 않는다.
        _, _, manager = service._open("a1")
        self.assertEqual(manager.approvals.get_task(manager.scope_for("ingest"))["status"], "committed")
        self.assertEqual(service.status("a1")["actions"], [])
        with self.assertRaises(ValueError):
            service.preview_approval("a1")

    def test_each_stop_condition_is_a_verified_preview_without_original_values_and_cannot_be_applied(self):
        cases = {
            "TIME_EDGE_MISSING": ("x,t\n1,\n2,2026-10-20\n3,2026-10-21\n", RULES),
            "TIME_DUPLICATE": ("x,t\n1,2026-10-19\n2,2026-10-19\n3,2026-10-20\n", RULES),
            "TIME_FREQUENCY_MISMATCH": ("x,t\n1,2026-10-19\n2,\n3,2026-10-22\n", RULES),
            "TIME_PARSE_FAILED": ("x,t\n1,2026-10-19\n2,19/10/2026\n3,2026-10-21\n", RULES),
            "TIME_AMBIGUOUS": ("x,t\n1,2026-03-07 02:30\n2,\n3,2026-03-09 02:30\n",
                               dict(RULES, timestamp_format="%Y-%m-%d %H:%M", timezone="America/New_York")),
        }
        cases["TIME_EDGE_MISSING_END"] = ("x,t\n1,2026-10-19\n2,2026-10-20\n3,\n", RULES)
        for index, (label, (text, rules)) in enumerate(cases.items()):
            with self.subTest(code=label):
                source, folder = self.source(text, name=f"c{index}")
                service, (_, result) = self.run_preview(source, folder, rules)
                preview = result["preview"]
                self.assertEqual((preview["status"], preview["code"], preview["rows"]),
                                 ("stopped", label.replace("_END", ""), []))
                # 실행 기록·결과 파일에는 원본 시간값이 없다.
                for path in (folder / "time_runs").rglob("*"):
                    if path.is_file() and path.name in {"events.jsonl", "selection.json", "state.json", "transport.json",
                                                        "preview.json", "binding.json", "diagnostic.json"}:
                        content = path.read_text(encoding="utf-8")
                        self.assertFalse([value for value in ("19/10/2026", "2026-10-19", "02:30") if value in content], path)
                with self.assertRaises(ValueError):
                    self.run_apply(source, folder)
                self.assertFalse((folder / "time_runs/apply/a1").exists())

    def test_apply_rejects_a_preview_of_other_rules_or_a_changed_preview_file(self):
        source, folder = self.source()
        self.run_preview(source, folder)
        # 다른 규칙의 적용은 미리보기와 대조에서 거부한다(새 미리보기가 필요).
        service = time_service.TabularTimeService("apply", root=folder / "time_runs")
        with self.assertRaises(ValueError):
            service.create("a1", source=source, rules=dict(RULES, timezone="UTC"), limits=asdict(source.limits),
                           runtime_id=RUNTIME_ID, preview_run="p1")
        with self.assertRaises(ValueError):
            service.create("a1", source=source, rules=RULES, limits=asdict(source.limits), runtime_id="f" * 64, preview_run="p1")
        # 커밋된 미리보기 파일이 바뀌면 미리보기 결과 검증부터 실패한다.
        committed = time_service.TabularTimeService("preview", root=folder / "time_runs").verified("p1")[0]
        preview = committed / "preview.json"
        preview.write_bytes(preview.read_bytes().replace(b"2026-10-20", b"2026-10-21"))
        with self.assertRaises(ValueError):
            self.run_apply(source, folder)

    def test_worker_refuses_to_apply_when_the_rejudgement_differs_from_the_confirmed_preview(self):
        source, folder = self.source()
        self.run_preview(source, folder)
        real = time_worker.execute_request

        def changed(request, root, runtime_id):
            # 승인 뒤 worker 입력의 미리보기가 다른 판정으로 바뀐 경우(해시는 승인 값과 다름)
            path = Path(root) / "input/preview.json"
            path.write_bytes(path.read_bytes().replace(b'"restored_count": 2', b'"restored_count": 3'))
            return real(request, root, runtime_id)

        with patch.object(time_worker, "execute_request", side_effect=changed):
            service, state = self.run_apply(source, folder)
        self.assertEqual(state["status"], "blocked_input")
        self.assertEqual(service.worker_diagnostic("a1"), {"stage": "COMPUTE", "code": "TIME_PREVIEW_CHANGED"})
        with self.assertRaises(ValueError):
            service.derived_source("a1")


class TimeResultTamperTest(TimeServiceFixture):
    """부모 결과 검증이 worker가 만든 자기 일관된 변조 결과(형식·지문은 맞춤)를 거부하는지 확인한다."""

    def binding(self, copy, **changes):
        path = copy / "binding.json"
        value = dict(json.loads(path.read_text(encoding="utf-8")), **changes)
        path.write_bytes(time_worker.dump(value))

    def rewrite(self, copy, row_change=None, provenance_change=None):
        """같은 형식의 새 묶음으로 바꾸고 binding의 묶음 지문도 맞춘다."""
        from src.preprocessing.tabular_bundle import BundleRef
        reference = BundleRef.from_dict(json.loads((copy / "binding.json").read_text(encoding="utf-8"))["bundle"])
        table = copy / "table"
        schema = read_bundle_schema(table, expected=reference)
        provenance = read_bundle_provenance(table, expected=reference)
        rows = list(iter_bundle_rows(table, expected=reference))
        if row_change:
            rows = [row_change(index, row) for index, row in enumerate(rows)]
        if provenance_change:
            provenance = provenance_change(dict(provenance))
        replacement = copy.parent / (copy.name + "-table")
        written = _write(replacement, schema, rows, provenance, 8 * 1024 * 1024)
        shutil.rmtree(table)
        shutil.move(str(replacement), str(table))
        self.binding(copy, bundle=asdict(written))

    def rehash_preview(self, copy, **changes):
        preview = json.loads((copy / "preview.json").read_text(encoding="utf-8"))
        (copy / "preview.json").write_bytes(time_worker.dump(dict(preview, **changes)))
        self.binding(copy, preview_sha256=hashlib.sha256((copy / "preview.json").read_bytes()).hexdigest())

    def test_tampered_application_results_are_rejected(self):
        import pandas as pd

        parent, folder = self.source()
        self.run_preview(parent, folder)
        service, _ = self.run_apply(parent, folder)
        output, _ = service.verified("a1")
        request = service._open("a1")[1]._request(output.name)
        cases = {
            "다른 셀 변경": lambda copy: self.rewrite(copy, row_change=lambda index, row: (
                row[0], row[1], [row[2][0] + (index == 0), *row[2][1:]])),
            "시간 셀이 다른 값": lambda copy: self.rewrite(copy, row_change=lambda index, row: (
                row[0], row[1], [*row[2][:2], "2026-10-25" if index == 1 else row[2][2]])),
            "시간 셀을 채우지 않음": lambda copy: self.rewrite(copy, row_change=lambda index, row: (
                row[0], row[1], [*row[2][:2], pd.NA if index == 4 else row[2][2]])),
            "결측이 아닌 셀을 채움": lambda copy: self.rewrite(copy, row_change=lambda index, row: (
                row[0], row[1], [*row[2][:2], "2026-10-18" if index == 0 else row[2][2]])),
            "provenance 불일치": lambda copy: self.rewrite(copy, provenance_change=lambda value: dict(value, restored_count=3)),
            "미리보기 건수 불일치": lambda copy: self.rehash_preview(copy, restored_count=1),
            "미리보기 행 변경": lambda copy: self.rehash_preview(copy, rows=[1, 3]),
            "미리보기 지문 불일치": lambda copy: self.binding(copy, preview_sha256="0" * 64),
        }
        for index, (label, tamper) in enumerate(cases.items()):
            with self.subTest(case=label):
                copy = self.root / f"copy{index}"
                shutil.copytree(output, copy, ignore=shutil.ignore_patterns("AGENT_COMPLETE.json"))
                self.assertTrue(read_application(copy, request, parent_directory=Path(parent.bundle_directory)))
                tamper(copy)
                with self.assertRaises(ValueError):
                    read_application(copy, request, parent_directory=Path(parent.bundle_directory))
        # 적용 결과(파생 표 포함)를 미리보기 결과로 읽지 않는다.
        with self.assertRaises(ValueError):
            read_preview(self.root / "copy0", request, parent_directory=Path(parent.bundle_directory))
        # 형식·digest가 자기 일관되지만 규칙과 다른 미리보기로 만든 적용 결과도 부모 재판정에서 거부한다.
        copy = self.root / "forged-apply"
        shutil.copytree(output, copy, ignore=shutil.ignore_patterns("AGENT_COMPLETE.json"))
        self.forge(copy, rows=[1, 3], times=["1999-01-01T00:00:00+09:00", "2100-01-01T00:00:00+09:00"])
        with self.assertRaises(ValueError):
            read_application(copy, request, parent_directory=Path(parent.bundle_directory))

    def forge(self, copy, **changes):
        """미리보기를 바꾸고 digest·건수·binding 지문을 다시 맞춘다(worker 손상 가정의 자기 일관된 위조)."""
        from datetime import datetime
        from src.preprocessing.time_restore import Restoration, restoration_digest
        preview = dict(json.loads((copy / "preview.json").read_text(encoding="utf-8")), **changes)
        times = tuple(datetime.fromisoformat(text) for text in preview["times"])
        preview.update(restored_count=len(preview["rows"]),
                       restoration_digest=restoration_digest(Restoration(tuple(preview["rows"]), times)))
        self.rehash_preview(copy, **preview)

    def preview_copy(self, parent, folder, name):
        service, (output, _) = self.run_preview(parent, folder)
        request = service._open("p1")[1]._request(output.name)
        copy = self.root / name
        shutil.copytree(output, copy, ignore=shutil.ignore_patterns("AGENT_COMPLETE.json"))
        self.assertTrue(read_preview(copy, request, parent_directory=Path(parent.bundle_directory)))
        return copy, request

    def test_self_consistent_previews_that_differ_from_the_rule_are_rejected(self):
        parent, folder = self.source()
        cases = {
            "행·시각 조작": dict(rows=[1, 3], times=["1999-01-01T00:00:00+09:00", "2100-01-01T00:00:00+09:00"]),
            "복원 행 누락": dict(rows=[1], times=["2026-10-20T00:00:00+09:00"]),
            "성공을 멈춤으로 위조": dict(status="stopped", code="TIME_EDGE_MISSING", rows=[], times=[]),
        }
        clean, request = self.preview_copy(parent, folder, "forged-clean")
        for index, (label, changes) in enumerate(cases.items()):
            with self.subTest(case=label):
                copy = self.root / f"forged{index}"
                shutil.copytree(clean, copy)
                self.forge(copy, **changes)
                with self.assertRaises(ValueError):
                    read_preview(copy, request, parent_directory=Path(parent.bundle_directory))
        # 멈춰야 하는 입력(앞 끝 결측)을 성공으로 위조한 미리보기도 거부한다.
        stopped, stopped_folder = self.source("x,t\n1,\n2,2026-10-20\n3,2026-10-21\n", name="edge")
        copy, request = self.preview_copy(stopped, stopped_folder, "forged-ready")
        self.forge(copy, status="ready", code=None, rows=[1], times=["2026-10-20T00:00:00+09:00"])
        with self.assertRaises(ValueError):
            read_preview(copy, request, parent_directory=Path(stopped.bundle_directory))


if __name__ == "__main__":
    unittest.main()

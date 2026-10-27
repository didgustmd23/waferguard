"""표형 v2 서비스·제안 결합·보고서 사실 연결을 합성 입력과 모의 OS 운송으로 검사한다."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents import tabular_input_service
from src.agents.artifacts import _fingerprint
from src.agents.csv_workflow_service import CsvWorkflowService
from src.agents.input_snapshot import create_snapshot
from src.agents.llm_workflow_binding import bind_workflow_step, create_bound_step
from src.agents.llm_workflow_contract import validate_workflow_proposal
from src.agents.csv_workflow_proposal_catalogue import workflow_proposal_specifications
from src.agents.preparation_module_plan import validate_preparation_modules
from src.agents.worker_contract import WorkerLimits
from src.agents.workflow_service_registry import CSV_IDS, TABULAR_IDS, workflow_service, workflow_specifications
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.split import SplitRules
from src.preprocessing.tabular_loaders import CsvReadSpec
from tests.agents import test_tabular_input as input_fixture


PIPELINE = Path(__file__).resolve().parents[2] / "configs/agents/pipelines/tabular_preparation_example.json"
CSV_TEXT = ("x,flag,label,g,id\n1.0,1,True,a,0012\n,0,False,a,0034\n9.0,1,True,a,0056\n"
            "100.0,0,False,b,0078\n,1,True,b,0090\n700.0,0,False,c,0102\n")


class TabularWorkflowFixture(unittest.TestCase):
    """입력 도구부터 v2 서비스까지 같은 합성 표를 쓰는 공통 준비."""

    csv_text = CSV_TEXT

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.csv"
        self.source.write_bytes(self.csv_text.encode("utf-8"))
        self.snapshot = create_snapshot(self.source, self.root / "snapshot", max_input_bytes=65536)
        self.limits = WorkerLimits(60, 2048, 2)
        self.calls = []
        self.profile = DatasetProfile("typed_workflow", "tabular_canonical_v2", "split_review_v1",
            ColumnRoles(numeric=("x",), target="label", group="g", identifiers=("id", "flag")), max_plot_features=1,
            split_rules=SplitRules(strategy="group", random_state=0, train_ratio=1 / 3, validation_ratio=1 / 3,
                                   test_ratio=1 / 3, require_all_classes=False))
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(json.dumps(self.profile.to_dict()), encoding="utf-8")
        self.handoff_runs, self.bundles = self.root / "handoff-runs", self.root / "bundles"
        self.handoff_runs.mkdir()
        self.bundles.mkdir()
        for identifier, directory in (("snapshot_split", "split"), ("partition_export", "export"),
                                      ("preparation_candidate", "candidate"), ("preparation_apply", "apply"),
                                      ("preparation_pipeline", "pipeline")):
            parent = self.root / directory
            parent.mkdir()
            self.start_patch("src.agents." + identifier + "_cli._directory", side_effect=lambda name, parent=parent, **kw: parent / name)
            self.start_patch("src.agents." + identifier + "_cli._current_actor", return_value="synthetic_actor")
        for module in ("csv_workflow_creation", "csv_handoff_workflow", "tabular_input_service", "tabular_workflow_service"):
            self.start_patch("src.agents." + module + "._current_actor", return_value="synthetic_actor")
        self.start_patch("src.agents.csv_handoff_workflow.directory_for",
                         side_effect=lambda name, **kw: self.handoff_runs / ("v2-" if kw.get("typed") else "v1-") / name)
        (self.handoff_runs / "v2-").mkdir()
        (self.handoff_runs / "v1-").mkdir()
        for target in ("src.agents.csv_handoff_cli._directory", "src.agents.csv_handoff_workflow._directory"):
            self.start_patch(target, side_effect=lambda name, **kw: self.bundles / name)
        for name in ("CsvWorkflowService", "TabularWorkflowService"):
            module = "csv_workflow_service" if name == "CsvWorkflowService" else "tabular_workflow_service"
            self.start_patch("src.agents." + module + "." + name + ".runtime", return_value={"supported": True, "notice": "synthetic OS only"})
        self.start_patch("src.agents.tabular_input_service.run_input_worker", side_effect=input_fixture.transport)
        from src.agents import snapshot_split_worker, partition_export_worker, module_pipeline_worker
        from src.agents import preparation_candidate_worker, preparation_apply_worker
        for target, worker in (("src.agents.snapshot_split.run_split_worker", snapshot_split_worker.execute_request),
                               ("src.agents.partition_export.run_export_worker", partition_export_worker.execute_request),
                               ("src.agents.module_pipeline.run_pipeline_worker", module_pipeline_worker.execute_request),
                               ("src.agents.preparation_candidate.run_candidate_worker", preparation_candidate_worker.execute_request),
                               ("src.agents.preparation_apply.run_apply_worker", preparation_apply_worker.execute_request)):
            self.start_patch(target, side_effect=lambda request, root, cancel, worker=worker: self.transport(request, root, worker))

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def transport(self, request, root, worker):
        self.calls.append(request.stage)
        reply = worker(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        outcome = {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes}}}
        (Path(root) / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
        return outcome

    def materialize(self, name="source1", runs=None):
        """입력 도구의 공개 API로 검사·생성까지 마친 실행 폴더와 출처 ID를 돌려준다.

        기본 위치는 작업 폴더 구조(tasks/<task>/input_runs)를 흉내 낸 관리 루트다.
        """
        self.start_patch("src.agents.tabular_workflow_service.TASK_ROOT", new=self.root / "tasks")
        runs = runs or self.root / "tasks" / "task1" / "input_runs"
        runs.mkdir(parents=True, exist_ok=True)
        inspect = tabular_input_service.TabularInputService("inspect", root=runs)
        build = tabular_input_service.TabularInputService("materialize", root=runs)
        for item, run, parent in ((inspect, name + "-inspect", None), (build, name, name + "-inspect")):
            item.create(run, snapshot_directory=str(self.root / "snapshot"), snapshot=asdict(self.snapshot),
                        spec=CsvReadSpec().to_dict(), limits=asdict(self.limits), inspection_run=parent,
                        max_output_bytes=8 * 1024 * 1024)
            self.assertEqual(input_fixture.drive(item, run)["status"], "completed")
        return str(runs / "materialize" / name), build.source_contract(name).source_id

    def finish(self, module_id, name):
        """서비스 공개 API로 단계별 명시 승인만 사용해 완료 지점까지 진행한다."""
        service = workflow_service(module_id, "2.0.0")
        approvals = 0
        for _ in range(20):
            state = service.status(name)
            if state["status"] != "created" and not state["actions"]:
                break
            action = state["actions"][0]["id"]
            approval_id = None
            if action == "approve":
                approvals += 1
                approval_id = service.record_approval(name, service.preview_approval(name), confirmed=True)["approval_id"]
            service.next(name, expected_stage=state["stage"], expected_revision=state["revision"], approval_id=approval_id)
        self.assertEqual(service.status(name)["status"], "completed")
        return approvals

    def proposal(self, modules=TABULAR_IDS, existing=()):
        rules = json.loads(validate_preparation_modules(json.loads(PIPELINE.read_text(encoding="utf-8"))).rules_json)
        specs = {item["module_id"]: item for item in workflow_proposal_specifications(include_tabular=True)}
        settings = {"tabular_split_review": asdict(self.profile.split_rules), "tabular_prepare_candidate": rules}
        steps, previous = [], []
        for index, module_id in enumerate(modules):
            steps.append({"step_id": f"s{index}", "module_id": module_id, "version": specs[module_id]["version"],
                          "inputs": specs[module_id]["inputs"], "settings": settings.get(module_id, {}), "depends_on": previous})
            previous = [f"s{index}"]
        context = {"allowed_modules": tuple((item, specs[item]["version"]) for item in modules),
                   "selected_settings": {key: settings.get(key, {}) for key in modules},
                   "existing_inputs": existing}
        response = {"schema_version": "llm_workflow_proposal_v1", "status": "proposed", "questions": [],
                    "workflow": {"workflow_id": "typed_flow", "steps": steps}}
        return validate_workflow_proposal(json.dumps(response), **context), context

    def values(self, module_id, source_run, source_id):
        return {
            "tabular_split_review": {"name": "split01", "source_run": source_run, "source_contract_id": source_id,
                                     "profile": str(self.profile_path)},
            "tabular_partition_export": {"name": "export01", "split_name": "split01", "source_contract_id": source_id,
                                         "max_partition_bytes": 65536},
            "tabular_prepare_candidate": {"name": "prep01", "split_name": "split01", "export_name": "export01",
                                          "source_contract_id": source_id, "pipeline": str(PIPELINE),
                                          "max_candidate_bytes": 65536, "max_partition_bytes": 65536},
            "tabular_prepare_apply": {"pipeline_name": "prep01", "source_contract_id": source_id},
            "tabular_local_handoff": {"name": "handoff01", "pipeline_name": "prep01", "source_contract_id": source_id,
                                      "max_bundle_bytes": 1024 ** 2},
        }[module_id]

    def run_chain(self, *, until="tabular_local_handoff"):
        """확정 제안의 각 단계를 binding으로 생성하고 서비스로 실행한다."""
        source_run, source_id = self.materialize()
        proposal, context = self.proposal()
        completed = {}
        for index, module_id in enumerate(TABULAR_IDS):
            binding = bind_workflow_step(proposal, **context, step_id=f"s{index}",
                                         values=self.values(module_id, source_run, source_id), limits=self.limits,
                                         completed_runs=dict(completed))
            created = create_bound_step(binding, confirmed=True)
            self.assertFalse(created["worker_started"])
            self.finish(module_id, created["run_name"])
            completed[f"s{index}"] = created["run_name"]
            if module_id == until:
                break
        return source_run, source_id


class RegistryTest(unittest.TestCase):
    def test_v2_ids_are_fixed_and_hidden_from_the_ui_catalogue(self):
        from src.agents.tabular_workflow_service import TabularWorkflowService

        for module_id in TABULAR_IDS:
            self.assertIsInstance(workflow_service(module_id, "2.0.0"), TabularWorkflowService)
            for version in ("1.0.0", "2.0.1", "2"):
                with self.assertRaises(ValueError):
                    workflow_service(module_id, version)
        for module_id in CSV_IDS:
            self.assertIsInstance(workflow_service(module_id, "1.0.0"), CsvWorkflowService)
            with self.assertRaises(ValueError):
                workflow_service(module_id, "2.0.0")
        self.assertFalse(set(TABULAR_IDS) & {item["contract"]["module_id"] for item in workflow_specifications()})
        # 모델 요청에 쓰는 기본 카탈로그는 기존 CSV 명세 그대로다.
        self.assertEqual([item["module_id"] for item in workflow_proposal_specifications()], list(CSV_IDS))
        specs = {item["module_id"]: item for item in workflow_proposal_specifications(include_tabular=True)}
        self.assertEqual({specs[item]["version"] for item in TABULAR_IDS}, {"2.0.0"})
        self.assertEqual({specs[item]["version"] for item in CSV_IDS}, {"1.0.0"})
        self.assertEqual(len({item["output"] for item in specs.values()}), len(specs))
        self.assertTrue(all("csv_file" not in json.dumps(workflow_service(m, "2.0.0").creation_schema()) for m in TABULAR_IDS))

    def test_v2_runtime_notice_does_not_claim_real_isolation_verification(self):
        from src.agents.tabular_workflow_service import TabularWorkflowService

        notice = TabularWorkflowService("tabular_split_review").runtime()["notice"]
        self.assertIn("AppContainer", notice)
        self.assertNotIn("합성 격리 7단계를 검증했습니다", notice)


class ProposalContractTest(TabularWorkflowFixture):
    def test_mixed_family_and_cross_family_inputs_are_rejected(self):
        self.proposal()
        specs = {item["module_id"]: item for item in workflow_proposal_specifications(include_tabular=True)}
        mixed = ("csv_split_review", "tabular_partition_export")
        steps = [{"step_id": "a", "module_id": mixed[0], "version": "1.0.0", "inputs": specs[mixed[0]]["inputs"],
                  "settings": asdict(self.profile.split_rules), "depends_on": []},
                 {"step_id": "b", "module_id": mixed[1], "version": "2.0.0", "inputs": specs[mixed[1]]["inputs"],
                  "settings": {}, "depends_on": ["a"]}]
        response = json.dumps({"schema_version": "llm_workflow_proposal_v1", "status": "proposed", "questions": [],
                               "workflow": {"workflow_id": "mixed", "steps": steps}})
        with self.assertRaises(ValueError):
            validate_workflow_proposal(response, allowed_modules=(("csv_split_review", "1.0.0"), ("tabular_partition_export", "2.0.0")),
                                       selected_settings={"csv_split_review": asdict(self.profile.split_rules)})
        with self.assertRaises(ValueError):
            self.proposal(modules=("tabular_partition_export",), existing=("split.confirmed",))
        self.assertIsNotNone(self.proposal(modules=("tabular_partition_export",), existing=("tabular_split.confirmed",))[0].plan_signature)
        with self.assertRaises(ValueError):
            self.proposal(modules=("tabular_split_review", "tabular_prepare_candidate"))


class TabularWorkflowServiceTest(TabularWorkflowFixture):
    def test_five_v2_services_run_end_to_end_with_lineage_and_report_sources(self):
        original = self.source.read_bytes()
        source_run, source_id = self.run_chain()
        self.assertEqual(original, self.source.read_bytes())
        self.assertEqual(self.calls.count("derive_preparation"), 1)
        self.assertEqual(self.calls.count("apply_preparation"), 1)
        names = {"tabular_split_review": "split01", "tabular_partition_export": "export01",
                 "tabular_prepare_candidate": "prep01", "tabular_prepare_apply": "prep01", "tabular_local_handoff": "handoff01"}
        for module_id, name in names.items():
            result = workflow_service(module_id, "2.0.0").results(name)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(any(item["kind"] == "artifact_list" for item in result["items"]), module_id)
            # 내보내기는 v1과 같이 구간 폴더만 제공하고 고정 보고서는 다른 단계에 있다.
            self.assertEqual(any(item["kind"] == "report" for item in result["items"]),
                             module_id != "tabular_partition_export", module_id)
        from src.agents.csv_workflow_binding_evidence import completed_workflow

        for module_id, name in names.items():
            self.assertEqual(completed_workflow(module_id, name)["lineage"]["source"], source_id)

        from src.agents.tabular_workflow_service import report_sources

        facts = report_sources({"split": "split01", "export": "export01", "candidate": "prep01",
                                "apply": "prep01", "handoff": "handoff01"}, source_id)
        self.assertEqual(facts["source"]["source_contract_id"], source_id)
        self.assertEqual(facts["source"]["read_spec"], CsvReadSpec().to_dict())
        self.assertEqual(facts["source"]["header_mapping"], [])
        self.assertEqual([column["name"] for column in facts["source"]["columns"]], ["x", "flag", "label", "g", "id"])
        self.assertEqual({column["name"]: column["dtype"] for column in facts["source"]["columns"]}["id"], "string")
        self.assertIn("status", facts["quality"])
        self.assertEqual({item["split"] for item in facts["proposal"]["summary"]}, {"train", "validation", "test"})
        self.assertEqual(facts["preprocessor"]["state"]["records"][0]["impute"], 5.0)
        self.assertEqual(facts["application"]["schema_version"], "preparation_application_v3")
        self.assertEqual(set(facts["application"]["partitions"]), {"train", "validation"})
        self.assertIs(facts["application"]["test_used"], False)
        self.assertIn("impact", facts["application"])
        self.assertIn("prepared/train_rows.json", facts["handoff"]["files"])
        self.assertEqual(facts["handoff"]["manifest_sha256"], _fingerprint(Path(facts["handoff"]["directory"]) / "HANDOFF_COMPLETE.json")["sha256"])
        self.assertTrue(Path(facts["handoff"]["readme"]).is_file())
        self.assertTrue(all(Path(path).is_file() for path in facts["reports"]))
        self.assertTrue(facts["local_only"])
        with self.assertRaises(ValueError):
            report_sources({"split": "split01", "export": "export01", "candidate": "prep01",
                            "apply": "prep01", "handoff": "handoff01"}, "tabular_source:" + "0" * 64)

        # 모든 단계 검증이 끝난 뒤 보고 근거를 읽기 직전에 위조한다. 위조 지점이 실제로 실행됐는지도 확인한다.
        from dataclasses import replace
        import sys
        from src.agents import tabular_workflow_service as service_module

        runs = {"split": "split01", "export": "export01", "candidate": "prep01", "apply": "prep01", "handoff": "handoff01"}
        real_committed, real_pipeline = service_module._committed_json, service_module._pipeline

        def committed_file(parent, relative):
            path = next((self.root / parent).rglob("attempts/*/" + relative))
            return path, path.parents[relative.count("/")] / "AGENT_COMPLETE.json"

        def rejected_after_verification(parent, relative, message, *, reseal):
            path, marker = committed_file(parent, relative)
            originals = {item: item.read_bytes() for item in (path, marker)}
            calls = []

            def tamper(output, name, **expected):
                if name == relative:
                    calls.append(name)
                    if not reseal:
                        # 크기 상한보다 해시 대조가 거부하도록 크기는 같고 내용만 다른 바이트로 바꾼다.
                        data = path.read_bytes()
                        path.write_bytes(data[:-1] + (b" " if data[-1:] != b" " else b"\n"))
                    else:
                        value = json.loads(path.read_text(encoding="utf-8"))
                        value["forged"] = True
                        path.write_text(json.dumps(value), encoding="utf-8")
                    if reseal:
                        record = json.loads(marker.read_text(encoding="utf-8"))
                        record["files"][relative] = _fingerprint(path)
                        marker.write_text(json.dumps(record), encoding="utf-8")
                return real_committed(output, name, **expected)

            try:
                with patch.object(service_module, "_committed_json", side_effect=tamper):
                    with self.assertRaisesRegex(ValueError, message):
                        report_sources(runs, source_id)
                self.assertEqual(calls, [relative])
            finally:
                for item, data in originals.items():
                    item.write_bytes(data)

        cases = (
            # quality는 승인 체인에 묶인 기대 해시가 없어 표식과 다른 파일만 거부한다(손상 검사 수준).
            ("split", "quality/quality.json", "보고 근거 파일이 완료 기록과 다릅니다", False),
            # 후보·전처리기는 검증 시점 표식 해시, 적용 기록은 봉인 manifest 해시에 묶여 표식 재작성도 거부한다.
            ("candidate", "candidate/candidate.json", "완료 기록이 검증 시점과 다릅니다", True),
            ("candidate", "candidate/preprocessor.json", "완료 기록이 검증 시점과 다릅니다", True),
            ("apply", "prepared/application.json", "보고 근거 파일이 검증된 결과와 다릅니다", True),
        )
        for parent, relative, message, reseal in cases:
            with self.subTest(relative=relative, reseal=reseal):
                rejected_after_verification(parent, relative, message, reseal=reseal)

        # 분할 EDA 표는 같은 분할 실행의 run_eda 문맥에 묶인 고정 파일 전체를 검증한 바이트로 돌려준다.
        from src.agents.adapters.tabular import REPORT_FILES

        self.assertEqual(set(facts["eda"]), REPORT_FILES)
        self.assertEqual(facts["split_rules"], asdict(self.profile.split_rules))
        eda_path, _ = committed_file("split", "eda/missing.csv")
        self.assertEqual(facts["eda"]["missing.csv"], eda_path.read_bytes())
        real_eda, eda_calls = service_module._committed_eda, []
        original = eda_path.read_bytes()

        def tampered_eda(output, context, profile, rows):
            eda_calls.append(context.tool_id)
            eda_path.write_bytes(original[:-1] + (b"9" if original[-1:] != b"9" else b"8"))
            return real_eda(output, context, profile, rows)

        try:
            with patch.object(service_module, "_committed_eda", tampered_eda):
                with self.assertRaisesRegex(ValueError, "보고 근거 파일이 완료 기록과 다릅니다"):
                    report_sources(runs, source_id)
        finally:
            eda_path.write_bytes(original)
        self.assertEqual(eda_calls, ["run_eda"])

        # R-1: 보고 근거로 받은 후보 증거가 승인된 적용 계약의 후보와 다르면 근거를 돌려주지 않는다.
        swapped = []

        def other_candidate(*args, **kwargs):
            result = real_pipeline(*args, **kwargs)
            if sys._getframe(1).f_code.co_name != "_report_sources":
                return result
            swapped.append(args)
            return result[:3] + (replace(result[3], output_columns_json="[]"),) + result[4:]

        # side_effect를 쓰지 않고 함수로 바꿔 호출자 프레임이 바로 _report_sources(검증 범위 안의 본문)가 되게 한다.
        with patch.object(service_module, "_pipeline", other_candidate):
            with self.assertRaisesRegex(ValueError, "승인된 적용 계약의 후보와 다릅니다"):
                report_sources(runs, source_id)
        self.assertEqual(len(swapped), 1)
        # 위조를 모두 되돌린 뒤에는 같은 근거를 다시 돌려준다.
        self.assertEqual(report_sources(runs, source_id)["application"], facts["application"])

    def test_source_mismatch_cross_version_and_unconfirmed_creation_are_rejected(self):
        source_run, source_id = self.materialize()
        split = workflow_service("tabular_split_review", "2.0.0")
        values = self.values("tabular_split_review", source_run, source_id)
        for changed in ({"source_contract_id": "tabular_source:" + "0" * 64}, {"source_contract_id": "csv:" + "0" * 64},
                        {"source_run": str(self.root / "tasks" / "task1" / "input_runs" / "inspect" / "source1-inspect")},
                        {"source_run": str(self.source)}):
            with self.assertRaises(ValueError):
                split.preview_form(dict(values, **changed), self.limits)
        preview = split.preview_form(values, self.limits)
        with self.assertRaises(ValueError):
            split.create(preview)
        self.assertFalse((self.root / "split" / "split01").exists())
        with self.assertRaises(ValueError):
            workflow_service("tabular_partition_export", "2.0.0").create(preview, confirmed=True)
        split.create(preview, confirmed=True)
        self.finish("tabular_split_review", "split01")
        export = workflow_service("tabular_partition_export", "2.0.0")
        export_values = self.values("tabular_partition_export", source_run, source_id)
        with self.assertRaises(ValueError):
            export.preview_form(dict(export_values, source_contract_id="tabular_source:" + "1" * 64), self.limits)
        with self.assertRaises(ValueError):
            CsvWorkflowService("csv_partition_export").preview_form(
                {"name": "export02", "split_name": "split01", "max_partition_bytes": 65536}, self.limits)
        with self.assertRaises(ValueError):
            CsvWorkflowService("csv_split_review").status("split01")
        legacy_profile = self.root / "legacy.json"
        legacy_profile.write_text(json.dumps(DatasetProfile("legacy", "tabular_csv_v1", "split_review_v1",
            ColumnRoles(numeric=("x",), target="label"), max_plot_features=1, split_rules=SplitRules()).to_dict()), encoding="utf-8")
        legacy = CsvWorkflowService("csv_split_review")
        legacy.create(legacy.preview_form({"name": "legacy01", "source": str(self.source), "profile": str(legacy_profile),
                                           "max_input_bytes": 65536, "allow_feature_findings": False}, self.limits), confirmed=True)
        with self.assertRaises(ValueError):
            split.status("legacy01")
        with self.assertRaises(ValueError):
            export.preview_form(dict(export_values, split_name="legacy01"), self.limits)

    def test_materialized_run_outside_managed_roots_is_rejected(self):
        outside, source_id = self.materialize(runs=self.root / "outside" / "input_runs")
        values = self.values("tabular_split_review", outside, source_id)
        with self.assertRaises(ValueError):
            workflow_service("tabular_split_review", "2.0.0").preview_form(values, self.limits)

    def test_changed_source_after_preview_and_recorded_results_are_refused(self):
        source_run, source_id = self.materialize()
        split = workflow_service("tabular_split_review", "2.0.0")
        preview = split.preview_form(self.values("tabular_split_review", source_run, source_id), self.limits)
        self.profile_path.write_text(json.dumps(self.profile.to_dict()) + " ", encoding="utf-8")
        with self.assertRaises(ValueError):
            split.create(preview, confirmed=True)
        self.assertFalse((self.root / "split" / "split01").exists())
        with self.assertRaises(ValueError):
            workflow_service("tabular_local_handoff", "2.0.0").recorded_results("handoff01", "0" * 64)


if __name__ == "__main__":
    unittest.main()

"""CSV CLI를 공통 승인 서비스에 연결한다. 계산·격리·승인 검증은 재사용한다."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .artifacts import check_artifacts
from .csv_workflow_creation import CsvCreationPreview, code_id, create, preview_creation
from .csv_workflow_schema import creation_schema
from .executor import RulesExecutor
from .workflow_approval import approval_progress, preview_approval, record_approval
from .worker_paths import _path


class SelectedExecutor(RulesExecutor):
    """기존 OS 잠금 안에서도 사용자가 본 단계·revision·정확한 승인을 다시 검사한다."""

    def __init__(self, original, tool, stage, revision):
        super().__init__(original.states, original.approvals, original.plan, tuple(original.tools.values()))
        self.tool, self.stage, self.revision = tool, stage, revision
        self.connection_code = code_id()

    def _run_step(self, state, approval_id):
        self.tool._check_arguments(self.tool.arguments)
        if code_id() != self.connection_code:
            raise ValueError("실행 잠금 대기 중 연결 코드가 변경됐습니다.")
        if state.current_stage != self.stage or state.revision != self.revision:
            raise ValueError("사용자가 선택한 단계·revision이 변경됐습니다.")
        if approval_id is not None and approval_id not in approval_progress(self, self.stage)["valid_approval_ids"]:
            raise ValueError("현재 범위의 정확한 미사용 승인 ID가 필요합니다.")
        return super()._run_step(state, approval_id)


class CsvWorkflowService:
    def __init__(self, module_id):
        from .workflow_service_registry import CSV_IDS

        if module_id not in CSV_IDS:
            raise ValueError("등록되지 않은 CSV 작업입니다.")
        self.module_id = module_id

    def runtime(self):
        local = self.module_id == "csv_local_handoff"
        supported = os.name == "nt" and sys.version_info[:2] in ((3, 10), (3, 12))

        return {"supported": supported, "code": "CSV_RUNTIME_SUPPORTED" if supported else "CSV_RUNTIME_UNVERIFIED",
                "notice": "로컬 고정 파일 복사입니다. AppContainer worker를 실행하는 작업이 아닙니다." if local else
                          "Windows Python 3.10·3.12를 지원합니다. 3.12 합성 격리 7단계를 검증했습니다. "
                          "관리자 권한·새 작업·별도 실행 승인이 필요하며 GPU 연산은 사용하지 않습니다."}

    def capabilities(self):
        return {"creation_supported": True, "runtime": self.runtime()}

    def creation_schema(self):
        return creation_schema(self.module_id)

    def preview_form(self, values, limits):
        return preview_creation(self.module_id, values, limits)

    def create(self, preview, *, confirmed=False):
        if not isinstance(preview, CsvCreationPreview) or preview.packet["module_id"] != self.module_id:
            raise ValueError("현재 모듈의 생성 미리보기가 필요합니다.")
        return {**create(preview, confirmed=confirmed), "runtime": self.runtime()}

    def _open(self, name):
        if self.module_id == "csv_split_review":
            from . import snapshot_split_cli as cli

            tool = cli.open_run(name)
        elif self.module_id == "csv_partition_export":
            from . import partition_export_cli as cli

            tool = cli.open_run(name)
        elif self.module_id in {"csv_prepare_candidate", "csv_prepare_apply"}:
            from . import preparation_pipeline_cli as pipeline
            from . import preparation_candidate_cli as candidate
            from . import preparation_apply_cli as application

            directory, record, source = pipeline.open_run(name)
            if (directory / "CANCEL_REQUESTED").exists():
                raise ValueError("취소된 통합 작업입니다.")
            cli, tool = candidate, source
            if self.module_id == "csv_prepare_apply":
                cli, tool = application, pipeline._application(directory, record, source)
                if tool is None:
                    raise ValueError("후보 검토 후 적용 작업을 먼저 준비하세요.")
        else:
            from . import csv_handoff_workflow as cli

            tool = cli.open_run(name)
            return cli, tool, cli.manager(tool)

        tool._check_arguments(tool.arguments)
        return cli, tool, cli.executor(tool)

    def _summary(self, tool):
        state = tool.states.load()
        reports = {stage: str(tool.states.directory / relative) for stage, relative in state.artifact_refs.items()}
        return {"stage": state.current_stage, "module_id": self.module_id,
                "limits": asdict(tool.limits), "arguments": tool.arguments,
                "local_reports": reports, "connection_code_id": code_id(),
                "notice": "현재 단계만 승인합니다. 원본 덮어쓰기·모델 학습·LLM 전송·외부 공유 없음."}

    def status(self, name):
        cli, tool, manager = self._open(name)
        state = tool.states.load()
        actions = []
        if not state.blockers and not state.cancel_requested and not tool.cancel.is_set() and state.status in {"created", "running", "awaiting_approval"}:
            progress = approval_progress(manager, state.current_stage)
            if not progress["task_exists"]:
                action = ("execute" if progress["valid_approval_ids"] else "approve") if state.status == "awaiting_approval" else "next"
                actions = [{"id": action}]
        return {"status": state.status, "stage": state.current_stage, "revision": state.revision,
                "actions": actions, "runtime": self.runtime(), "local_only": True}

    def preview_approval(self, name):
        cli, tool, manager = self._open(name)
        return preview_approval(tool, manager, self._summary)

    def record_approval(self, name, preview, *, confirmed=False):
        if not self.runtime()["supported"]:
            raise ValueError("현재 Python 환경의 CSV 실행은 검증되지 않았습니다.")
        cli, tool, manager = self._open(name)
        result = record_approval(tool, manager, self._summary, preview, confirmed=confirmed)
        proof = {"code_id": code_id(), "scope": asdict(manager.scope_for(tool.states.load().current_stage))}
        # 연결 코드 변경 후 이전 승인으로 실행하지 않는다. 기존 승인 DB 형식은 유지한다.
        with (tool.states.directory / ("ui_approval_" + result["approval_id"] + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(proof, stream, sort_keys=True)
        return result

    def next(self, name, *, expected_stage, expected_revision, approval_id=None):
        if type(expected_stage) is not str or type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("현재 단계와 정수 revision을 명시해야 합니다.")
        if not self.runtime()["supported"]:
            raise ValueError("현재 Python 환경의 CSV 실행은 검증되지 않았습니다.")
        cli, tool, manager = self._open(name)
        selected_state = tool.states.load()
        if selected_state.current_stage != expected_stage or selected_state.revision != expected_revision:
            raise ValueError("선택한 단계·revision이 변경됐습니다. 실행하지 않습니다.")
        if approval_id is not None:
            import re
            from .split_evidence import read_metadata

            if type(approval_id) is not str or re.fullmatch("[a-f0-9]{32}", approval_id) is None:
                raise ValueError("고정 형식의 승인 ID가 필요합니다.")
            proof = read_metadata(tool.states.directory / ("ui_approval_" + approval_id + ".json"))
            if proof != {"code_id": code_id(), "scope": asdict(manager.scope_for(expected_stage))}:
                raise ValueError("승인 후 연결 코드·범위가 변경됐습니다.")
        selected = SelectedExecutor(manager, tool, expected_stage, expected_revision)
        result = asdict(selected.run_next(approval_id=approval_id))
        native_requested = self.module_id != "csv_local_handoff" and approval_id is not None and expected_stage != "review_eda"
        return {**result, "worker_started": None if native_requested else False, "local_only": True}

    def results(self, name):
        cli, tool, manager = self._open(name)
        state = tool.states.load()
        if state.status not in {"completed", "blocked_input"}:
            raise ValueError("완료·품질 검토 중단 작업만 조회할 수 있습니다.")
        items = []
        if state.status == "completed":
            for step in tool.plan.steps:
                task = manager.approvals.get_task(manager.scope_for(step.stage))
                if task is None or task["status"] != "committed":
                    raise ValueError("커밋된 결과 증거가 없습니다.")
                output = manager._output(task)
                if not check_artifacts(output, manager._context(step)).reusable:
                    raise ValueError("결과 파일 해시·완료 기록이 변경됐습니다.")
                if not manager._call_tool(manager.tools[step.action_id].validate, state, step, output):
                    raise ValueError("도메인 결과 검증에 실패했습니다.")
                items.append({"kind": "artifact_list", "label": step.stage, "path": str(output)})
                # 결과 데이터 전체를 표시 목록으로 펼치지 않고 고정 보고서만 제공한다.
                for relative in ("quality/quality.md", "eda/eda.md", "proposal.md", "candidate/candidate.md",
                                 "prepared/application.md"):
                    path = output / relative
                    if path.is_file():
                        path = _path(path, directory=False)
                        label = step.stage + "/" + relative
                        items.append({"kind": "report", "label": label, "path": str(path)})
                        if relative == "eda/eda.md":
                            # 커밋·도메인 검증을 마친 폴더의 고정 집계만 제공한다.
                            for table in ("missing.csv", "numeric_statistics.csv", "pearson.csv", "correlation_support.csv"):
                                aggregate = _path(path.parent / table, directory=False)
                                items.append({"kind": "graph_evidence", "path": str(aggregate),
                                              "label": step.stage + "/eda/" + table,
                                              "plot_limit": tool.contract.profile.max_plot_features})
                        for child in path.parent.iterdir():
                            if child.suffix.lower() == ".png" and child.is_file():
                                items.append({"kind": "image", "label": step.stage + "/" + child.relative_to(output).as_posix(),
                                              "path": str(_path(child, directory=False))})
                if self.module_id == "csv_local_handoff" and step.stage == "ingest":
                    from .csv_handoff_cli import _directory

                    bundle = _path(_directory(name), directory=True)
                    items.extend([{"kind": "artifact_list", "label": "handoff", "path": str(bundle)},
                                  {"kind": "report", "label": "handoff/README.md", "path": str(bundle / "README.md")}])
        if tool.states.load() != state:
            raise ValueError("조회 중 상태가 변경됐습니다.")
        return {"status": state.status, "items": items, "local_only": True,
                "notice": "완료 결과는 원래 도메인 검증기를 통과해야 표시합니다. 품질 중단의 미검증 부분 결과는 표시하지 않습니다."}

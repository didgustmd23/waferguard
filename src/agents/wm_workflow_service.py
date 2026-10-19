"""WM 모듈의 공통 작업 서비스. CLI/UI용 처리기를 복제하지 않고 기존 구현을 호출한다."""

from dataclasses import asdict
from pathlib import Path

from .artifacts import _root
from .split_evidence import read_committed_artifact
from .wm_module_catalogue import _cli, open_wm_execution, wm_modules
from .wm_runtime import runtime_status
from .workflow_approval import approval_progress
from .worker_paths import _path


class WmWorkflowService:
    """등록된 WM 8개가 공유하는 표면. 화면·LLM·셸 명령을 포함하지 않는다.

    생성은 기존 통합 생성기를, 현재 단계 처리는 기존 실행기를 사용한다.
    status/results는 명시적 정밀 조회용이다. 큰 입력 해시 검사를 포함할 수 있으므로
    UI 렌더링마다 자동 호출하지 않는다. 반환한 행동 목록 자체는 실행 권한이 아니다.
    """

    def __init__(self, module_id, version="1.0.0"):
        self.module = next((item for item in wm_modules()
                            if (item.contract.module_id, item.contract.version) == (module_id, version)), None)

        if self.module is None:
            raise ValueError("정확히 등록된 WM 모듈 ID·버전이 필요합니다.")

        self.module_id = module_id
        self.version = version

    def _open(self, name):
        """기존 소유자·코드·입력·선행 증거 검사와 실행 관리자를 재사용한다."""
        tool, manager = open_wm_execution(self.module_id, self.version, name)
        return _cli(self.module.cli_id), tool, manager

    def capabilities(self):
        from .wm_run_creation import CREATION_MODULES

        operations = ["status", "preview_approval", "record_approval", "next", "results"]
        if self.module_id in CREATION_MODULES:
            operations = ["preview_creation", "create", *operations]

        return {"service_id": "wm_workflow_v1", "module_id": self.module_id, "version": self.version,
                "creation_supported": self.module_id in CREATION_MODULES,
                "creation_form_ready": self.module_id in CREATION_MODULES,
                "operations": operations,
                "runtime": runtime_status(self.module_id), "local_only": True}

    def creation_schema(self):
        from .wm_creation_schema import creation_schema

        return creation_schema(self.module_id)

    def preview_form(self, values, limits):
        """표시 명세의 값을 검증한 뒤 기존 생성 미리보기로 전달한다."""
        from .wm_creation_schema import creation_arguments

        return self.preview_creation(limits=limits, **creation_arguments(self.module_id, values))

    def preview_creation(self, pipeline, bindings, step_id, name, limits, **options):
        """정상 개발 경로의 기존 생성 미리보기. 충돌 분기는 미지원으로 명시한다."""
        from .wm_run_creation import CREATION_MODULES, preview_creation

        if self.module_id not in CREATION_MODULES:
            raise ValueError("이 모듈은 통합 생성 전환 전입니다. 기존 충돌 분기 생성 CLI를 사용하세요.")

        split_config = options.pop("split_config", None)
        preview = preview_creation(pipeline, bindings, step_id, name, split_config, limits, **options)

        if preview.packet["module_id"] != self.module_id:
            raise ValueError("선택한 서비스와 생성할 모듈이 다릅니다.")

        return preview

    def create(self, preview, *, confirmed=False):
        from .wm_run_creation import CreationPreview, create_and_assign

        if (confirmed is not True or not isinstance(preview, CreationPreview)
                or preview.packet["module_id"] != self.module_id):
            raise ValueError("현재 모듈에 연결된 생성 미리보기가 필요합니다.")

        return create_and_assign(preview, confirmed=confirmed)

    def status(self, name):
        """현재 상태·가능 행동을 같은 JSON 형식으로 반환한다. 자동 진행하지 않는다."""
        cli, tool, manager = self._open(name)
        state = tool.states.load()
        details = cli.status(tool)
        actions = []

        if not state.blockers and not tool.cancel.is_set() and not state.cancel_requested:
            step = next((item for item in manager.plan.steps if item.stage == state.current_stage), None)

            if step is not None and state.status in {"created", "running", "awaiting_approval"}:
                progress = approval_progress(manager, state.current_stage)

                if not progress["task_exists"]:
                    if state.status == "awaiting_approval":
                        action = "execute" if progress["valid_approval_ids"] else "approve"
                    else:
                        action = "next"

                    actions.append({"id": action, "requires_confirmation": action != "next",
                                    "approval_id_required": action == "execute"})

        if tool.states.load() != state:
            raise ValueError("조회 중 작업 상태가 변경되었습니다.")

        return {"schema_version": "workflow_service_status_v1", "module_id": self.module_id,
                "run_name": name, "status": state.status, "stage": state.current_stage,
                "revision": state.revision, "actions": actions, "details": details,
                "runtime": runtime_status(self.module_id), "local_only": True}

    def preview_approval(self, name):
        cli, tool, _ = self._open(name)
        return cli.preview_approval(tool)

    def record_approval(self, name, preview, *, confirmed=False):
        cli, tool, _ = self._open(name)
        return cli.record_approval(tool, preview, confirmed=confirmed)

    def next(self, name, *, expected_stage, expected_revision, approval_id=None):
        """사용자가 본 단계·revision만 한 번 진행한다. UI의 일반 '다음'이 승인을 대신하지 않는다."""
        _, tool, manager = self._open(name)
        expected = tool.states.load()

        if (type(expected_stage) is not str or type(expected_revision) is not int
                or expected.current_stage != expected_stage or expected.revision != expected_revision):
            raise ValueError("현재 단계·revision을 다시 확인하세요.")

        from .wm_workflow_execution import _SelectedExecutor, _approved_id

        def check_selection():
            tool._check_arguments(tool.arguments)

            if tool.states.load() != expected or tool.cancel.is_set():
                raise ValueError("선택한 실행 단계 또는 취소 상태가 변경되었습니다.")

        check_selection()

        if approval_id is not None:
            _approved_id(tool, approval_id)
        elif expected.status == "awaiting_approval" and approval_progress(manager, expected_stage)["valid_approval_ids"]:
            raise ValueError("승인된 단계 실행에는 정확한 승인 ID를 지정하세요.")

        result = _SelectedExecutor(manager, check_selection, tool).run_next(approval_id=approval_id)
        return {**asdict(result), "module_id": self.module_id, "run_name": name,
                "worker_started": tool.worker_started, "local_only": True}

    def results(self, name):
        """기존 부모 검증을 통과한 결과 참조만 공통 표시 항목으로 반환한다."""
        cli, tool, manager = self._open(name)
        state = tool.states.load()

        if state.status not in {"completed", "blocked_input"}:
            raise ValueError("완료 또는 품질 검토 중단 결과만 조회할 수 있습니다.")

        details = cli.status(tool)
        scope = manager.scope_for(self.module.worker_stage)

        if state.status == "completed":
            output = read_committed_artifact(tool.states.directory, scope).directory
        else:
            task = manager.approvals.get_task(scope)

            if task is None:
                return {"status": state.status, "items": [], "details": details, "local_only": True}

            output = _root(manager._output(task))

        # 각 도메인 검증기를 유지한다. 공통 서비스가 통계·맵을 다시 분석하지 않는다.
        tool.inspect_attempt(output)
        items = {}

        def add(path, kind=None):
            path = Path(path).absolute()
            relative = path.relative_to(output)
            path = _path(path, directory=path.is_dir())
            suffix = path.suffix.lower()
            kind = kind or {".csv": "table", ".png": "image", ".md": "report", ".json": "report"}.get(suffix)

            if kind is not None:
                items[relative.as_posix()] = {"label": relative.as_posix(), "kind": kind,
                                             "path": str(path), "verified": True}

        for key, value in details.items():
            if not key.startswith("local_") or type(value) is not str:
                continue

            path = Path(value)
            if path.is_dir():
                add(path, "artifact_list")
            else:
                add(path)
                children = list(path.parent.iterdir())

                if len(children) > 128:
                    raise ValueError("보고서 표시 항목 상한을 초과했습니다.")

                for child in children:
                    if child.is_file() and child.suffix.lower() in {".csv", ".png", ".md"}:
                        add(child)

        if tool.states.load() != state:
            raise ValueError("결과 조회 중 상태가 변경되었습니다.")

        return {"schema_version": "workflow_service_results_v1", "module_id": self.module_id,
                "status": state.status, "items": list(items.values()), "details": details,
                "notice": "검사 시점의 로컬 결과 참조입니다. 표시 전 파일 검증을 유지해야 합니다.",
                "local_only": True, "llm_transmitted": False}

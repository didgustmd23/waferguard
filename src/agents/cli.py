"""합성 데모용 로컬 CLI: 생성·조회·단계 실행·승인·재개·취소."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets

from .approval_store import ApprovalStore
from .artifacts import _is_link, _root
from .audit import append_event
from .demo_tools import DEMO_VERSION, demo_plan, demo_tools
from .executor import RulesExecutor, _run_lock
from .state import LocalStateStore


def _current_actor():
    """사용자 입력·환경 변수 대신 OS가 제공하는 현재 계정 식별값을 사용한다."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        # NameSamCompatible: DOMAIN\사용자 형태의 현재 보안 주체 이름.
        function = ctypes.WinDLL("secur32", use_last_error=True).GetUserNameExW
        function.argtypes = [ctypes.c_int, wintypes.LPWSTR, ctypes.POINTER(wintypes.ULONG)]
        function.restype = ctypes.c_ubyte
        buffer = ctypes.create_unicode_buffer(1024)
        length = wintypes.ULONG(len(buffer))

        if not function(2, buffer, ctypes.byref(length)):
            raise RuntimeError("현재 OS 계정을 확인할 수 없습니다.")

        return buffer.value

    return f"uid:{os.geteuid()}"


def _print(value):
    """CLI 출력은 로컬 터미널에서만 사용한다."""
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _open_demo(directory):
    """고정 데모 계약과 실행 주체를 확인한다. 실제 전처리 실행은 지원하지 않는다."""
    root = _root(directory)
    paths = (root / "agent_demo.json", root / "state.json", root / "approvals.sqlite3")

    if any(not path.is_file() or _is_link(path) for path in paths):
        raise ValueError("유효한 합성 데모 실행 폴더가 아닙니다.")

    marker = json.loads(paths[0].read_text(encoding="utf-8"))

    if marker != {"demo_version": DEMO_VERSION}:
        raise ValueError("지원하지 않는 데모 계약입니다.")

    states = LocalStateStore(root)
    state = states.load()

    if state.actor_id != _current_actor():
        raise ValueError("실행을 생성한 OS 계정과 현재 계정이 다릅니다.")

    return states


def _executor(states):
    """실행기에는 코드에 고정된 합성 계획·도구만 연결한다."""
    approvals = ApprovalStore(states.directory / "approvals.sqlite3")

    return RulesExecutor(states, approvals, demo_plan(), demo_tools())


def _init_demo(directory):
    """새 폴더에 독립된 합성 데모 상태를 생성한다."""
    root = Path(os.path.abspath(directory))

    if root.exists() or root.is_symlink():
        raise FileExistsError("합성 데모에는 존재하지 않는 새 폴더를 지정하세요.")

    root.parent.mkdir(parents=True, exist_ok=True)
    _root(root.parent)
    states = LocalStateStore(root)
    state = states.create(_current_actor(), policy_version=demo_plan().policy_version)

    with (root / "agent_demo.json").open("x", encoding="utf-8") as stream:
        json.dump({"demo_version": DEMO_VERSION}, stream)

    ApprovalStore(root / "approvals.sqlite3")
    append_event(
        root / "events.jsonl", "run_created", "RUN_CREATED",
        stage=state.current_stage, revision=state.revision,
    )

    _print({"status": state.status, "stage": state.current_stage, "synthetic_demo": True})


# ==========================================
# 승인 화면
# - 현재 실행·단계·도구·출력 대상·영향을 로컬에 표시한다.
# - 사용자가 화면의 일회성 확인 문구를 직접 입력해야 발급한다.
# - 확인 도중 실행이 바뀌면 승인을 발급하지 않는다.
# - 실행 예약·승인 소비는 별도 resume 명령에서 수행한다.
# ==========================================
def _approve(states):
    """대상을 확인하는 대화형 합성 데모 승인 화면."""
    if not os.isatty(0):
        raise ValueError("승인은 대화형 터미널에서 직접 진행하세요.")

    executor = _executor(states)
    state = states.load()

    if state.status != "awaiting_approval" or state.cancel_requested or state.blockers:
        raise ValueError("현재 실행은 승인 가능한 대기 상태가 아닙니다.")

    scope = executor.scope_for(state.current_stage)
    token = f"DEMO-{secrets.token_hex(4).upper()}"

    _print({
        "synthetic_demo": True,
        "run_id": scope.run_id,
        "stage": scope.stage,
        "tool_id": scope.tool_id,
        "plan_id": scope.plan_id,
        "source_id": scope.source_id,
        "config_id": scope.config_id,
        "output_root": str(states.directory / "attempts"),
        "effect": "새 attempt 폴더에 고정 합성 JSON 파일 생성",
        "lifetime_minutes": 30,
    })

    answer = input(f"승인하려면 {token} 를 입력하세요: ")

    if answer != token:
        _print({"code": "APPROVAL_DECLINED"})

        return

    with _run_lock(states.directory) as acquired:
        if not acquired:
            raise RuntimeError("실행기가 사용 중입니다. 상태 확인 후 다시 진행하세요.")

        latest = states.load()

        if latest != state or executor.scope_for(latest.current_stage) != scope:
            raise ValueError("확인 중 실행 상태가 변경되었습니다. 다시 검토하세요.")

        # 기록이 실패하면 승인을 발급하지 않는다.
        append_event(
            states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED",
            stage=state.current_stage, revision=state.revision,
        )
        record = executor.approvals.record_confirmed_approval(scope)

    _print({"approval_id": record.approval_id, "expires_at": record.expires_at.isoformat()})


def _cancel(states):
    """취소 요청을 저장하고 이 실행의 미사용 승인을 폐기한다."""
    # 현재 도구는 같은 프로세스에서 실행한다. 잠금 중 강제 취소는 후속 worker 범위다.
    with _run_lock(states.directory) as acquired:
        if not acquired:
            raise RuntimeError("단계 실행 중입니다. 현재 단계 반환 후 취소를 요청하세요.")

        state = states.load()

        if state.status == "completed":
            raise ValueError("완료된 실행은 취소할 수 없습니다.")

        state.cancel_requested = True
        state.status = "cancelled"
        states.save(state, expected_revision=state.revision)

        approvals = ApprovalStore(states.directory / "approvals.sqlite3")
        approvals.revoke_run(state.run_id)
        append_event(
            states.directory / "events.jsonl", "state_saved", "RUN_CANCEL_REQUESTED",
            stage=state.current_stage, revision=state.revision,
        )

    _print({"status": "cancelled", "code": "RUN_CANCEL_REQUESTED"})


def parse_args(argv=None):
    """명령별 필수 인자를 명시하고 임의 실행 명령은 받지 않는다."""
    parser = argparse.ArgumentParser(description="전처리 에이전트 합성 데모 CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    planning = commands.add_parser("plan", help="합성 계획 또는 데이터 프로파일 단계 조회")
    planning.add_argument("--profile", help="실행하지 않고 검토할 로컬 프로파일 JSON")

    for name in ("init-demo", "status", "step", "approve", "resume", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True)

        if name == "resume":
            command.add_argument("--approval-id", required=True)

    return parser.parse_args(argv)


def main(argv=None):
    """명령을 처리하고 예상 가능한 입력·저장 오류는 한국어로 알린다."""
    args = parse_args(argv)

    try:
        if args.command == "plan":
            if args.profile:
                from src.preprocessing.profiles import load_profile
                from .adapters.registry import ADAPTER_IDS
                from .workflow import WORKFLOW_TEMPLATES

                profile = load_profile(args.profile)

                if profile.adapter_id not in ADAPTER_IDS:
                    raise ValueError("등록되지 않은 데이터 어댑터입니다.")

                _print({
                    "planning_only": True,
                    "source_bound": False,
                    "profile_id": profile.profile_id,
                    "adapter_id": profile.adapter_id,
                    "workflow_id": profile.workflow_id,
                    "config_id": profile.signature,
                    "stages": WORKFLOW_TEMPLATES[profile.workflow_id],
                })
            else:
                _print({"synthetic_demo": True, "plan": demo_plan().to_dict()})
        elif args.command == "init-demo":
            _init_demo(args.run_dir)
        else:
            states = _open_demo(args.run_dir)

            if args.command == "status":
                state = states.load()
                _print({
                    "status": state.status, "stage": state.current_stage,
                    "revision": state.revision, "blockers": state.blockers,
                    "synthetic_demo": True,
                })
            elif args.command == "approve":
                _approve(states)
            elif args.command == "cancel":
                _cancel(states)
            else:
                approval_id = getattr(args, "approval_id", None)
                result = _executor(states).run_next(approval_id=approval_id)
                _print(asdict(result))

                if result.status in {"blocked_input", "failed_runtime"}:
                    return 1
    except (OSError, ValueError, TypeError, RuntimeError, EOFError):
        print("명령을 처리하지 못했습니다. 데모 경로·실행 상태·계정·접근 권한을 확인하세요.")

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

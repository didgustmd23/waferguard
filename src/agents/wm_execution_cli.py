"""WM 계획에 배정한 현재 작업의 승인·한 단계 실행을 위한 로컬 공통 CLI."""

import argparse
import json
import sqlite3

from .wm_module_plan import load_wm_modules
from .wm_run_bindings import load_bindings
from .wm_workflow_execution import dispatch_stage
from .wm_workflow_guidance import recommend_next


def main():
    parser = argparse.ArgumentParser(description="WM 공통 상태 조회·사용자 승인·한 단계 실행")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("status", "next", "approve"):
        sub = commands.add_parser(command)
        sub.add_argument("--pipeline", required=True)
        sub.add_argument("--bindings", required=True)

        if command != "status":
            sub.add_argument("--step", required=True)
            sub.add_argument("--name", required=True)
            sub.add_argument("--stage", required=True)

        if command == "next":
            sub.add_argument("--approval-id")

    args = parser.parse_args()

    try:
        if args.command == "status":
            plan = load_wm_modules(args.pipeline)
            result = recommend_next(plan, load_bindings(args.bindings, plan))
        else:
            result = dispatch_stage(args.pipeline, args.bindings, args.step, args.name, args.stage, args.command,
                                    approval_id=getattr(args, "approval_id", None))

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("status") in {"blocked", "blocked_input", "failed_runtime"} else 0
    except (ValueError, TypeError, OSError, KeyError, RuntimeError, EOFError, sqlite3.Error):
        # 사용자 파일의 내용이나 상세 오류를 고정 결과 코드에 붙이지 않는다.
        print(json.dumps({"status": "blocked", "code": "WM_EXECUTION_REQUEST_BLOCKED",
                          "message": "계획·배정·작업·현재 단계·승인·로컬 기록을 확인하세요. 자동 재시도하지 않습니다.",
                          "local_only": True}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

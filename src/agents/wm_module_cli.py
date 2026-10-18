"""공통 등록부의 목록과 WM 정적 계획을 확인한다. 작업 생성·승인·worker 실행은 하지 않는다."""

import argparse
from dataclasses import asdict
import json

from .agent_module_registry import build_agent_registry
from .wm_module_plan import load_wm_modules


def main():
    parser = argparse.ArgumentParser(description="공통 모듈 목록·WM 실행 계획 검증")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    validate = commands.add_parser("validate")
    validate.add_argument("--pipeline", required=True)
    args = parser.parse_args()

    try:
        if args.command == "list":
            result = {"modules": [asdict(item) for item in build_agent_registry().contracts()],
                      "worker_started": False}
        else:
            plan = load_wm_modules(args.pipeline)
            result = {"status": "plan_validated", "signature": plan.signature,
                      "plan": plan.to_dict(), "worker_started": False}

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, TypeError, OSError, KeyError):
        print(json.dumps({"status": "failed", "code": "WM_MODULE_PLAN_REJECTED",
                          "message": "등록 모듈·버전·증거 연결·구간·설정 형식을 확인하세요.",
                          "worker_started": False}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

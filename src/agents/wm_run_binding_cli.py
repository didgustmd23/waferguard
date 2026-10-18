"""WM 계획과 실제 작업 배정을 읽기 전용으로 대조한다. 생성·승인·실행 명령은 없다."""

import argparse
import json

from .wm_module_plan import load_wm_modules
from .wm_run_bindings import inspect_bindings, load_bindings


def main():
    parser = argparse.ArgumentParser(description="WM 모듈 계획과 기존 실행 결과의 연결 검사")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--bindings", required=True)
    args = parser.parse_args()

    try:
        plan = load_wm_modules(args.pipeline)
        result = inspect_bindings(plan, load_bindings(args.bindings, plan))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "bindings_verified" else 2
    except (ValueError, TypeError, OSError, KeyError, RuntimeError):
        print(json.dumps({"status": "failed", "code": "WM_RUN_BINDING_CHECK_FAILED",
                          "message": "계획·작업 배정·로컬 완료 증거를 확인하세요.",
                          "worker_started": False, "approval_consumed": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

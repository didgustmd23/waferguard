"""WM 다음 행동 안내 전용 CLI. 생성·승인·실행 함수는 호출하지 않는다."""

import argparse
import json

from .wm_module_plan import load_wm_modules
from .wm_run_bindings import load_bindings
from .wm_workflow_guidance import recommend_next


def main():
    parser = argparse.ArgumentParser(description="WM 계획의 다음 작업 안내: 조회만 수행")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--bindings", required=True)
    args = parser.parse_args()

    try:
        plan = load_wm_modules(args.pipeline)
        result = recommend_next(plan, load_bindings(args.bindings, plan))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["status"] == "blocked" else 0
    except (ValueError, TypeError, OSError, KeyError, RuntimeError):
        print(json.dumps({"status": "blocked", "code": "WM_GUIDANCE_FAILED",
                          "message": "계획·작업 배정·로컬 기록을 확인하세요.",
                          "worker_started": False, "approval_consumed": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

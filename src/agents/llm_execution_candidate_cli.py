"""복원한 제안의 로컬 입력 결합·작업 준비 CLI. 실행 승인·worker 시작은 별도다."""

import argparse
import json
import sys

from .llm_execution_candidate import create_candidate, preview_candidate
from .worker_contract import WorkerLimits


def main():
    parser = argparse.ArgumentParser(description="저장 제안을 기존 파이프라인 승인 작업에 연결")
    parser.add_argument("command", choices=("preview", "create"))
    parser.add_argument("--proposal-name", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--name", help="새 승인 작업 이름. 소문자·숫자·밑줄로 48자 이하")
    parser.add_argument("--memory-mib", type=int, required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()

    if args.command == "create" and not args.name:
        parser.error("작업 준비에는 --name으로 새 이름을 지정하세요.")

    try:
        limits = WorkerLimits(args.wall_seconds, args.memory_mib, args.processes)
        preview = preview_candidate(args.proposal_name, args.snapshot_dir, args.profile, limits)

        if args.command == "preview":
            result = preview

        else:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise ValueError("작업 준비 확인은 대화형 터미널에서만 지원합니다.")

            print(json.dumps({"requested_run_name": args.name, **preview}, ensure_ascii=False, indent=2))
            print("입력·프로파일·유효 품질 기준·모듈·코드 지문·자원을 로컬에서 확인하세요.")

            if input("기존 승인 작업의 준비만 진행하려면 CREATE를 입력하세요: ").strip() != "CREATE":
                result = {"status": "declined", "worker_started": False,
                          "model_called": False, "execution_approval_issued": False}
            else:
                result = create_candidate(
                    args.name, args.proposal_name, args.snapshot_dir, args.profile, limits,
                    expected_selection_id=preview["selection_id"], confirmed=True,
                )

    except (EOFError, KeyboardInterrupt):
        result = {"status": "interrupted", "worker_started": False,
                  "automatic_retry": False, "notice": "부분 작업이 남았는지 로컬에서 확인하세요."}
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        print(json.dumps({"status": "failed", "code": "EXECUTION_CANDIDATE_REJECTED",
                          "worker_started": False, "model_called": False,
                          "execution_approval_issued": False, "automatic_retry": False}))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

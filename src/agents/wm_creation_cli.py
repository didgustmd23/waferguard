"""WM 정상 흐름의 작업 준비 CLI. 생성 확인과 실행 승인을 분리한다."""

import argparse
import json
import os
import secrets

from .wm_run_creation import create_and_assign, preview_creation
from .worker_contract import WorkerLimits


def main():
    parser = argparse.ArgumentParser(description="WM EDA·분할·변환·캐시·Dataset 작업 준비·새 배정 파일 생성")
    parser.add_argument("command", choices=("preview", "create"))
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split-config", help="제안 단계만 사용. 생략 시 기존 WM 분할 설정 사용")
    parser.add_argument("--transform-config", help="변환 검사만 사용. 생략 시 기존 WM 변환 설정 사용")
    parser.add_argument("--cache-config", help="캐시 준비만 사용. 생략 시 기존 WM 캐시 설정 사용")
    parser.add_argument("--canonical-dir", help="첫 EDA 전용 canonical 폴더. 원본은 보존하고 새 복사본 생성")
    parser.add_argument("--include-examples", action="store_true", default=None, help="첫 EDA 전용 예시 맵 그림 생성 요청")
    parser.add_argument("--max-group-ratio", type=float, help="첫 EDA 전용 연결 그룹 검토 상한. 기본 0.2")
    parser.add_argument("--batch-size", type=int, help="Dataset 전용. 기본 32")
    parser.add_argument("--max-input-bytes", type=int, help="첫 EDA canonical 복사 상한은 필수 지정. Dataset 캐시 복사 기본 8 GiB")
    parser.add_argument("--max-runtime-bytes", type=int, help="Dataset 전용 설치 런타임 복사 상한. 기본 8 GiB")
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--wall-seconds", type=int, default=300)
    parser.add_argument("--max-output-bytes", type=int, help="기본값: EDA/제안 64 MiB·확정 2 GiB·변환 256 MiB·캐시 8 GiB·Dataset 1 MiB")
    args = parser.parse_args()

    try:
        from .wm_dataset_creation import DEFAULT_OPTIONS

        requested = {key: getattr(args, key) for key in DEFAULT_OPTIONS}
        dataset_options = None
        canonical_options = None
        canonical_requested = any(value is not None for value in
                                  (args.canonical_dir, args.include_examples, args.max_group_ratio))

        if canonical_requested:
            canonical_options = {"max_input_bytes": args.max_input_bytes,
                                 "include_examples": args.include_examples is True,
                                 "max_group_ratio": 0.2 if args.max_group_ratio is None else args.max_group_ratio}
            # 같은 옵션 이름을 쓰더라도 canonical 복사 예산을 Dataset 설정으로 전달하지 않는다.
            requested["max_input_bytes"] = None

        if any(value is not None for value in requested.values()):
            dataset_options = {key: DEFAULT_OPTIONS[key] if value is None else value
                               for key, value in requested.items()}

        preview = preview_creation(args.pipeline, args.bindings, args.step, args.name, args.split_config,
                                   WorkerLimits(args.wall_seconds, args.memory_mib, args.processes), args.max_output_bytes,
                                   transform_config=args.transform_config, cache_config=args.cache_config,
                                   dataset_options=dataset_options, canonical_dir=args.canonical_dir,
                                   canonical_options=canonical_options)
        print(json.dumps({"status": "creation_preview", "preview_id": preview.signature,
                          "selection": preview.packet, "notice": "작업 준비·새 배정만 생성합니다. 실행 승인이 아닙니다."},
                         ensure_ascii=False, indent=2))

        if args.command == "preview":
            return 0

        if not os.isatty(0):
            raise ValueError("생성 확인은 대화형 터미널에서 직접 진행하세요.")

        token = "CREATE-" + secrets.token_hex(4).upper()

        if input(f"작업 준비·새 배정 생성을 확인하려면 {token} 을 입력하세요: ") != token:
            print(json.dumps({"status": "declined", "worker_started": False}, ensure_ascii=False))
            return 2

        print(json.dumps(create_and_assign(preview, confirmed=True), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, OSError, KeyError, RuntimeError, EOFError):
        print(json.dumps({"status": "failed", "code": "WM_CREATION_FAILED", "worker_started": False,
                          "message": "작업 이름·선행 결과·설정·기록을 확인하세요. 부분 생성 작업과 기록은 보존되며 자동 재시도하지 않습니다."},
                         ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""합성 또는 사용자 문장의 미리보기·제안 호출. 파일 입력·출력·실행 옵션은 없다."""

import argparse
import json
import sys

from .llm_proposal import (
    EXAMPLES, build_example_payload, build_request_payload, propose_example, propose_request,
)
from .llm_config import DEFAULT_CONFIG, load_local_config
from .local_ollama import LocalProposalError


def _interactive_request(config, *, preview_only):
    """셸 인자 대신 터미널에서 문장을 받고 실제 전송 내용을 보여 준 뒤 확인한다."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise LocalProposalError("INTERACTIVE_TERMINAL_REQUIRED")

    print("입력한 문장 자체가 모델에 전달됩니다. 원본·통계·비밀정보를 넣지 마세요.")
    print("파일·보고서 자동 첨부, 승인 발급, 전처리 실행은 하지 않습니다.")

    try:
        request = input("요청 문장: ")
        payload = build_request_payload(request, config)
    except (ValueError, UnicodeError) as error:
        raise LocalProposalError("REQUEST_TEXT_INVALID") from error

    preview = {"status": "preview", "endpoint": config.endpoint, "payload": payload,
               "model_called": False, "worker_started": False, "synthetic_only": False}

    if preview_only:
        return preview

    print(json.dumps(preview, ensure_ascii=False, indent=2))

    if input("위 내용을 로컬 모델에 전달하려면 SEND를 입력하세요: ").strip() != "SEND":
        return {"status": "not_sent", "model_called": False, "worker_started": False}

    result = propose_request(request, config).to_dict()
    result.update(status="proposal_validated", model_called=True, synthetic_only=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="로컬 Qwen 실행안 제안 검사")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", choices=tuple(EXAMPLES))
    source.add_argument("--interactive-request", action="store_true", help="터미널에서 직접 요청 입력")
    parser.add_argument("--preview", action="store_true", help="실제 전송 내용을 표시하고 호출하지 않음")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="로컬 모델 JSON 설정 경로")
    parser.add_argument("--confirm-local-model-call", action="store_true", help="로컬 모델 호출 의사 확인")
    args = parser.parse_args()

    try:
        config = load_local_config(args.config)
    except (OSError, ValueError, UnicodeError):
        print(json.dumps({"status": "failed", "code": "MODEL_CONFIG_INVALID",
                          "model_called": False, "worker_started": False}))
        return 1

    if args.interactive_request:
        try:
            result = _interactive_request(
                config, preview_only=args.preview or not args.confirm_local_model_call,
            )
        except (EOFError, KeyboardInterrupt):
            # 통신 도중 중단됐을 수도 있어 전송하지 않았다고 단정하지 않는다.
            result = {"status": "interrupted", "delivery_unknown": True,
                      "worker_started": False, "automatic_retry": False}
        except LocalProposalError as error:
            print(json.dumps({"status": "failed", "code": error.code,
                              "worker_started": False, "automatic_retry": False}, ensure_ascii=False))
            return 1

    elif args.preview:
        result = {
            "status": "preview", "endpoint": config.endpoint,
            "payload": build_example_payload(args.example, config), "model_called": False,
            "worker_started": False,
        }

    else:
        if not args.confirm_local_model_call:
            parser.error("호출 전에 --preview로 확인하고 --confirm-local-model-call을 지정하세요.")

        try:
            result = propose_example(args.example, config).to_dict()
            result.update(status="proposal_validated", model_called=True, synthetic_only=True)
        except LocalProposalError as error:
            print(json.dumps({"status": "failed", "code": error.code,
                              "worker_started": False, "automatic_retry": False}, ensure_ascii=False))
            return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

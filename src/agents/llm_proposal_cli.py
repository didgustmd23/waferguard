"""요청의 미리보기·제안·선택적 로컬 보관. 데이터 첨부·승인·실행은 하지 않는다."""

import argparse
import json
import sys

from src.preprocessing.module_contracts import check_id
from .llm_proposal import (
    EXAMPLES, build_example_payload, build_request_payload, propose_example, propose_request,
)
from .llm_config import DEFAULT_CONFIG, load_local_config
from .local_ollama import LocalProposalError
from .llm_proposal_store import load_proposal, save_proposal


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


def _save_selected_result(name, result):
    """실행안을 보여 준 뒤 별도 저장 확인을 받는다. 저장은 실행 승인이 아니다."""
    if result.get("status") != "proposal_validated":
        return result

    if result["proposal"]["status"] != "proposed":
        result["save_status"] = "not_saved_clarification"
        return result

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("제안·지문만 로컬에 보관합니다. 요청 원문 별도 저장·승인 발급·전처리 실행은 없습니다.")

    if input("표시한 실행안을 저장하려면 SAVE를 입력하세요: ").strip() != "SAVE":
        result["save_status"] = "declined"
        return result

    receipt = save_proposal(name, result)
    result["saved_proposal"] = receipt
    return result


def main():
    parser = argparse.ArgumentParser(description="로컬 Qwen 실행안 제안 검사")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", choices=tuple(EXAMPLES))
    source.add_argument("--interactive-request", action="store_true", help="터미널에서 직접 요청 입력")
    source.add_argument("--load-name", help="저장된 제안을 현재 계약으로 재검증해 조회")
    parser.add_argument("--preview", action="store_true", help="실제 전송 내용을 표시하고 호출하지 않음")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="로컬 모델 JSON 설정 경로")
    parser.add_argument("--save-name", help="제안 후 사용자 확인을 받아 새 이름으로 로컬 보관")
    parser.add_argument("--confirm-local-model-call", action="store_true", help="로컬 모델 호출 의사 확인")
    args = parser.parse_args()

    if args.load_name:
        if args.preview or args.confirm_local_model_call or args.save_name:
            parser.error("복원 조회에는 호출·미리보기·저장 옵션을 함께 사용할 수 없습니다.")

        try:
            result = load_proposal(args.load_name)
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            print(json.dumps({"status": "failed", "code": "SAVED_PROPOSAL_REJECTED",
                              "model_called": False, "worker_started": False}))
            return 1

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.save_name:
        try:
            check_id(args.save_name)
        except ValueError:
            parser.error("저장 이름은 소문자로 시작하는 64자 이하 영문·숫자·밑줄 ID여야 합니다.")

        if args.preview or not args.confirm_local_model_call:
            parser.error("미리보기는 저장하지 않습니다. 호출 확인 옵션과 함께 사용하세요.")

        if not sys.stdin.isatty() or not sys.stdout.isatty():
            parser.error("제안 저장은 사용자 확인이 가능한 터미널에서만 지원합니다.")

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

    if args.save_name:
        try:
            result = _save_selected_result(args.save_name, result)
        except (EOFError, KeyboardInterrupt):
            result["save_status"] = "interrupted"
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            print(json.dumps({"status": "failed", "code": "PROPOSAL_SAVE_FAILED",
                              "worker_started": False, "automatic_retry": False}))
            return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

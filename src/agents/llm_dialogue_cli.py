"""로컬 제안 대화의 전송 확인·답변 입력. 실행 승인은 별도 기능이다."""

import argparse
import json
import sys

from src.preprocessing.module_contracts import check_id
from .llm_config import DEFAULT_CONFIG, load_local_config
from .llm_proposal import EXAMPLES
from .llm_proposal_cli import _save_selected_result
from .local_ollama import LocalProposalError


def main():
    parser = argparse.ArgumentParser(description="LangGraph 로컬 전처리 제안 대화")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", choices=tuple(EXAMPLES))
    source.add_argument("--interactive-request", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--save-name", help="완성 제안을 별도 SAVE 확인 후 새 이름으로 저장")
    args = parser.parse_args()

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(json.dumps({"status": "failed", "code": "INTERACTIVE_TERMINAL_REQUIRED"}))
        return 1

    session = None

    try:
        if args.save_name:
            check_id(args.save_name)

        # 선택한 대화 CLI에서만 추가 의존성을 읽는다. 기존 CLI의 독립성은 유지한다.
        from .llm_dialogue import ProposalDialogue

        config = load_local_config(args.config)
        session = ProposalDialogue(config)
        print("입력 문장과 이전 질문·답변이 로컬 모델에 전달됩니다. 원본·통계·비밀정보를 넣지 마세요.")
        print("대화는 메모리에만 유지합니다. 답변·SEND·제안 저장은 전처리 실행 승인이 아닙니다.")
        text = EXAMPLES[args.example] if args.example else input("요청 문장: ")

        while True:
            preview = session.preview(text)
            print(json.dumps(preview, ensure_ascii=False, indent=2))

            if input("위 누적 내용을 로컬 모델에 전달하려면 SEND를 입력하세요: ").strip() != "SEND":
                print(json.dumps({"status": "dialogue_cancelled", "worker_started": False,
                                  "execution_approved": False}))
                return 0

            result = session.send(text, confirmed=True)
            print(json.dumps(result, ensure_ascii=False, indent=2))

            if result["status"] != "awaiting_answer":
                if args.save_name and result["status"] == "proposal_validated":
                    saved = _save_selected_result(args.save_name, result)
                    print(json.dumps(saved, ensure_ascii=False, indent=2))

                return 0

            # 질문은 위 JSON에 이스케이프하여 표시한다. 모델 출력으로 터미널을 제어하지 않는다.
            text = input("위 질문에 대한 답변: ")
    except (EOFError, KeyboardInterrupt):
        print(json.dumps({"status": "interrupted", "delivery_unknown": True,
                          "worker_started": False, "automatic_retry": False}))
        return 1
    except LocalProposalError as error:
        print(json.dumps({"status": "failed", "code": error.code,
                          "worker_started": False, "automatic_retry": False}))
        return 1
    except ImportError:
        print(json.dumps({"status": "failed", "code": "DIALOGUE_DEPENDENCY_MISSING"}))
        return 1
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"status": "failed", "code": "DIALOGUE_INPUT_OR_SAVE_REJECTED",
                          "worker_started": False, "automatic_retry": False}))
        return 1
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())

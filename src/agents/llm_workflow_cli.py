"""CSV 실행안 연결 검사. 전송마다 확인하며 저장·실행·승인 DB에 연결하지 않는다."""

import argparse
import json
from pathlib import Path
import sys

from .llm_config import DEFAULT_CONFIG, load_local_config
from .local_ollama import LocalProposalError
from .workflow_service_registry import CSV_IDS


EXAMPLES = {
    "prepare": "선택한 예제 규칙 그대로 시간순 분할, 개발 구간 내보내기, Train 후보 산출, 저장 기준 적용을 제안하세요.",
    "clarify": "전처리하고 싶지만 어떤 작업을 할지는 아직 정하지 않았습니다.",
}


def main():
    parser = argparse.ArgumentParser(description="CSV workflow 제안·확인 질문 연결 검사 (실행 없음)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", choices=tuple(EXAMPLES))
    source.add_argument("--interactive-request", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--use-example-settings", action="store_true",
                        help="시간순 70/15/15·결측률 50%%·중앙값 대치 예제의 전체 규칙을 명시적으로 선택")
    args = parser.parse_args()

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(json.dumps({"status": "failed", "code": "INTERACTIVE_TERMINAL_REQUIRED"}))
        return 1

    session = None

    try:
        from .llm_workflow_dialogue import WorkflowProposalDialogue

        selected = {}

        if args.use_example_settings:
            path = Path(__file__).resolve().parents[2] / "configs/agents/llm/workflow_proposal_example.json"
            example = json.loads(path.read_text(encoding="utf-8"))
            selected = {step["module_id"]: step["settings"] for step in example["workflow"]["steps"]}

        session = WorkflowProposalDialogue(load_local_config(args.config),
                                           allowed_modules=tuple((item, "1.0.0") for item in CSV_IDS),
                                           selected_settings=selected)
        print("공개 규칙과 아래 문장만 로컬 모델에 전달합니다. 데이터·경로·비밀정보를 입력하지 마세요.")
        print("설정은 세션 동안 고정됩니다. 변경은 새 대화가 필요하며 이 기능은 작업을 생성하지 않습니다.")
        text = EXAMPLES[args.example] if args.example else input("요청 문장: ")

        while True:
            print(json.dumps(session.preview(text), ensure_ascii=False, indent=2))

            if input("실제 전달 자료를 확인했다면 SEND를 입력하세요: ").strip() != "SEND":
                print(json.dumps({"status": "dialogue_cancelled", "worker_started": False}))
                return 0

            result = session.send(text, confirmed=True)
            print(json.dumps(result, ensure_ascii=False, indent=2))

            if result["status"] != "awaiting_answer":
                return 0

            text = input("확인 질문에 대한 답변: ")
    except (EOFError, KeyboardInterrupt):
        print(json.dumps({"status": "interrupted", "delivery_unknown": True, "automatic_retry": False}))
        return 1
    except LocalProposalError as error:
        print(json.dumps({"status": "failed", "code": error.code, "automatic_retry": False}))
        return 1
    except (ValueError, TypeError, KeyError, OSError, ImportError):
        print(json.dumps({"status": "failed", "code": "WORKFLOW_DIALOGUE_INPUT_OR_DEPENDENCY_REJECTED"}))
        return 1
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())

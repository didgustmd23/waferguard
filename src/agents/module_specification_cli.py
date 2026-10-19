"""통합 모듈 명세의 읽기 전용 조회. 작업 생성·모델·승인·worker 호출은 없다."""

import argparse
import json

from .agent_module_registry import build_agent_registry


def main():
    parser = argparse.ArgumentParser(description="공통 UI·서비스용 모듈 명세 조회")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--audience", choices=("all", "ui", "llm"), default="all")
    listing.add_argument("--details", action="store_true")
    showing = commands.add_parser("show")
    showing.add_argument("--module", required=True)
    showing.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    try:
        registry = build_agent_registry()

        if args.command == "show":
            result = {"module": registry.specification(args.module, args.version)}
        else:
            modules = registry.specifications(audience=args.audience)
            if not args.details:
                modules = [{"module_id": item["contract"]["module_id"],
                            "version": item["contract"]["version"], "title": item["title"],
                            "execution_kind": item["execution"]["kind"],
                            "settings_form_ready": item["settings"]["form_ready"],
                            "result_views": item["result_views"]} for item in modules]

            result = {"module_count": len(modules), "audience": args.audience, "modules": modules}

        result.update(status="catalogue_validated", declaration_only=True,
                      model_called=False, worker_started=False, approval_issued=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"status": "failed", "code": "MODULE_SPECIFICATION_INVALID",
                          "worker_started": False}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

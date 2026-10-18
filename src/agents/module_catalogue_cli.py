"""고정 사용자 모듈 목록·명세 검사 CLI. 임의 모듈 등록·import·데이터 실행 없음."""

import argparse
import json

from .module_catalogue import MODULE_ID, VERSION, DESCRIPTOR_SHA256, inspect_module


def main():
    parser = argparse.ArgumentParser(description="고정 사용자 모듈 등록 명세 검사")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    check = commands.add_parser("check")
    check.add_argument("--module-id", required=True)
    check.add_argument("--version", required=True)
    args = parser.parse_args()

    try:
        module_id, version = (MODULE_ID, VERSION) if args.command == "list" else (args.module_id, args.version)
        contract = inspect_module(module_id, version)
        result = {"status": "module_spec_verified", "module_id": contract.module_id,
                  "version": contract.version, "kind": contract.kind,
                  "descriptor_sha256": DESCRIPTOR_SHA256,
                  "execution_scope": "synthetic_local_only", "native_execution_enabled": False,
                  "module_imported": False, "data_read": False}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print("모듈 명세 검사를 완료하지 못했습니다. 고정 ID·버전·소스·명세 해시를 확인하세요.")
        print("변경한 코드를 자동 import하거나 검사 기준을 자동 갱신하지 않습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

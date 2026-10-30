"""실험적 Graph RAG 색인 관리 명령. 사용자가 명시적으로 실행할 때만 지식집을 검사하거나 색인을 만든다.

    python -m src.agents.rag_index_cli verify tabular
    python -m src.agents.rag_index_cli build tabular [--previous <이전 색인 폴더 이름>]

지식집·색인 위치는 프로필 고정 경로다(임의 경로를 받지 않는다). build는 기존 작업 설정(configs/agents/llm/local_ollama.json)의
loopback Ollama에 문서마다 1회씩 임베딩을 보내며, 실패하면 재시도 없이 멈추고 정상 색인 이름을 만들지 않는다.
이미 같은 이름의 색인이 있으면 덮어쓰지 않는다. 출처 해시(검토 시점 값)는 이 명령이 갱신하지 않는다.
"""

import argparse
import sys

from . import rag_index, rag_knowledge, rag_tabular
from .llm_config import load_local_config
from .rag_embed import EmbedError, OllamaEmbedder


PROFILES = {"tabular": rag_tabular}


def main(argv=None):
    parser = argparse.ArgumentParser(description="실험적 Graph RAG 지식집 검사·색인 구축(명시 실행 전용)")
    parser.add_argument("command", choices=("verify", "build"))
    parser.add_argument("profile", choices=sorted(PROFILES))
    parser.add_argument("--previous", help="벡터를 재사용할 이전 색인 폴더 이름(같은 모델·같은 임베딩 입력만 재사용)")
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]

    if args.previous is not None and not rag_index.is_index_name(args.previous):
        print("오류: 이전 색인 이름 형식이 올바르지 않습니다.")
        return 2

    try:
        pack = rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
    except rag_knowledge.KnowledgeError as error:
        print("지식집 검사 실패:", error.code, ",".join(error.targets))
        return 1

    print("지식집:", pack["pack"]["pack_id"], pack["pack"]["version"], "항목", len(pack["knowledge"]["items"]),
          "관계", len(pack["graph"]["edges"]), "· 색인 이름:", rag_index.index_name(pack))

    if pack["changed_sources"]:
        print("검토 이후 바뀐 비강제 출처:", ",".join(pack["changed_sources"]))

    if args.command == "verify":
        try:
            rag_index.load_index(profile.INDEX_ROOT, pack)
        except rag_index.RagIndexError as error:
            print("색인 검사 실패:", error.code)
            return 1

        print("색인 검사 통과")
        return 0

    embedder = OllamaEmbedder(load_local_config(), limit=len(pack["knowledge"]["items"]))

    try:
        result = rag_index.build_index(profile.INDEX_ROOT, pack, embedder, previous=args.previous)
    except (EmbedError, rag_index.RagIndexError) as error:
        print("색인 구축 실패(재시도하지 않음):", error.code, "· 임베딩 호출", embedder.calls["embed"], "회")
        return 1

    print("색인 구축 완료:", result["name"], "· 새 임베딩", result["embedded"], "· 재사용", result["reused"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

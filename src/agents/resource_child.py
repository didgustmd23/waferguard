"""자원 제한 검사 전용 고정 payload. 데이터·LLM·네트워크를 사용하지 않는다."""

import subprocess
import sys
import time


def main():
    mode = sys.argv[1]

    if mode == "normal":
        return 0

    if mode == "sleep":
        time.sleep(30)
        return 0

    if mode == "memory":
        # 최대 256MiB만 요청한다. 호스트의 메모리를 무제한 소비하지 않는다.
        chunks = []

        try:
            for _ in range(32):
                chunks.append(bytearray(8 * 1024 * 1024))
        except MemoryError:
            return 72

        return 0

    if mode in {"spawn", "descendant"}:
        try:
            child = subprocess.Popen(
                [sys.executable, "-I", "-S", "-B", "-c", "import time; time.sleep(30)"],
                creationflags=0x08000000,
            )
        except OSError:
            return 73

        if mode == "descendant":
            # 루트가 먼저 끝나도 자식은 부모 실행기의 Job 정리 대상이어야 한다.
            return 0

        child.terminate()
        child.wait(timeout=5)
        return 0

    return 74


if __name__ == "__main__":
    raise SystemExit(main())

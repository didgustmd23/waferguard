"""기존 worker 진입점 호환용. 새 전송은 범용 진입점을 사용한다."""

from .pickle_import_worker import execute_request, main


if __name__ == "__main__":
    raise SystemExit(main())

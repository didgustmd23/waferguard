"""완료된 CSV 전처리 결과의 로컬 전달 묶음 생성·검사. 실행·재산출·업로드 없음."""

import argparse
import json
from pathlib import Path
import re
import uuid

from . import preparation_pipeline_cli as pipeline
from . import preparation_apply_cli as application
from .artifacts import _root, _fingerprint
from .csv_handoff_contract import PAYLOAD_FILES, PREPARED_FILES, MAX_BUNDLE_BYTES, feature_mapping, verify_bundle
from .executor import _run_lock
from .split_evidence import read_metadata
from .worker_paths import _path


HANDOFF_README = """# CSV 전처리 전달 묶음

Train에서 산출하고 별도 승인한 저장 기준을 Train·Validation에 적용한 결과입니다.
원본 CSV·Test·승인 DB·실행 가능한 스크립트·pickle·모델은 포함하지 않습니다.

## 파일 읽기

- prepared/train_features.csv와 validation_features.csv: 수치 특징과 고정된 출력 컬럼 순서.
- prepared/train_rows.json과 validation_rows.json: 원본 행 위치와 라벨·ID·그룹·시간의 문자열 값.
- preprocessor.json: Train에서 산출한 제거·대치·스케일·범주 기준. 적용 시 다시 산출하지 않습니다.
- feature_mapping.json: 출력 컬럼과 원본 특징의 연결. 순위·중요도를 뜻하지 않습니다.
- provenance.json과 prepared/application.json: 출처 식별값과 적용 기록.
- HANDOFF_COMPLETE.json: 파일 목록·크기·SHA-256. 인증서나 외부 승인 증거가 아닙니다.

특징 CSV의 i번째 행과 rows.json의 i번째 metadata·row_positions가 대응합니다.
row_positions는 원래 입력의 위치이므로 현재 CSV의 인덱스로 사용하거나 재정렬하지 마세요.
Train과 Validation은 각각의 rows.json으로 연결해야 합니다.

```python
import json
from pathlib import Path
import pandas as pd

bundle = Path("data/processed/agents/handoffs/선택한_묶음_이름")
rows = json.loads((bundle / "prepared/train_rows.json").read_text(encoding="utf-8"))
X_train = pd.read_csv(bundle / "prepared/train_features.csv")
metadata = pd.DataFrame(rows["metadata"], columns=rows["metadata_columns"])
assert len(X_train) == len(metadata) == len(rows["row_positions"])
```

target 컬럼명은 preprocessor.json의 roles.target에서 확인합니다. target이 null이면 라벨이 없습니다.
라벨·ID는 문자열입니다. 숫자 라벨이 필요하면 담당 프로젝트의 허용 값·매핑을 확인하고
명시적으로 변환하세요. SECOM의 라벨 매핑을 다른 데이터에 자동 적용하지 않습니다.
위 예시를 실행하기 전 전달 검사 CLI로 묶음을 확인하세요. 모델 학습은 예시에 포함하지 않습니다.

## 검사와 보안 범위

같은 프로젝트 환경에서 python -m src.agents.csv_handoff_cli verify --name 선택한_묶음_이름 을 실행합니다.
검사는 파일 해시·수치 형식·특징 순서·행 연결의 정합성을 확인합니다. 원본 대비 통계의 정답,
라벨 진실성·최종 성능·배포 승인·원래 승인 이력 전체를 독립 인증하지 않습니다.
로컬 관리자가 파일과 완료 기록을 함께 고칠 수 있으므로 해시는 암호학적 진본 인증이 아닙니다.

이 묶음에는 실제 특징·라벨·식별자·처리 통계가 포함됩니다. 로컬 데이터로 관리하고 LLM·외부
추적 서비스·Git에 보내지 마세요. 팀원 전달도 별도의 허용된 공유 방식으로 수행하세요.
"""


def _directory(name, *, create_parent=False):
    """자동 공유를 피하고 프로젝트의 데이터 폴더 아래에만 내보낸다."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("전달 이름은 소문자로 시작하는 영문·숫자·밑줄·하이픈 1~48자여야 합니다.")

    root = Path(__file__).resolve().parents[2]

    for part in ("data", "processed", "agents", "handoffs"):
        root = root / part

        if create_parent:
            root.mkdir(exist_ok=True)

        _root(root)

    return root / name


def _write_json(path, value, maximum=8 * 1024 * 1024):
    """메타데이터도 쓰기 전에 조각별 바이트 상한을 적용한다."""
    size = 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, indent=2)

    with path.open("xb") as stream:
        for chunk in encoder.iterencode(value):
            raw = chunk.encode("utf-8")
            size += len(raw)

            if size > maximum:
                raise ValueError("전달 메타데이터의 크기 상한을 초과했습니다.")

            stream.write(raw)

    return size


def _copy_checked(source, destination, budget):
    """고정 파일만 스트리밍 복사한다. 합계 상한을 넘기거나 일부만 잘라 성공하지 않는다."""
    source = _path(source, directory=False)

    if source.stat().st_size > budget:
        raise ValueError("전달 묶음의 합계 크기 상한을 초과했습니다.")

    expected = _fingerprint(source)
    copied = 0

    with source.open("rb") as reader, destination.open("xb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            copied += len(chunk)

            if copied > budget:
                raise ValueError("복사 중 전달 크기 상한을 초과했습니다.")

            writer.write(chunk)

    if _fingerprint(destination) != expected or _fingerprint(_path(source, directory=False)) != expected:
        raise ValueError("복사 중 원본 또는 전달 파일이 변경되었습니다.")

    return copied


def export_bundle(pipeline_name, name, maximum, *, confirmed=False):
    """완료 증거를 재검증한 후 새 로컬 묶음만 생성한다. worker를 시작하지 않는다."""
    if not confirmed:
        raise ValueError("실제 데이터가 포함된 로컬 전달 묶음 생성을 명시적으로 확인하세요.")

    if type(maximum) is not int or not 0 < maximum <= MAX_BUNDLE_BYTES:
        raise ValueError("전달 합계 상한은 1 바이트~2 GiB 사이여야 합니다.")

    source_root, record, candidate_tool = pipeline.open_run(pipeline_name)

    with _run_lock(source_root) as acquired:
        if not acquired:
            raise ValueError("통합 실행의 다른 명령이 진행 중입니다.")

        source_root, record, candidate_tool = pipeline.open_run(pipeline_name)

        if (source_root / "CANCEL_REQUESTED").exists():
            raise ValueError("취소 요청이 있는 실행은 전달하지 않습니다.")

        tool = pipeline._application(source_root, record, candidate_tool)

        if tool is None:
            raise ValueError("후보 검토와 적용을 완료한 통합 실행이 필요합니다.")

        status = application.status(tool)

        if status.get("application_status") != "applied" or status["status"] != "completed":
            raise ValueError("적용 완료·파일 검증을 통과한 실행만 전달할 수 있습니다.")

        prepared = _root(status["local_result_directory"])
        target = _directory(name, create_parent=True)

        if target.exists() or target.is_symlink():
            raise FileExistsError("기존 전달 묶음은 덮어쓰지 않습니다. 새 이름을 지정하세요.")

        # 최종 경로와 임시 경로는 검증된 동일 부모 아래에만 생성한다.
        parent = _root(target.parent)
        partial = parent / (name + ".partial-" + uuid.uuid4().hex)
        partial.mkdir()
        (partial / "prepared").mkdir()
        total = 0

        for filename in PREPARED_FILES:
            total += _copy_checked(prepared / filename, partial / "prepared" / filename, maximum - total)

        total += _copy_checked(tool.contract.candidate.directory / "preprocessor.json", partial / "preprocessor.json", maximum - total)
        saved = read_metadata(partial / "preprocessor.json")
        report = read_metadata(partial / "prepared/application.json")
        total += _write_json(partial / "feature_mapping.json", {"schema_version": "feature_mapping_v1",
                    "features": feature_mapping(saved["state"], saved["output_columns"])}, min(maximum - total, 8 * 1024 * 1024))
        total += _write_json(partial / "provenance.json", {
            "schema_version": "csv_handoff_provenance_v1", "pipeline_name": pipeline_name,
            "pipeline_signature": record["plan_signature"], "candidate_id": tool.contract.candidate.signature,
            "apply_run_id": tool.states.load().run_id, "apply_plan_id": tool.plan.plan_id,
            "preprocessor_sha256": tool.contract.candidate.state_sha256, "application_binding": report["binding"],
            "test_used": False, "model_trained": False,
        }, min(maximum - total, 8 * 1024 * 1024))

        if total + len(HANDOFF_README.encode("utf-8")) > maximum:
            raise ValueError("안내서를 포함한 전달 크기 상한을 초과했습니다.")

        with (partial / "README.md").open("x", encoding="utf-8", newline="") as stream:
            stream.write(HANDOFF_README)

        files = {relative: _fingerprint(partial / relative) for relative in PAYLOAD_FILES}
        total = sum(value["size"] for value in files.values())

        if total > maximum:
            raise ValueError("안내·메타데이터를 포함한 전달 합계 상한을 초과했습니다.")

        # 복사 전후 모두 원래 커밋·상태·운송 증거를 확인한다.
        if application.status(tool) != status or (source_root / "CANCEL_REQUESTED").exists():
            raise ValueError("내보내기 중 실행 상태 또는 취소 여부가 변경되었습니다.")

        _write_json(partial / "HANDOFF_COMPLETE.json", {"schema_version": "csv_preparation_handoff_v1",
                    "files": files, "payload_bytes": total, "max_bundle_bytes": maximum}, min(maximum - total, 65536))
        verify_bundle(partial)

        if (target.exists() or target.is_symlink() or _root(target.parent) != parent
                or (source_root / "CANCEL_REQUESTED").exists()):
            raise ValueError("전달 대상 경로가 변경되었습니다.")

        partial.rename(target)

    return {"status": "handoff_created", "directory": str(target), "worker_started": False,
            "refit": False, "test_used": False, "notice": "로컬 전달 복사본입니다. 자동 공유·Git 업로드는 하지 않습니다."}


def main():
    parser = argparse.ArgumentParser(description="CSV 전처리 전달 묶음의 로컬 생성·검증")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--pipeline-name", required=True)
    export.add_argument("--name", required=True)
    export.add_argument("--max-bundle-bytes", type=int, default=128 * 1024 * 1024)
    export.add_argument("--confirm-local-export", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--name", required=True)
    args = parser.parse_args()

    try:
        if args.command == "export":
            result = export_bundle(args.pipeline_name, args.name, args.max_bundle_bytes, confirmed=args.confirm_local_export)
        else:
            result = verify_bundle(_directory(args.name))

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print("전달 명령을 완료하지 못했습니다. 완료 실행·파일 계약·새 이름·상한·권한을 확인하세요.")
        print("기존 결과는 보존합니다. 생성 중단 시 남은 .partial 폴더는 전달 완료 결과가 아닙니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

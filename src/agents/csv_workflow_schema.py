"""CSV 승인 작업의 공통 입력 폼. 계산 규칙은 기존 JSON·코어 검증기를 사용한다."""


def creation_schema(module_id):
    def field(name, title, kind="text", default="", **bounds):
        return {"name": name, "title": title, "kind": kind, "default": default,
                "help": "기존 완료 작업·설정만 선택합니다. 미리보기는 승인이나 실행이 아닙니다.", **bounds}

    def budget(name, title, default):
        return field(name, title, "integer", default, minimum=1, maximum=1024 ** 3)

    name = field("name", "새 작업 이름")
    split = field("split_name", "완료된 CSV 분할 작업 이름")
    export = field("export_name", "완료된 CSV 내보내기 작업 이름")
    pipeline = field("pipeline_name", "전처리 통합 작업 이름")
    fields = {
        "csv_split_review": [name, field("source", "원본 CSV", "csv_file"),
            field("profile", "분할 프로파일 JSON", "json_file"),
            budget("max_input_bytes", "입력 복사 상한 (byte)", 8 * 1024 ** 2),
            field("allow_feature_findings", "특징 결측·상수 통지만 무시", "boolean", False)],
        "csv_partition_export": [name, split, budget("max_partition_bytes", "구간 출력 상한 (byte)", 32 * 1024 ** 2)],
        "csv_prepare_candidate": [name, split, export,
            field("pipeline", "전처리 모듈 계획 JSON", "json_file", "configs/agents/pipelines/tabular_preparation_example.json"),
            budget("max_candidate_bytes", "후보 출력 상한 (byte)", 4 * 1024 ** 2),
            budget("max_partition_bytes", "적용 출력 상한 (byte)", 32 * 1024 ** 2)],
        "csv_prepare_apply": [pipeline],
        "csv_local_handoff": [name, pipeline, budget("max_bundle_bytes", "전달 묶음 상한 (byte)", 128 * 1024 ** 2)],
    }
    if module_id not in fields:
        raise ValueError("등록되지 않은 CSV 작업입니다.")

    return {"supported": True, "fields": fields[module_id], "uses_job": module_id != "csv_local_handoff",
            "notice": "기존 CSV CLI·검증기를 사용합니다. 저장 기준 적용 시 자원은 후보 작업과 같아야 합니다. "
                      "CSV 격리 실행은 Windows Python 3.10·3.12를 지원합니다. 새 작업과 별도 승인이 필요합니다."}

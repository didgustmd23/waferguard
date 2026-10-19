"""WM 생성기의 화면용 입력 명세. 실행 검증은 기존 생성기가 최종 담당한다."""

from .wm_run_creation import CREATION_MODULES


def creation_schema(module_id):
    """위젯·import 경로 대신 고정 자료형과 의미를 반환한다. 파일은 읽지 않는다."""
    if module_id not in CREATION_MODULES:
        return {"supported": False, "fields": []}

    def field(name, title, kind, default="", help="", **bounds):
        return {"name": name, "title": title, "kind": kind, "default": default,
                "help": help, **bounds}

    fields = [
        field("pipeline", "모듈 계획 JSON", "json_file", "configs/agents/pipelines/wm_development_example.json",
              "처리 순서와 선행 의존성을 정의한 기존 계획입니다. 파일은 변경하지 않습니다."),
        field("bindings", "현재 작업 배정 JSON", "json_file", help="같은 계획 서명의 기존 배정 파일입니다. 최초 단계는 runs가 빈 배정을 사용합니다."),
        field("step_id", "계획의 단계 ID", "text", {
            "wm_canonical_eda": "eda", "wm_split_propose": "proposal", "wm_split_commit": "split",
            "wm_transform_check": "transform", "wm_cache_prepare": "cache", "wm_dataset_check": "dataset",
        }[module_id], "계획 안의 첫 미배정 단계만 생성할 수 있습니다."),
        field("name", "새 작업 이름", "text", help="영문·숫자 등 기존 작업 이름 규칙을 따르며 기존 기록을 덮어쓰지 않습니다."),
    ]
    _, _, default_output, maximum = CREATION_MODULES[module_id]
    fields.append(field("max_output_bytes", "출력 크기 상한 (byte)", "integer", default_output,
                        "디스크 출력 예산입니다. RAM 제한과 다르며 초과 시 일부만 저장하지 않습니다.", minimum=1, maximum=maximum))

    config_fields = {
        "wm_split_propose": ("split_config", "분할 설정 JSON", "configs/wm811k/split.json"),
        "wm_transform_check": ("transform_config", "변환 설정 JSON", "configs/wm811k/transform.json"),
        "wm_cache_prepare": ("cache_config", "캐시 설정 JSON", "configs/wm811k/cache.json"),
    }
    if module_id in config_fields:
        name, title, default = config_fields[module_id]
        fields.append(field(name, title, "json_file", default, "기존 검증기로 설정을 검사하고 생성 시 확인한 복사본을 사용합니다."))

    if module_id == "wm_canonical_eda":
        fields.extend([
            field("canonical_dir", "canonical 폴더", "directory", help="검증 대상 canonical 묶음입니다. pickle 파일을 직접 지정하는 칸이 아닙니다."),
            field("max_input_bytes", "입력 복사 상한 (byte)", "integer", 4 * 1024 ** 3,
                  "원본 바이트의 격리 복사 예산입니다. 초과 입력을 자르지 않습니다.", minimum=1, maximum=32 * 1024 ** 3),
            field("include_examples", "예시 맵 그림 생성", "boolean", False,
                  "로컬 결과에 실제 맵 예시가 포함될 수 있습니다. LLM 전송은 하지 않습니다."),
            field("max_group_ratio", "최대 연결 그룹 검토 상한", "number", 0.2,
                  "큰 연결 그룹에 대한 검토 기준이며 자동 제외 비율이 아닙니다.", minimum=0.000001, maximum=1.0),
        ])

    if module_id == "wm_dataset_check":
        fields.append(field("batch_size", "검사 배치 크기", "integer", 32,
                            "Train·Validation의 입력 배치를 검사합니다. 학습은 하지 않습니다.", minimum=1, maximum=256))
        for name, title in (("max_input_bytes", "캐시 입력 복사 상한 (byte)"),
                            ("max_runtime_bytes", "런타임 복사 상한 (byte)")):
            fields.append(field(name, title, "integer", 8 * 1024 ** 3,
                                "디스크 복사 예산이며 worker RAM 상한과 다릅니다.", minimum=1, maximum=32 * 1024 ** 3))

    return {"schema_version": "workflow_creation_form_v1", "supported": True, "fields": fields,
            "notice": "명세는 입력 안내입니다. 실행 권한이 아니며 기존 생성기가 모든 조건을 재검사합니다."}


def creation_arguments(module_id, values):
    """폼 값을 기존 생성기의 인자로 제한적으로 변환한다. 추가 옵션은 거부한다."""
    schema = creation_schema(module_id)
    if not schema["supported"] or type(values) is not dict or set(values) != {f["name"] for f in schema["fields"]}:
        raise ValueError("현재 생성 명세의 정확한 입력 항목이 필요합니다.")

    for field in schema["fields"]:
        value = values[field["name"]]
        kind = field["kind"]
        expected = {"integer": int, "number": float, "boolean": bool}.get(kind, str)
        if type(value) is not expected:
            raise ValueError("생성 설정의 자료형이 올바르지 않습니다.")

        if expected is str:
            if not value.strip() or len(value) > 4096 or any(ord(c) < 32 for c in value):
                raise ValueError("생성 설정 경로·이름에 빈 값 또는 제어 문자를 사용할 수 없습니다.")
        elif expected in (int, float) and not field["minimum"] <= value <= field["maximum"]:
            raise ValueError("생성 설정이 허용 범위를 벗어났습니다.")

    options = {key: values[key] for key in ("max_output_bytes", "split_config", "transform_config", "cache_config", "canonical_dir") if key in values}
    if module_id == "wm_canonical_eda":
        options["canonical_options"] = {key: values[key] for key in ("max_input_bytes", "include_examples", "max_group_ratio")}
    elif module_id == "wm_dataset_check":
        options["dataset_options"] = {key: values[key] for key in ("batch_size", "max_input_bytes", "max_runtime_bytes")}

    return {key: values[key] for key in ("pipeline", "bindings", "step_id", "name")} | options

"""검사를 통과한 전처리 요약을 LLM 없이 한국어 로컬 보고서로 구성한다."""

import html


def _cell(value):
    """입력 컬럼명이 Markdown 표나 HTML을 삽입하지 않도록 이스케이프한다."""
    return html.escape(str(value)).replace("|", "&#124;").replace("`", "&#96;").replace("\r", " ").replace("\n", " ")


def render_preparation_report(contract):
    """통과한 구조화 결과만 렌더링한다. 모델 판단·자연어 추측은 추가하지 않는다."""
    if not isinstance(contract, dict) or set(contract) != {"schema_version", "status", "checks", "summary"}:
        raise ValueError("검증 완료 계약 결과가 필요합니다.")

    if contract["schema_version"] != "prepared_check_v1" or contract["status"] != "passed":
        raise ValueError("통과하지 않은 계약으로 완료 보고서를 생성할 수 없습니다.")

    if not isinstance(contract["checks"], list) or not contract["checks"] or any(
        not isinstance(check, dict) or set(check) != {"code", "passed"} or check["passed"] is not True
        for check in contract["checks"]
    ):
        raise ValueError("완료 보고서에 실패한 검사가 포함되어 있습니다.")

    summary = contract["summary"]
    rules = summary["rules"]
    lines = [
        "# 전처리 최종 입력 계약 보고서", "", "## 1. 처리 요약", "",
        "Train에서만 처리 기준을 학습하고, 저장된 변환기로 Train·Validation 출력을 대조했습니다.", "",
        f"입력 특징 **{summary['input_features']}개** → 유지한 입력 특징 **{summary['retained_input_features']}개** → 변환 출력 **{summary['output_features']}개**.", "",
        "원핫 인코딩 때문에 출력 열 수가 유지한 원본 특징 수보다 많을 수 있습니다.", "",
        "## 2. 적용 설정", "", "| 항목 | 설정 |", "| --- | --- |",
    ]
    labels = {
        "missing_ratio_threshold": "Train 결측률 제거 기준(초과)", "drop_constant": "상수 특징 제거",
        "numeric_imputation": "수치 결측 처리", "scaling": "스케일링", "outlier": "이상치 처리",
        "iqr_multiplier": "IQR 배수", "binning": "수치 범주화", "n_bins": "목표 구간 수",
        "bin_encoding": "구간 인코딩", "categorical_imputation": "범주 결측 처리",
        "unknown_category": "새 범주 정책", "max_categories": "특징별 범주 상한",
        "max_output_features": "출력 열 상한", "max_output_cells": "배치 출력 셀 상한",
        "algorithm_version": "알고리즘 버전",
    }
    lines.extend(f"| {label} | {_cell(rules[key])} |" for key, label in labels.items())
    lines.extend([
        "", "## 3. 분할별 처리 결과", "",
        "| 구간 | 행 수 | 출력 열 | 원본 특징 결측 | 유지 특징 결측 | 출력 결측 | 새 범주 발생 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for row in summary["splits"]:
        lines.append(
            f"| {_cell(row['split'])} | {row['rows']} | {row['output_features']} | "
            f"{row['input_missing_count']} | {row['retained_input_missing_count']} | "
            f"{row['output_missing_count']} | {row['unknown_category_count']} |"
        )

    lines.extend([
        "", "결측 수는 해당 특징 셀의 개수입니다. 열 제거·인코딩으로 비교 대상 열 수가 바뀌므로 결측 수만으로 품질 개선을 판단하지 않습니다.",
        "수치 결측 처리 none에서는 허용된 수치 출력에 NaN이 남을 수 있습니다. 후속 모델의 결측 지원을 확인하세요.",
        "새 범주 ignore에서는 해당 특징의 원핫 출력이 모두 0이며 발생 건수를 위에 기록했습니다.",
        "", "## 4. Train 기준 제거 특징", "", "| 특징 | 제거 이유 |", "| --- | --- |",
    ])
    reasons = {"HIGH_MISSING_RATIO": "결측률 기준 초과", "CONSTANT_OR_ALL_MISSING": "상수 또는 전체 결측"}

    if summary["removed"]:
        lines.extend(f"| {_cell(row['column'])} | {reasons[row['reason']]} |" for row in summary["removed"])
    else:
        lines.append("| 없음 | 제거 설정 미적용 또는 해당 특징 없음 |")

    lines.extend(["", "## 5. 최종 검사", "", "| 검사 코드 | 결과 |", "| --- | --- |"])
    lines.extend(f"| {_cell(check['code'])} | 통과 |" for check in contract["checks"])
    lines.extend([
        "", "## 6. 산출물과 처리 범위", "",
        "실행기가 생성한 report_context.json의 prepared_artifact에서 준비 산출물 위치를 확인하세요.",
        "preprocessor.json은 저장 변환기, split_contract.json은 분할 계약, row_contract.json은 특징 행과 원본 위치·정답의 연결입니다.",
        "train_features.csv·validation_features.csv와 함께 같은 원본·설정 버전으로 사용하세요.", "",
        "Test는 변환하지 않았습니다. 모델 학습·성능 평가·전체 실데이터 실행 환경 검증 결과가 아닙니다.",
        "최종 검사는 저장 변환기의 재학습 없이 수행했습니다. 데이터셋·환경 전반의 무누수나 보안 인증을 뜻하지 않습니다.",
        "이 보고서·상세 통계·컬럼·식별값·경로는 로컬 전용이며 LLM·외부 추적으로 전달하지 마세요.", "",
    ])

    return "\n".join(lines)

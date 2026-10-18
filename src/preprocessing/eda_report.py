"""EDA 계산과 분리한 그림·파일·로컬 보고서 생성. 현재 CLI는 전체 보고서를 사용한다."""

import json
import math
from pathlib import Path

import pandas as pd

from .eda import collect_eda


def plot_missing(tables, limit):
    """결측률 상위 특징만 표시한다. 전체 집계는 계산 결과에 남는다."""
    import matplotlib.pyplot as plt

    selected = tables["missing"].sort_values("missing_ratio", ascending=False).head(limit)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(selected.index, selected["missing_ratio"])
    axis.set(ylabel="Missing ratio", ylim=(0, 1), title="Missingness")
    axis.tick_params(axis="x", rotation=60)
    figure.tight_layout()

    return figure


def plot_numeric(data, limit):
    """대치·스케일링 없이 수치 특징의 분포를 표시한다."""
    import matplotlib.pyplot as plt

    names = data.roles.numeric[:limit]
    count = max(1, len(names))
    figure, axes = plt.subplots(math.ceil(count / 3), 3, figsize=(12, 3 * math.ceil(count / 3)), squeeze=False)

    for axis in axes.flat:
        axis.set_visible(False)

    for axis, name in zip(axes.flat, names):
        axis.set_visible(True)
        axis.hist(data.frame[name].dropna(), bins=30)
        axis.set_title(name)

    if not names:
        axes.flat[0].set_visible(True)
        axes.flat[0].text(0.5, 0.5, "No numeric features", ha="center")

    figure.tight_layout()

    return figure


def plot_correlations(tables, limit):
    """계산된 상관행렬을 표시한다. 통계를 다시 계산하지 않는다."""
    import matplotlib.pyplot as plt

    matrix = tables["pearson"].iloc[:limit, :limit]
    figure, axis = plt.subplots(figsize=(8, 7))

    if matrix.empty:
        axis.text(0.5, 0.5, "No numeric features", ha="center")
    else:
        artist = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(matrix)), matrix.columns, rotation=90)
        axis.set_yticks(range(len(matrix)), matrix.index)
        figure.colorbar(artist, ax=axis)

    axis.set_title("Pearson correlation")
    figure.tight_layout()

    return figure


def write_eda_report(data, tables, output_dir, *, max_plot_features=12):
    """전체 분석 결과를 저장한다. 부분 분석 출력 계약은 후속 등록부 단계에서 정의한다."""
    if type(max_plot_features) is not int or not 1 <= max_plot_features <= 30:
        raise ValueError("그래프 특징 상한은 1~30 사이 정수여야 합니다.")

    required = {
        "summary", "missing", "numeric_statistics", "categorical_distribution",
        "target_distribution", "pearson", "spearman", "correlation_support",
    }

    # 기존 부모 검증기가 기대하는 전체 보고서에 부분 결과를 섞지 않는다.
    if set(tables) != required:
        raise ValueError("전체 EDA 보고서에 필요한 분석 결과가 모두 있어야 합니다.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(output / f"{name}.csv", encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    plotters = (
        ("missing", lambda: plot_missing(tables, max_plot_features)),
        ("distributions", lambda: plot_numeric(data, max_plot_features)),
        ("correlations", lambda: plot_correlations(tables, max_plot_features)),
    )

    for name, plotter in plotters:
        figure = plotter()

        try:
            figure.savefig(output / f"{name}.png", dpi=120)
        finally:
            plt.close(figure)

    (output / "summary.json").write_text(
        json.dumps(tables["summary"], ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    (output / "eda.md").write_text(report_markdown(), encoding="utf-8")
    # worker의 생성 완료 표시이며 부모의 검증·커밋 완료와는 구분한다.
    (output / "SUCCESS.json").write_text(json.dumps({"status": "eda_complete"}), encoding="utf-8")

    return {"status": "eda_complete"}


def generate_eda_report(data, output_dir, *, max_plot_features=12):
    """현재 전체 EDA 실행 경로. 계산 결과를 명시적으로 보고서 계층에 전달한다."""
    if type(max_plot_features) is not int or not 1 <= max_plot_features <= 30:
        raise ValueError("그래프 특징 상한은 1~30 사이 정수여야 합니다.")

    tables = collect_eda(data)

    return write_eda_report(data, tables, output_dir, max_plot_features=max_plot_features)


def report_markdown():
    """생성기와 부모 검증기가 공유하는 고정 안내. 데이터 해석은 포함하지 않는다."""
    return (
        "# 공통 EDA 보고서\n\n"
        "![결측](missing.png)\n\n![수치 분포](distributions.png)\n\n"
        "![상관](correlations.png)\n\n"
        "표에는 지정된 전체 특징을 기록하며 그래프의 특징 수만 제한합니다.\n\n"
        "상관분석은 지정한 수치 특징만 사용합니다. target·group·ID는 제외합니다. "
        "상수 열과 표본 부족의 NaN은 상관이 없다는 뜻의 0이 아닙니다. "
        "변수 쌍별 관측 수는 correlation_support.csv를 확인하세요.\n\n"
        "동일 특징 행 수는 동일 실물 중복의 확정 결과가 아닙니다. "
        "분할·특징 선택·대치·스케일링·라벨 변경을 수행하지 않았습니다.\n\n"
        "구조·품질 EDA와 모델 선택을 구분하고 학습용 변환은 Train에서만 fit하세요. "
        "보고서·수치·그림·컬럼명은 로컬 전용이며 LLM에 전달하지 않습니다.\n"
    )

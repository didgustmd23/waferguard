"""전처리 후보 보고서의 고정 안내. 실제 수치·설정은 로컬 JSON으로 확인한다."""


def candidate_markdown(report):
    """후보의 변경 범위를 사람이 읽을 표로 작성한다. 원문 컬럼은 Markdown에서 이스케이프한다."""
    import html

    def cell(value):
        text = html.escape(str(value)).replace("\\", "\\\\").replace("\n", " ").replace("\r", " ")

        for character in "|`*_[]()!#":
            text = text.replace(character, "\\" + character)

        return text

    introduction = ("# 전처리 기준 후보\n\n"
            "Train에서만 산출한 후보입니다. 아직 Train·Validation에 변환을 적용하지 않았습니다.\n\n"
            "candidate.json에서 설정·제거 목록·사유·유지 특징과 출력 폭을 확인하세요. "
            "preprocessor.json에는 Train 대치값·변환 기준이 들어 있습니다. 로컬에서만 확인하세요.\n\n"
            "후보 산출 완료는 적용 승인·전처리 완료·모델 학습 완료를 뜻하지 않습니다.\n")
    lines = [introduction, "## 변경 예정 범위", "",
             f"- Train 행 수: {report['binding']['train_rows']}",
             f"- 제거 후보: {len(report['removed'])}개", f"- 유지 입력 특징: {len(report['retained'])}개",
             f"- 적용 후 예상 출력 특징: {report['output_feature_count']}개", "",
             "## 선택한 규칙", "", "| 항목 | 설정 |", "| --- | --- |"]
    lines.extend(f"| {cell(name)} | {cell(value)} |" for name, value in report["binding"]["rules"].items())
    lines.extend(["", "## 제거 후보", "", "| 특징 | 사유 |", "| --- | --- |"])
    lines.extend(f"| {cell(item['column'])} | {cell(item['reason'])} |" for item in report["removed"])

    if not report["removed"]:
        lines.append("| 없음 | 제거 후보가 없습니다. |")

    impact = report.get("impact")

    if impact is not None:
        lost = [item for item in impact if item["constant_after"] and item["distinct_before"] > 1]
        lines.extend(["", "## 예상 처리 영향 (Train 후보 계산)", "",
                      f"- clipping으로 바뀌는 값: {sum(item['clipped'] or 0 for item in impact)}개 "
                      f"(특징 {sum(1 for item in impact if item['clipped'])}개)",
                      f"- 변환 후 값이 하나뿐인 특징: {sum(1 for item in impact if item['constant_after'])}개 "
                      f"(그중 변환 전에는 값이 여러 개였던 특징 {len(lost)}개)", "",
                      "| 정보 손실 특징 | 변환 전 고유값 | clipping 변경 값 |", "| --- | --- | --- |"])
        lines.extend(f"| {cell(item['column'])} | {item['distinct_before']} | {cell(item['clipped'])} |" for item in lost)

        if not lost:
            lines.append("| 없음 | - | - |")

    lines.extend(["", "유지 특징의 정확한 순서와 처리 기준은 로컬 JSON에서 확인하세요.",
                  "변경 예정 목록일 뿐입니다. 실제 적용은 별도 승인이 필요합니다."])
    return "\n".join(lines) + "\n"

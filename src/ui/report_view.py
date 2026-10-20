"""검증된 결과 바이트만으로 보고서의 상대 PNG 참조를 표시한다."""

from itertools import islice
from pathlib import PurePosixPath
import re

from .image_validation import valid_report_png


IMAGE_REFERENCE = re.compile(r"^[ \t]*!\[([^\]\n]{0,500})\]\(([^\n)]{1,2048})\)[ \t\r]*$", re.MULTILINE)
MAX_REPORT_CHARACTERS = 100000
MAX_INLINE_IMAGES = 32
MAX_INLINE_TABLES = 16
MAX_TABLE_ROWS = 200
MAX_TABLE_COLUMNS = 100


def _table_cells(line):
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return None

    # 이스케이프한 셀 안의 세로줄은 열 구분자로 취급하지 않는다.
    return [cell.strip().replace(r"\|", "|").replace(r"\_", "_")
            for cell in re.split(r"(?<!\\)\|", line[1:-1])]


def render_text_with_tables(st, text, table_budget=MAX_INLINE_TABLES):
    """머리글·구분선이 있는 표만 해석한다. 셀의 링크·HTML은 실행하지 않는다."""
    lines, pending = text.splitlines(), []
    index, rendered = 0, 0
    fence = None
    while index < len(lines):
        line = lines[index]
        marker = line.strip()[:3]
        if marker in {"```", "~~~"}:
            fence = None if fence == marker else marker if fence is None else fence
            pending.append(line)
            index += 1
            continue

        header = _table_cells(line)
        divider = _table_cells(lines[index + 1]) if index + 1 < len(lines) else None
        is_table = (fence is None and rendered < table_budget and header and divider and
                    2 <= len(header) <= MAX_TABLE_COLUMNS and len(header) == len(divider) and
                    all(header) and len(set(header)) == len(header) and
                    all(re.fullmatch(r":?-{3,}:?", cell) for cell in divider))
        if not is_table:
            pending.append(line)
            index += 1
            continue

        if pending:
            st.text("\n".join(pending).strip())
            pending = []

        rows, total_rows = [], 0
        index += 2
        while index < len(lines):
            cells = _table_cells(lines[index])
            if cells is None or len(cells) != len(header):
                break

            if total_rows < MAX_TABLE_ROWS:
                rows.append(cells)
            total_rows += 1
            index += 1

        # Markdown 셀 해석 없이 일반 문자열 셀로 표시해 외부 이미지 로드를 막는다.
        import pandas as pd

        st.dataframe(pd.DataFrame(rows, columns=header), hide_index=True)
        if total_rows > MAX_TABLE_ROWS:
            st.caption("표는 처음 200행까지만 표시합니다. 전체 행은 원래 로컬 결과에서 확인하세요.")
        rendered += 1

    if pending and "\n".join(pending).strip():
        st.text("\n".join(pending).strip())

    return rendered


def referenced_png(report_name, reference, files):
    """파일을 열지 않고 이미 검증한 같은 결과 묶음에서만 정확히 하나를 찾는다."""
    if any(character in reference for character in (":", "\\", "?", "#")):
        return None

    relative = PurePosixPath(reference)
    report = PurePosixPath(report_name)
    if (relative.is_absolute() or report.is_absolute() or
            ".." in relative.parts or ".." in report.parts or relative.suffix.lower() != ".png"):
        return None

    name = (report.parent / relative).as_posix()
    matches = [item for item in files if item["name"] == name and item["suffix"] == ".png"]
    if len(matches) != 1:
        return None

    data = matches[0]["data"]
    return data if len(data) <= 8 * 1024 ** 2 and valid_report_png(data) else None


def render_report(st, report, files):
    """표와 검증된 이미지만 구조화하고 나머지 링크·HTML은 해석하지 않는다."""
    text = report["data"].decode("utf-8")[:MAX_REPORT_CHARACTERS]
    offset = 0
    tables_left = MAX_INLINE_TABLES
    # 악성 보고서의 반복 이미지 참조가 무제한 UI 요소를 만들지 않도록 제한한다.
    for match in islice(IMAGE_REFERENCE.finditer(text), MAX_INLINE_IMAGES):
        preceding = text[offset:match.start()].strip()
        if preceding:
            tables_left -= render_text_with_tables(st, preceding, tables_left)

        image = referenced_png(report["name"], match.group(2), files)
        if image is None:
            st.warning("이 이미지 참조는 검증된 결과 목록과 연결되지 않아 표시하지 않았습니다.")
        else:
            if match.group(1):
                st.text(match.group(1))
            st.image(image)

        offset = match.end()

    remaining = text[offset:].strip()
    if remaining:
        render_text_with_tables(st, remaining, tables_left)
        if IMAGE_REFERENCE.search(remaining):
            st.caption("보고서 본문 이미지는 최대 32개까지 표시합니다. 나머지는 개별 결과에서 확인하세요.")

"""구간 CSV의 파일 전체가 아니라 레코드·필드 수를 제한하는 스트리밍 파서."""

import csv


MAX_RECORD_BYTES = 1024 * 1024
MAX_COLUMNS = 1024


class BoundedCsvLines:
    """인용된 줄바꿈도 한 레코드의 읽기 예산에 포함한다."""

    def __init__(self, stream):
        self.stream = stream
        self.used = 0

    def __iter__(self):
        return self

    def __next__(self):
        # 한 줄을 먼저 무제한으로 읽지 않는다. UTF-8 바이트 길이는 읽은 직후 확인한다.
        line = self.stream.readline(MAX_RECORD_BYTES - self.used + 1)

        if not line:
            raise StopIteration

        self.used += len(line.encode("utf-8"))

        if self.used > MAX_RECORD_BYTES:
            raise ValueError("CSV 레코드 크기 상한을 초과했습니다.")

        return line


def iter_csv_rows(stream, *, max_columns=MAX_COLUMNS):
    """부모·worker가 동일한 레코드 크기·컬럼 수 제한을 사용한다."""
    if type(max_columns) is not int or not 1 <= max_columns <= 100_000:
        raise ValueError("CSV 컬럼 수 제한은 1~100000 범위의 정수여야 합니다.")

    lines = BoundedCsvLines(stream)

    for row in csv.reader(lines, strict=True):
        if len(row) > max_columns:
            raise ValueError("CSV 컬럼 수 상한을 초과했습니다.")

        lines.used = 0

        yield row

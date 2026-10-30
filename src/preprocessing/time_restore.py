"""시간 분할 준비의 결측 시각 복원 규칙. 사용자가 정한 형식·시간대·주기·행 순서로 하나로 정해지는 내부 결측만 복원 대상으로 판정한다.

파일을 바꾸지 않는다. 판정(find_restorations)은 복원 대상 행 번호와 시각만 돌려주고, 적용(apply_restorations)은 입력을 바꾸지 않은 새 값 목록을 만든다.
미리보기·적용 실행과 파일 생성은 격리 worker(tabular_time_worker)가 한다.
앞뒤 값 채우기·평균/선형 보간·모델 추측·앞뒤 끝 외삽·결측 행 자동 삭제·재표본화·구간 간격(gap)은 하지 않는다.
오류 메시지에는 원본 시간값·컬럼 값·경로를 넣지 않는다.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones


# ==========================================
# 복원 규칙 계약
# - 주기: daily(달력상 매일, 같은 현지 시각), weekdays(월~금, 같은 현지 시각), fixed(고정 분 간격, 절대 시간)
# - 형식: strptime 지시자 중 허용 목록만, 날짜(네 자리 연 %Y·월·일)를 반드시 포함, 왕복 변환이 같은 값이어야 함
#   두 자리 연도(%y)는 세기 기준이 정해지지 않아 받지 않는다
# - 시간대: 형식에 %z가 없으면(현지 시각) IANA 시간대를 지정해야 하고, %z가 있으면 지정하지 않는다
# - 행 순서 확인: 원본 행 순서가 시간 순서이고 각 행이 주기상 한 시점이라는 사용자 확인이 있어야 한다
# ==========================================
FREQUENCIES = ("daily", "weekdays", "fixed")
MAX_INTERVAL_MINUTES = 7 * 24 * 60
MAX_FORMAT_LENGTH = 64
MAX_NAME_LENGTH = 256
FORMAT_DIRECTIVES = {"Y", "m", "d", "H", "M", "S", "f", "z"}
FORMAT_LITERALS = set("-/:. T")
_TOKEN = re.compile(r"%(.)|(.)", re.S)

CODES = {
    "TIME_RULES_INVALID": "시간 복원 설정의 형식이 계약과 다릅니다",
    "TIME_FREQUENCY_INVALID": "복원 주기는 매일(daily)·평일(weekdays)·고정 간격(fixed) 중 하나여야 하고, 고정 간격일 때만 1분~7일 사이의 분 단위 간격을 지정합니다",
    "TIME_ROW_ORDER_UNCONFIRMED": "원본 행 순서가 시간 순서이고 각 행이 주기상 한 시점인지 확인되지 않아 복원하지 않습니다",
    "TIME_FORMAT_INVALID": "시간 형식이 허용 지시자(%Y %m %d %H %M %S %f %z, 두 자리 연도 %y 제외)와 구분 문자(- / : . T 공백)로 연(%Y)·월·일을 포함해 정확히 되읽히지 않습니다",
    "TIME_TIMEZONE_INVALID": "시간대 이름이 IANA 시간대 목록에 없습니다",
    "TIME_TIMEZONE_REQUIRED": "시간대 정보가 없는 현지 시각이므로 시간대를 지정해야 합니다",
    "TIME_TIMEZONE_CONFLICT": "시간값에 이미 시간대가 있어(형식의 %z 또는 시간대가 있는 시각 자료형) 별도 시간대를 함께 지정할 수 없습니다",
    "TIME_PARSE_FAILED": "결측이 아닌 시간값 중 지정한 형식으로 읽히지 않는 값이 있습니다",
    "TIME_EDGE_MISSING": "첫 행·마지막 행처럼 앞뒤 기준 시각이 없는 결측은 복원하지 않습니다",
    "TIME_DUPLICATE": "같은 시각의 행이 둘 이상 있어 행마다 주기상 한 시점이라는 규칙과 맞지 않습니다",
    "TIME_FREQUENCY_MISMATCH": "기존 시간값의 간격·순서가 지정한 주기와 맞지 않습니다",
    "TIME_AMBIGUOUS": "시간대·서머타임 처리로 시각이 하나로 정해지지 않습니다",
    "TIME_RESTORATION_MISMATCH": "복원 대상이 현재 시간값의 결측 위치와 맞지 않습니다",
}


class TimeRestoreError(ValueError):
    """복원하지 않고 멈춘 확인된 원인. 메시지는 고정 문구이며 원본 값을 담지 않는다."""

    def __init__(self, code):
        super().__init__(CODES[code])
        self.code, self.message = code, CODES[code]


def check_format(text):
    """허용 지시자·구분 문자만 쓰고 연·월·일을 포함하며 왕복 변환이 같은 시각이 되는 형식인지. 시간대 지시자 포함 여부를 돌려준다."""
    if type(text) is not str or not 0 < len(text) <= MAX_FORMAT_LENGTH:
        raise TimeRestoreError("TIME_FORMAT_INVALID")

    directives = []

    for match in _TOKEN.finditer(text):
        directive, literal = match.groups()

        if directive is not None:
            if directive not in FORMAT_DIRECTIVES:
                raise TimeRestoreError("TIME_FORMAT_INVALID")

            directives.append(directive)
        elif literal not in FORMAT_LITERALS:
            raise TimeRestoreError("TIME_FORMAT_INVALID")

    if len(set(directives)) != len(directives) or not {"Y", "m", "d"} <= set(directives):
        raise TimeRestoreError("TIME_FORMAT_INVALID")

    aware = "z" in directives
    sample = datetime(2001, 2, 3, 4, 5, 6, 789000, tzinfo=timezone(timedelta(hours=9)) if aware else None)

    try:
        parsed = datetime.strptime(sample.strftime(text), text)
    except ValueError as error:
        raise TimeRestoreError("TIME_FORMAT_INVALID") from error

    # 형식에 없는 성분은 기본값(0)으로 읽히므로 형식에 있는 성분만 같은지 본다.
    parts = {"Y": "year", "m": "month", "d": "day", "H": "hour", "M": "minute", "S": "second", "f": "microsecond"}

    if any(getattr(parsed, parts[item]) != getattr(sample, parts[item]) for item in directives if item in parts):
        raise TimeRestoreError("TIME_FORMAT_INVALID")

    if aware and parsed.utcoffset() != sample.utcoffset():
        raise TimeRestoreError("TIME_FORMAT_INVALID")

    return aware


def check_timezone(name):
    if type(name) is not str or not 0 < len(name) <= MAX_FORMAT_LENGTH or name not in available_timezones():
        raise TimeRestoreError("TIME_TIMEZONE_INVALID")

    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise TimeRestoreError("TIME_TIMEZONE_INVALID") from error


@dataclass(frozen=True)
class TimeRestoreRules:
    """사용자가 지정·승인한 시간 복원 규칙. 모델이 만들거나 추측하지 않는다."""

    column: str
    timestamp_format: str | None
    timezone: str | None
    frequency: str
    interval_minutes: int | None
    row_order_confirmed: bool

    def __post_init__(self):
        if type(self.column) is not str or not 0 < len(self.column) <= MAX_NAME_LENGTH:
            raise TimeRestoreError("TIME_RULES_INVALID")

        if self.frequency not in FREQUENCIES or (self.frequency == "fixed") != (self.interval_minutes is not None):
            raise TimeRestoreError("TIME_FREQUENCY_INVALID")

        if self.interval_minutes is not None and (type(self.interval_minutes) is not int
                                                  or not 1 <= self.interval_minutes <= MAX_INTERVAL_MINUTES):
            raise TimeRestoreError("TIME_FREQUENCY_INVALID")

        if self.row_order_confirmed is not True:
            raise TimeRestoreError("TIME_ROW_ORDER_UNCONFIRMED")

        aware = self.timestamp_format is not None and check_format(self.timestamp_format)

        if self.timezone is not None:
            if aware:
                raise TimeRestoreError("TIME_TIMEZONE_CONFLICT")

            check_timezone(self.timezone)
        elif self.timestamp_format is not None and not aware:
            raise TimeRestoreError("TIME_TIMEZONE_REQUIRED")


@dataclass(frozen=True)
class Restoration:
    """복원 대상 행 번호(0부터)와 그 시각(시간대 포함). 개수는 len(rows)."""

    rows: tuple
    times: tuple


# ==========================================
# 판정
# ==========================================

def _missing(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return True

    # pandas NaN·NaT·NA도 결측으로 본다. 다른 자료형은 파싱 단계가 판정한다.
    return (isinstance(value, float) and math.isnan(value)) or type(value).__name__ in ("NaTType", "NAType")


def _local(wall, zone):
    """현지 시각을 시간대에 붙인다. 존재하지 않는 시각(서머타임 시작)과 두 번 있는 시각(서머타임 끝)은 모호하다."""
    first, second = wall.replace(tzinfo=zone, fold=0), wall.replace(tzinfo=zone, fold=1)

    if first.utcoffset() != second.utcoffset():
        raise TimeRestoreError("TIME_AMBIGUOUS")

    if first.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != wall:
        raise TimeRestoreError("TIME_AMBIGUOUS")

    return first


def _parse(value, rules, zone):
    if rules.timestamp_format is not None:
        if type(value) is not str:
            raise TimeRestoreError("TIME_PARSE_FAILED")

        try:
            parsed = datetime.strptime(value.strip(), rules.timestamp_format)
        except ValueError as error:
            raise TimeRestoreError("TIME_PARSE_FAILED") from error
    elif isinstance(value, datetime):
        parsed = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    else:
        raise TimeRestoreError("TIME_PARSE_FAILED")

    if parsed.tzinfo is not None:
        if zone is not None:
            raise TimeRestoreError("TIME_TIMEZONE_CONFLICT")

        return parsed

    if zone is None:
        raise TimeRestoreError("TIME_TIMEZONE_REQUIRED")

    return _local(parsed, zone)


def _wall(value):
    return value.replace(tzinfo=None)


def _weekdays_after(day, count):
    while count:
        day += timedelta(days=1)
        count -= day.weekday() < 5

    return day


def _shifted_wall(start, count, rules):
    """현지 시각 주기(daily·weekdays)에서 기준 시각의 날짜만 count번 옮긴 현지 시각. 시각(시·분·초)은 그대로 둔다."""
    if rules.frequency == "weekdays":
        if start.weekday() >= 5:
            raise TimeRestoreError("TIME_FREQUENCY_MISMATCH")

        day = _weekdays_after(start.date(), count)
    else:
        day = start.date() + timedelta(days=count)

    return datetime.combine(day, _wall(start).time())


def _step(start, count, rules, zone, offsets):
    """기준 시각에서 주기 count번 뒤의 시각(시간대 포함). 고정 간격은 절대 시간으로, 현지 시각 주기는 날짜만 옮겨 계산한다."""
    if rules.frequency == "fixed":
        moved = start.astimezone(timezone.utc) + timedelta(minutes=rules.interval_minutes * count)
        return moved.astimezone(zone if zone is not None else start.tzinfo)

    wall = _shifted_wall(start, count, rules)

    if zone is not None:
        return _local(wall, zone)

    # 시간대 표기(%z)가 있는 값은 앞뒤 기준의 시간대가 같을 때만 복원 시각의 시간대가 정해진다.
    if len(offsets) != 1:
        raise TimeRestoreError("TIME_AMBIGUOUS")

    return wall.replace(tzinfo=start.tzinfo)


def _unambiguous_wall(value, zone):
    """복원 시각을 현지 시각으로 적을 때 같은 표기가 두 시각을 가리키지 않는지(서머타임 끝)."""
    if zone is not None:
        _local(_wall(value.astimezone(zone)), zone)


def find_restorations(values, rules):
    """행 순서의 시간값에서 복원할 내부 결측 행과 시각을 판정한다. 하나라도 규칙에 맞지 않으면 TimeRestoreError로 멈춘다.

    values는 원본 행 순서의 값이다(결측: None·빈 문자열·NaN·NaT·pandas.NA). 형식이 있으면 문자열만 받는다.
    결측은 앞뒤 유효 시각 사이에서 주기 칸 수가 정확히 맞을 때만 복원 대상이 되고, 앞뒤 끝의 결측은 복원하지 않는다.
    """
    if not isinstance(rules, TimeRestoreRules):
        raise TimeRestoreError("TIME_RULES_INVALID")

    values = list(values)
    zone = None if rules.timezone is None else check_timezone(rules.timezone)
    parsed = {}

    for index, value in enumerate(values):
        if not _missing(value):
            parsed[index] = _parse(value, rules, zone)

    if not parsed or 0 not in parsed or len(values) - 1 not in parsed:
        raise TimeRestoreError("TIME_EDGE_MISSING")

    instants = [item.astimezone(timezone.utc) for item in parsed.values()]

    if len(set(instants)) != len(instants):
        raise TimeRestoreError("TIME_DUPLICATE")

    rows, times = [], []
    known = sorted(parsed)

    for before, after in zip(known, known[1:]):
        start, end, gap = parsed[before], parsed[after], after - before

        # 다음 유효 값은 기준 시각에서 행 칸 수만큼 주기를 옮긴 시각과 같아야 한다(고정 간격은 절대 시각, 그 외는 현지 시각).
        if rules.frequency == "fixed":
            matches = _step(start, gap, rules, zone, {start.utcoffset()}) == end
        else:
            matches = _shifted_wall(start, gap, rules) == _wall(end)

        if not matches:
            raise TimeRestoreError("TIME_FREQUENCY_MISMATCH")

        offsets = {start.utcoffset(), end.utcoffset()}

        for count in range(1, gap):
            restored = _step(start, count, rules, zone, offsets)
            _unambiguous_wall(restored, zone)
            rows.append(before + count)
            times.append(restored)

    return Restoration(tuple(rows), tuple(times))


# ==========================================
# 적용 (입력을 바꾸지 않은 새 값 목록)
# - 복원 대상 행은 결측이어야 하고, 그 밖의 값(결측이 아닌 값)은 그대로 둔다
# - 형식이 있으면 복원 시각을 규칙 형식으로 적고, 다시 읽은 시각이 복원 시각과 같아야 한다
# ==========================================

def is_missing(value):
    """복원 판정과 같은 결측 기준(None·빈 문자열·NaN·NaT·pandas.NA)."""
    return _missing(value)


def format_time(value, rules):
    """복원 시각을 규칙대로 적는다. 형식이 있으면 문자열(현지 시각 형식은 시간대의 현지 표기, %z 형식은 기준 시각의 오프셋), 없으면 시각 그대로다."""
    if not isinstance(rules, TimeRestoreRules) or not isinstance(value, datetime) or value.tzinfo is None:
        raise TimeRestoreError("TIME_RULES_INVALID")

    if rules.timestamp_format is None:
        return value

    zone = None if rules.timezone is None else check_timezone(rules.timezone)
    text = (value.astimezone(zone) if zone is not None else value).strftime(rules.timestamp_format)

    # 적은 표기가 같은 시각·오프셋으로 되읽혀야 한다(형식에 없는 성분이 있거나 표기가 두 시각을 가리키면 멈춤).
    try:
        parsed = _parse(text, rules, zone)
    except TimeRestoreError as error:
        raise TimeRestoreError("TIME_AMBIGUOUS") from error

    if parsed != value or parsed.utcoffset() != value.utcoffset():
        raise TimeRestoreError("TIME_AMBIGUOUS")

    return text


def apply_restorations(values, restoration, rules):
    """판정 결과를 적용한 새 값 목록을 돌려준다. 입력 목록은 바꾸지 않으며 복원 대상이 결측이 아니면 멈춘다."""
    if (not isinstance(restoration, Restoration) or type(restoration.rows) is not tuple
            or type(restoration.times) is not tuple or len(restoration.rows) != len(restoration.times)):
        raise TimeRestoreError("TIME_RULES_INVALID")

    result = list(values)
    previous = -1

    for row, time in zip(restoration.rows, restoration.times):
        if type(row) is not int or not previous < row < len(result) or not _missing(result[row]):
            raise TimeRestoreError("TIME_RESTORATION_MISMATCH")

        result[row], previous = format_time(time, rules), row

    return result


def restoration_digest(restoration):
    """복원 대상 행 번호와 ISO 시각의 정규 JSON sha256. 미리보기·적용·검증이 같은 판정인지 대조하는 식별값이다."""
    if not isinstance(restoration, Restoration):
        raise TimeRestoreError("TIME_RULES_INVALID")

    payload = {"rows": list(restoration.rows), "times": [item.isoformat() for item in restoration.times]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = ["CODES", "FREQUENCIES", "MAX_INTERVAL_MINUTES", "Restoration", "TimeRestoreError", "TimeRestoreRules",
           "apply_restorations", "check_format", "check_timezone", "find_restorations", "format_time", "is_missing",
           "restoration_digest"]

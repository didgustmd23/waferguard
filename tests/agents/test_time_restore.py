"""시간 분할 준비의 결측 시각 복원 판정(time_restore). 합성 시간값으로 유일한 내부 결측 복원과 각 중단 조건을 확인한다.

값을 채우거나 파일을 바꾸지 않는 판정 함수다. 날짜는 고정값을 쓰고 현재 시각에 의존하지 않는다.
"""

from datetime import datetime, timedelta, timezone
import unittest

from src.preprocessing.time_restore import (MAX_INTERVAL_MINUTES, TimeRestoreError, TimeRestoreRules, check_format,
                                            find_restorations)


def rules(**changes):
    values = {"column": "t", "timestamp_format": "%Y-%m-%d", "timezone": "Asia/Seoul", "frequency": "daily",
              "interval_minutes": None, "row_order_confirmed": True}
    values.update(changes)
    return TimeRestoreRules(**values)


class RestorationTest(unittest.TestCase):
    def assert_code(self, code, values, rule):
        with self.assertRaises(TimeRestoreError) as caught:
            find_restorations(values, rule)

        self.assertEqual(caught.exception.code, code)
        # 오류 문구에 원본 시간값을 넣지 않는다.
        self.assertFalse([value for value in values if isinstance(value, str) and value and value in caught.exception.message])

    def test_daily_rule_restores_only_the_missing_friday_between_known_days(self):
        # 2026-10-19는 월요일이다. 월·화·수·목·빈 행·토·일 + 매일 한 행 규칙이면 금요일 하나로 정해진다.
        values = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "", "2026-10-24", "2026-10-25"]
        before = list(values)
        result = find_restorations(values, rules())
        self.assertEqual(result.rows, (4,))
        self.assertEqual(result.times[0].replace(tzinfo=None), datetime(2026, 10, 23))
        self.assertEqual(result.times[0].weekday(), 4)
        self.assertEqual(values, before)

    def test_consecutive_internal_gaps_and_missing_markers_are_restored_in_row_order(self):
        values = ["2026-10-19", None, float("nan"), "  ", "2026-10-23"]
        result = find_restorations(values, rules())
        self.assertEqual(result.rows, (1, 2, 3))
        self.assertEqual([item.day for item in result.times], [20, 21, 22])

    def test_weekdays_rule_skips_the_weekend(self):
        # 목(10-22)·빈 행·월(10-26): 평일 규칙이면 빈 행은 금요일 하나다.
        result = find_restorations(["2026-10-22", "", "2026-10-26"], rules(frequency="weekdays"))
        self.assertEqual([item.date().isoformat() for item in result.times], ["2026-10-23"])
        self.assert_code("TIME_FREQUENCY_MISMATCH", ["2026-10-23", "2026-10-24"], rules(frequency="weekdays"))

    def test_fixed_interval_uses_absolute_minutes(self):
        rule = rules(timestamp_format="%Y-%m-%d %H:%M", frequency="fixed", interval_minutes=30)
        result = find_restorations(["2026-10-19 09:00", "", "", "2026-10-19 10:30"], rule)
        self.assertEqual([item.strftime("%H:%M") for item in result.times], ["09:30", "10:00"])

    def test_values_with_offset_restore_only_when_both_sides_share_it(self):
        rule = rules(timestamp_format="%Y-%m-%d %H:%M%z", timezone=None)
        result = find_restorations(["2026-10-19 09:00+0900", "", "2026-10-21 09:00+0900"], rule)
        self.assertEqual(result.times[0].utcoffset(), timedelta(hours=9))
        self.assert_code("TIME_AMBIGUOUS", ["2026-03-07 09:00-0500", "", "2026-03-09 09:00-0400"], rule)

    def test_no_missing_value_restores_nothing(self):
        self.assertEqual(find_restorations(["2026-10-19", "2026-10-20"], rules()).rows, ())

    def test_edge_missing_is_not_extrapolated(self):
        for values in (["", "2026-10-20", "2026-10-21"], ["2026-10-19", "2026-10-20", ""], ["", ""], []):
            with self.subTest(values=values):
                self.assert_code("TIME_EDGE_MISSING", values, rules())

    def test_duplicate_time_conflicts_with_one_row_per_period(self):
        self.assert_code("TIME_DUPLICATE", ["2026-10-19", "2026-10-19", "2026-10-20"], rules())

    def test_values_that_do_not_follow_the_period_or_order_are_rejected(self):
        cases = (["2026-10-19", "2026-10-21"],                  # 하루 빠졌는데 결측 행이 없음
                 ["2026-10-19", "", "2026-10-20"],              # 결측 칸 수와 날짜 간격이 다름
                 ["2026-10-20", "2026-10-19"])                  # 행 순서가 시간 역순
        for values in cases:
            with self.subTest(values=values):
                self.assert_code("TIME_FREQUENCY_MISMATCH", values, rules())

        rule = rules(timestamp_format="%Y-%m-%d %H:%M", frequency="fixed", interval_minutes=60)
        self.assert_code("TIME_FREQUENCY_MISMATCH", ["2026-10-19 09:00", "2026-10-19 09:30"], rule)

    def test_parse_failure_of_a_present_value_stops(self):
        self.assert_code("TIME_PARSE_FAILED", ["2026-10-19", "19/10/2026", "2026-10-21"], rules())
        self.assert_code("TIME_PARSE_FAILED", ["2026-10-19", 20261020, "2026-10-21"], rules())

    def test_daylight_saving_gaps_and_folds_are_ambiguous(self):
        zone = "America/New_York"
        # 2026-03-08 02:30은 존재하지 않는 현지 시각이다(서머타임 시작). 매일 같은 시각 규칙으로 복원할 수 없다.
        daily = rules(timestamp_format="%Y-%m-%d %H:%M", timezone=zone)
        self.assert_code("TIME_AMBIGUOUS", ["2026-03-07 02:30", "", "2026-03-09 02:30"], daily)
        # 2026-11-01 01:30은 두 번 있는 현지 시각이다(서머타임 끝). 값 자체도, 복원 결과의 현지 표기도 하나로 정해지지 않는다.
        self.assert_code("TIME_AMBIGUOUS", ["2026-10-31 01:30", "2026-11-01 01:30", "2026-11-02 01:30"], daily)
        # 00:30(EDT)과 02:30(EST) 사이는 절대 시간 3시간이다. 1시간 간격으로 정해지는 두 시각은 모두 현지 표기 01:30이라
        # 적으면 서로 구분되지 않는다.
        hourly = rules(timestamp_format="%Y-%m-%d %H:%M", timezone=zone, frequency="fixed", interval_minutes=60)
        self.assert_code("TIME_AMBIGUOUS", ["2026-11-01 00:30", "", "", "2026-11-01 02:30"], hourly)
        self.assertEqual(len(find_restorations(["2026-11-01 03:30", "", "2026-11-01 05:30"], hourly).rows), 1)


class RulesContractTest(unittest.TestCase):
    def assert_rule_code(self, code, **changes):
        with self.assertRaises(TimeRestoreError) as caught:
            rules(**changes)

        self.assertEqual(caught.exception.code, code)

    def test_row_order_must_be_confirmed_by_the_user(self):
        for value in (False, None, "yes", 1):
            with self.subTest(value=value):
                self.assert_rule_code("TIME_ROW_ORDER_UNCONFIRMED", row_order_confirmed=value)

    def test_frequency_and_interval_combinations(self):
        for changes in ({"frequency": "hourly"}, {"frequency": None}, {"frequency": "fixed"},
                        {"frequency": "daily", "interval_minutes": 60}, {"frequency": "fixed", "interval_minutes": 0},
                        {"frequency": "fixed", "interval_minutes": MAX_INTERVAL_MINUTES + 1},
                        {"frequency": "fixed", "interval_minutes": True}, {"frequency": "fixed", "interval_minutes": 1.5}):
            with self.subTest(changes=changes):
                self.assert_rule_code("TIME_FREQUENCY_INVALID", **changes)

        self.assertEqual(rules(frequency="fixed", interval_minutes=MAX_INTERVAL_MINUTES).interval_minutes, MAX_INTERVAL_MINUTES)

    def test_format_allows_only_listed_directives_with_a_full_date(self):
        for text in ("%Y-%m-%d", "%y/%m/%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M%z", "%d.%m.%Y"):
            with self.subTest(text=text):
                check_format(text)

        for text in ("", "%A %Y-%m-%d", "%Y-%m", "%Y-%Y-%m-%d", "%Y년%m월%d일", "%Y-%m-%d %p", "%Y%y-%m-%d", "x" * 65):
            with self.subTest(text=text):
                self.assert_rule_code("TIME_FORMAT_INVALID", timestamp_format=text)

        with self.assertRaises(TimeRestoreError):
            check_format(None)

    def test_timezone_rules(self):
        self.assert_rule_code("TIME_TIMEZONE_INVALID", timezone="Mars/Base")
        self.assert_rule_code("TIME_TIMEZONE_INVALID", timezone="../etc/passwd")
        self.assert_rule_code("TIME_TIMEZONE_REQUIRED", timezone=None)
        self.assert_rule_code("TIME_TIMEZONE_CONFLICT", timestamp_format="%Y-%m-%d %H:%M%z", timezone="UTC")
        self.assertIsNone(rules(timestamp_format="%Y-%m-%d %H:%M%z", timezone=None).timezone)

    def test_column_name_is_required(self):
        for value in ("", None, 3, "x" * 257):
            with self.subTest(value=value):
                self.assert_rule_code("TIME_RULES_INVALID", column=value)

    def test_datetime_values_without_format_follow_the_same_rules(self):
        rule = rules(timestamp_format=None, timezone=None)
        aware = datetime(2026, 10, 19, tzinfo=timezone.utc)
        result = find_restorations([aware, None, aware + timedelta(days=2)], rule)
        self.assertEqual(result.times, (aware + timedelta(days=1),))

        with self.assertRaises(TimeRestoreError) as caught:
            find_restorations([datetime(2026, 10, 19), None, datetime(2026, 10, 21)], rule)

        self.assertEqual(caught.exception.code, "TIME_TIMEZONE_REQUIRED")


if __name__ == "__main__":
    unittest.main()

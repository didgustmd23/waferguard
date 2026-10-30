"""시각처럼 보이는 컬럼 질문이 표형 v1·v2의 지원 범위와 맞는지 검사한다. 합성 집계만 쓰고 실행·모델 호출은 없다."""

import unittest

from src.agents import llm_task_assist as rules, tabular_v2_task as v2


SUMMARY = {"version": 1, "row_count": 1000, "column_count": 4, "repeated_rows": 0}
UNSUPPORTED = "시간 형식을 지정할 수 없어"


def column(name, unique, group="string"):
    return {"column": name, "dtype_group": group, "missing_count": 0, "missing_ratio": 0, "observed_unique": unique,
            "finite_count": 1000 if group == "numeric" else 0}


COLUMNS = [column("event_time", 900), column("event_date", 12), column("other_time", 30), column("label", 2, "numeric")]


def ask(*, time_split, strategy, taken=False):
    split, preparation = rules._defaults()
    roles = {"event_time": "범주형", "event_date": "범주형", "other_time": "timestamp" if taken else "범주형", "label": "target"}
    plan = {"roles": roles, "split": dict(split, strategy=strategy), "preparation": preparation}
    return rules.plan_questions("label 예측", SUMMARY, COLUMNS, plan, time_split=time_split)


def pick(questions, prefix, subject):
    return next(item for item in questions if item["id"].startswith(prefix) and item["subjects"] == [subject])


def values(item):
    return [option["value"] for option in item["options"]]


class V2TimeQuestionTest(unittest.TestCase):
    def test_time_split_recommends_timestamp_with_settings(self):
        questions = ask(time_split=True, strategy="time")
        limit, candidate = pick(questions, "category_limit:", "event_time"), pick(questions, "time_candidate:", "event_date")
        for item in (limit, candidate):
            self.assertEqual(item["recommended"], "timestamp")
            self.assertIn("timestamp", values(item))
            self.assertIn("시간 설정", item["text"])
            self.assertNotIn(UNSUPPORTED, item["text"])
        self.assertEqual((limit["level"], candidate["level"]), ("advisory", "advisory"))
        self.assertEqual(values(candidate), ["제외", "범주형", "timestamp"])

    def test_other_split_offers_timestamp_but_recommends_exclude_with_the_condition(self):
        for strategy in ("random", "group", None):
            questions = ask(time_split=True, strategy=strategy)
            limit, candidate = pick(questions, "category_limit:", "event_time"), pick(questions, "time_candidate:", "event_date")
            for item in (limit, candidate):
                self.assertEqual(item["recommended"], "제외", strategy)
                self.assertIn("timestamp", values(item))
                self.assertIn("분할 방식 time 선택과 시간 설정이 함께", item["text"])
                self.assertNotIn(UNSUPPORTED, item["text"])
        # random에서 반드시 상한을 넘으면 필수 질문이 유지된다.
        self.assertEqual(pick(ask(time_split=True, strategy="random"), "category_limit:", "event_time")["level"], "required")
        self.assertEqual(values(pick(ask(time_split=True, strategy="random"), "category_limit:", "event_time")), ["제외", "timestamp"])

    def test_timestamp_taken_by_another_column_is_not_offered_again(self):
        questions = ask(time_split=True, strategy="time", taken=True)
        for item in (pick(questions, "category_limit:", "event_time"), pick(questions, "time_candidate:", "event_date")):
            self.assertEqual(item["recommended"], "제외")
            self.assertNotIn("timestamp", values(item))
            self.assertIn("다른 컬럼이 이미 timestamp 역할", item["text"])
            self.assertNotIn(UNSUPPORTED, item["text"])

    def test_existing_timestamp_role_is_required_only_outside_time_split(self):
        self.assertFalse([item for item in ask(time_split=True, strategy="time", taken=True) if item["id"].startswith("time_role:")])
        item = pick(ask(time_split=True, strategy="random", taken=True), "time_role:", "other_time")
        self.assertEqual((item["level"], item["recommended"], values(item)), ("required", "제외", ["제외"]))

    def test_adapter_uses_the_v2_contract(self):
        split, preparation = rules._defaults()
        record = {"goal": "label 예측", "diagnostic": {"summary": SUMMARY, "columns": COLUMNS}}
        plan = {"roles": {"event_time": "범주형", "event_date": "범주형", "other_time": "범주형", "label": "target"},
                "split": split, "preparation": preparation, "time": {}}
        item = pick(v2.questions(record, plan), "time_candidate:", "event_date")
        self.assertIn("timestamp", values(item))
        self.assertNotIn(UNSUPPORTED, item["text"])


class V1TimeQuestionTest(unittest.TestCase):
    def test_v1_keeps_the_unsupported_text_and_options(self):
        for strategy in ("random", "group", "time", None):
            for taken in (False, True):
                questions = ask(time_split=False, strategy=strategy, taken=taken)
                limit, candidate = pick(questions, "category_limit:", "event_time"), pick(questions, "time_candidate:", "event_date")
                self.assertEqual((candidate["level"], candidate["recommended"], values(candidate)), ("advisory", "제외", ["제외", "범주형"]))
                self.assertIn(UNSUPPORTED + " timestamp로 쓸 수 없습니다", candidate["text"])
                self.assertEqual(limit["recommended"], "제외")
                self.assertNotIn("timestamp", values(limit))
                self.assertIn(UNSUPPORTED + " timestamp로 쓸 수 없으니", limit["text"])

    def test_v1_timestamp_role_keeps_its_required_question(self):
        item = pick(ask(time_split=False, strategy="random", taken=True), "time_role:", "other_time")
        self.assertEqual((item["level"], item["recommended"], values(item)), ("required", "제외", ["제외"]))
        self.assertIn("통합 작업에서는 형식을 지정할 수 없어", item["text"])


if __name__ == "__main__":
    unittest.main()

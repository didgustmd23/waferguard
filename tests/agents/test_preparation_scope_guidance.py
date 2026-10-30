"""권장 지시문의 전처리 적용 범위 안내와 후보 영향 표·경고의 값 변화 구분. 합성 입력·합성 후보 집계만 쓰며 실제 모델·worker는 쓰지 않는다.

- 지시문: 표형(v1·v2) 요청에만 옵션 적용 범위·few_values 의미 안내가 들어가고 배열 요청에는 들어가지 않는다.
- 후보: 상수화·일부 병합·원래 상수·변화 없음을 저장 근거(변환 전 고유값·변환 후 상수·구간 경계·clip 경계)로 구분하고,
  실제로 적용된 설정만 관련 설정으로 안내한다.
"""

import json
import unittest
from unittest.mock import patch

from src.agents import llm_task_assist, tabular_task
from tests.agents import test_array_task as array_tests
from tests.agents.test_report_display_fix import _frames, _screen
from tests.agents.test_tabular_report_scope import record, sources
from tests.agents.test_tabular_task_v2 import TabularTaskV2Fixture


# 지시문에 들어가야 하는 의미(고정 핵심 어구). 문장 전체를 고정하지 않는다.
GUIDANCE = ("최종 역할", "binning은 수치형 특징", "범주형 인코딩이 아닙니다", "few_values는 관측값 종류가 적다", "단정하지 마세요")
IMPACT_TITLE = "예상 처리 영향 (Train 후보 계산, 영향이 있는 특징만)"
# 이미 검증한 7개 열 순서 뒤에 새 3개 열을 붙인다(고정 정답, 제품 상수에서 가져오지 않음).
IMPACT_HEADER = ["특징", "변환 전 고유값", "clipping 변경 값", "구간 수", "출력 열", "변환 후 상수", "정보 손실",
                 "변환 후 고유값", "값 변화", "관련 설정"]


class TabularGuidanceTest(TabularTaskV2Fixture):
    def test_v2_request_carries_scope_guidance_and_keeps_contract(self):
        task = self.diagnosed()
        payload = self.requests[0]
        system = payload["messages"][0]["content"]
        for phrase in GUIDANCE:
            self.assertIn(phrase, system)
        content = json.loads(payload["messages"][1]["content"])
        # 원본 행·경로는 보내지 않는다. 허용 값·응답 형식은 기존 계약 그대로 요청에 있다.
        self.assertEqual(set(content) - {"allowed_values_note"},
                         {"goal", "row_count", "repeated_rows", "subjects", "base_plan", "allowed_values", "scope_note"})
        self.assertIn("changes", payload["format"]["properties"])
        self.assertEqual(task["draft"]["source"], "model")

    def test_connection_failure_is_not_retried(self):
        task = self.diagnosed(requester=self.refusing)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual((task["draft"]["source"], task["draft"]["code"]), ("default", "OLLAMA_CONNECTION_FAILED"))

    def test_v1_shares_the_guidance(self):
        for phrase in GUIDANCE:
            self.assertIn(phrase, llm_task_assist.instructions())


class ArrayGuidanceTest(unittest.TestCase):
    """배열 요청에는 표형 전용 안내를 넣지 않는다(기존 배열 흐름 fixture를 빌려 실제 요청을 포착)."""
    flow = array_tests.ArrayTaskFlowTest
    setUp, transport, requester, reviewed = flow.setUp, flow.transport, flow.requester, flow.reviewed

    def test_array_request_has_no_tabular_guidance(self):
        self.reviewed()
        sent = "".join(self.requests)
        self.assertTrue(sent)
        for phrase in GUIDANCE:
            self.assertNotIn(phrase, sent)


def _impact(column, distinct, *, constant=False, clipped=None, missing=0):
    return {"column": column, "kind": "numeric", "width": 1, "missing": missing, "imputed": 0, "clipped": clipped,
            "distinct_before": distinct, "constant_after": constant, "unknown": 0, "missing_after": 0}


def _record(column, *, edges=None, clip=None):
    return {"column": column, "kind": "numeric", "impute": 0.0, "clip": clip, "center": 0.0, "scale": 1.0, "edges": edges, "width": 1}


def _candidate(impact, records, train_rows=100):
    candidate = {"binding": {"train_rows": train_rows, "rules": {}}, "removed": [], "retained": [item["column"] for item in impact],
                 "output_feature_count": len(impact), "impact": impact}
    return candidate, {"state": {"records": records}}


# 구간화만 켠 후보: s 상수화(2종→1), p 일부 병합(3종→구간 2), a 연속값 일부 병합(50종→구간 4), k 원래 상수, x 구간화 없음·변화 없음.
BINNING_ONLY = _candidate(
    [_impact("s", 2, constant=True), _impact("p", 3), _impact("a", 50), _impact("k", 1, constant=True), _impact("x", 40)],
    [_record("s", edges=[]), _record("p", edges=[2.0]), _record("a", edges=[1.0, 2.0, 3.0]), _record("k", edges=[]), _record("x")])


class ImpactChangeTest(unittest.TestCase):
    def table(self, expected):
        return {row["특징"]: row for row in expected["table"]}

    def test_binning_only_distinguishes_changes_and_names_only_binning(self):
        expected = tabular_task._expected_impact(*BINNING_ONLY)
        rows = self.table(expected)
        self.assertEqual((rows["s"]["변환 후 고유값"], rows["s"]["값 변화"], rows["s"]["관련 설정"]), ("1", "상수화(여러 값 → 1)", "구간화"))
        self.assertEqual((rows["p"]["변환 후 고유값"], rows["p"]["값 변화"], rows["p"]["관련 설정"]), ("최대 2", "일부 병합(3종 → 구간 2개)", "구간화"))
        self.assertEqual(rows["a"]["값 변화"], "일부 병합(50종 → 구간 4개)")
        self.assertEqual((rows["k"]["값 변화"], rows["k"]["정보 손실"], rows["k"]["관련 설정"]), ("원래 상수(변환 전 1종)", "-", "-"))
        self.assertNotIn("x", rows)
        warnings = " ".join(tabular_task._impact_warnings(expected, 0))
        self.assertIn("s(구간화)", warnings)
        self.assertNotIn("k(", warnings)
        self.assertIn("p(3종 → 구간 2개)", warnings)
        self.assertIn("의도된 결과일 수 있", warnings)
        # 구간화만 켰으면 관련 없는 clipping 해제를 해결 예시로 들지 않는다.
        self.assertNotIn("극단값 자르기", warnings)
        self.assertNotIn("clipping", warnings)

    def test_binning_and_clipping_together_are_not_reduced_to_one_cause(self):
        expected = tabular_task._expected_impact(*_candidate(
            [_impact("b", 4, constant=True, clipped=3)], [_record("b", edges=[], clip=[1.0, 1.0])]))
        row = self.table(expected)["b"]
        self.assertEqual(row["관련 설정"], "구간화 · 극단값 자르기")
        warnings = " ".join(tabular_task._impact_warnings(expected, 0))
        self.assertIn("b(구간화 · 극단값 자르기)", warnings)
        self.assertIn("분리하지 않았", warnings)

    def test_clipping_only_names_clipping(self):
        expected = tabular_task._expected_impact(*_candidate(
            [_impact("c", 5, constant=True, clipped=4)], [_record("c", clip=[2.0, 2.0])]))
        self.assertEqual(self.table(expected)["c"]["관련 설정"], "극단값 자르기")
        warnings = " ".join(tabular_task._impact_warnings(expected, 0))
        self.assertIn("c(극단값 자르기)", warnings)
        self.assertNotIn("구간화", warnings)

    def test_previous_candidate_without_impact_keeps_notice(self):
        candidate, preprocessor = BINNING_ONLY
        old = dict(candidate)
        old.pop("impact")
        expected = tabular_task._expected_impact(old, preprocessor)
        self.assertEqual(tabular_task._impact_warnings(expected, 0), ["처리 영향(clipping 변경 값·변환 후 상수)은 " + expected["missing"] + "입니다."])

    def test_apply_screen_shows_new_columns_in_fixed_order(self):
        from src.agents import task_orchestrator
        from src.ui import task_view

        value = sources()
        candidate, preprocessor = value["candidate"], value["preprocessor"]
        candidate["impact"][0].update(distinct_before=3, constant_after=True)
        preprocessor["state"]["records"][0].update(edges=[])
        with patch.object(tabular_task, "_origin", return_value=("기본 계획", None)):
            built = tabular_task.review_facts(record(), candidate, preprocessor, None)
        stored = json.loads(json.dumps({"signature": "s", "facts": built}, ensure_ascii=False, sort_keys=True))
        task = {"task_id": "0" * 32, "kind": "tabular_v2", "attempt": 1, "phase": "awaiting_apply", "candidate": stored}

        def render(st):
            with patch.object(task_orchestrator, "apply_revisable", return_value=False):
                task_view._render_apply(st, task)

        frame = next(frame for frame in _frames(_screen(render)) if "값 변화" in frame.columns)
        self.assertEqual(list(frame.columns), IMPACT_HEADER)
        self.assertEqual(frame.values.tolist()[0], ["x", "3", "미집계", "1", "1", "예", "예", "1", "상수화(여러 값 → 1)", "구간화"])

    def apply_screen(self, candidate, preprocessor):
        """review_facts로 만든 후보를 저장 형식(JSON 왕복)으로 바꿔 결정 D 화면을 AppTest로 그린다."""
        from src.agents import task_orchestrator
        from src.ui import task_view

        with patch.object(tabular_task, "_origin", return_value=("기본 계획", None)):
            built = tabular_task.review_facts(record(), candidate, preprocessor, None)
        stored = json.loads(json.dumps({"signature": "s", "facts": built}, ensure_ascii=False, sort_keys=True))
        task = {"task_id": "0" * 32, "kind": "tabular_v2", "attempt": 1, "phase": "awaiting_apply", "candidate": stored}

        def render(st):
            with patch.object(task_orchestrator, "apply_revisable", return_value=False):
                task_view._render_apply(st, task)

        return _screen(render)

    def test_max_bins_note_follows_impact_table(self):
        value = sources()
        value["candidate"]["impact"][0].update(distinct_before=5, constant_after=False)
        value["preprocessor"]["state"]["records"][0].update(edges=[1.0, 2.0])
        elements = self.apply_screen(value["candidate"], value["preprocessor"])
        kinds = [kind for kind, _ in elements]
        index = next(i for i, (kind, item) in enumerate(elements) if kind == "dataframe" and "값 변화" in item.columns)
        frame = elements[index][1]
        self.assertEqual(list(frame.columns), IMPACT_HEADER)
        # c는 원핫(출력 2열)이라 표에 남지만 변환 후 고유값은 세지 않았으므로 상한이 아닌 미집계다.
        self.assertEqual(frame.values.tolist(), [["x", "5", "미집계", "3", "1", "아니오", "-", "최대 3", "일부 병합(5종 → 구간 3개)", "구간화"],
                                                 ["c", "2", "미집계", "-", "2", "아니오", "-", "미집계", "-", "-"]])
        # 상한 설명은 영향 표 바로 다음 요소이고, 다음 표(출력 폭) 제목보다 앞이다.
        self.assertEqual(kinds[index + 1], "caption")
        note = elements[index + 1][1]
        for phrase in ("'최대 N'은 저장된 구간 수로 계산한 상한", "실제 변환 결과에서 센 고유값 수가 아닙니다", "'1'은 변환 후 상수", "'미집계'는 세지 않은 값"):
            self.assertIn(phrase, note)
        # 화면의 N은 같은 행의 구간 수(저장된 구간 경계 + 1)와 같다.
        self.assertEqual(frame.values.tolist()[0][7], "최대 " + frame.values.tolist()[0][3])
        headings = [item for kind, item in elements if kind == "markdown" and item.startswith("#### ")]
        self.assertEqual(headings, ["#### 특징별 처리", "#### 제거 후보", "#### " + IMPACT_TITLE, "#### 출력 폭 (구간화·원핫)",
                                    "#### 실제 실행 설정 (후보 계산에 쓴 값)"])
        self.assertEqual(sum(1 for kind, item in elements if kind == "caption" and "최대 N" in item), 1)
        # 경고 문장은 그대로 있다(일부 병합 안내).
        self.assertTrue(any(kind == "text" and "일부 값이 같은 구간에 모이는" in item for kind, item in elements))

    def test_constant_only_candidate_shows_no_max_note(self):
        value = sources()
        value["candidate"]["impact"][0].update(distinct_before=3, constant_after=True)
        value["preprocessor"]["state"]["records"][0].update(edges=[])
        elements = self.apply_screen(value["candidate"], value["preprocessor"])
        frame = next(item for kind, item in elements if kind == "dataframe" and "값 변화" in item.columns)
        # 상수화는 확정값 1이며 상한 설명을 붙이지 않는다.
        self.assertEqual(frame.values.tolist()[0][7:], ["1", "상수화(여러 값 → 1)", "구간화"])
        self.assertFalse(any("최대 N" in str(item) for _, item in elements))

    def test_previous_candidate_screen_omits_table_and_note(self):
        value = sources()
        value["candidate"].pop("impact")
        value["preprocessor"]["state"]["records"][0].update(edges=[1.0])
        elements = self.apply_screen(value["candidate"], value["preprocessor"])
        self.assertFalse(any(kind == "markdown" and IMPACT_TITLE in item for kind, item in elements))
        self.assertFalse(any("최대" in str(item) for _, item in elements))
        self.assertTrue(any(kind == "text" and tabular_task.OLD_CANDIDATE in item for kind, item in elements))

    def test_report_warnings_use_actual_settings(self):
        from tests.agents.test_tabular_report_scope import facts

        value = sources()
        value["candidate"]["impact"][0].update(distinct_before=3, constant_after=True)
        value["preprocessor"]["state"]["records"][0].update(edges=[])
        texts = [item["내용"] for item in facts(record(), value)["warnings"]]
        self.assertTrue(any("x(구간화)" in text for text in texts), texts)
        self.assertFalse(any("극단값 자르기 none" in text for text in texts), texts)


if __name__ == "__main__":
    unittest.main()

"""계획 확인 화면의 권장안 거부 안내·검색 근거 실패 문구·역할 열 순서를 검사한다. 모델·worker 호출은 없다."""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.agents import tabular_v2_task, task_rag, task_recommend
from src.agents.task_report import REJECTED_GUIDANCE, code_label, code_text, recommendation_guidance, recommendation_notice
from src.ui import tabular_task_form
from src.ui.task_view import default_plan_line


def record(code, **draft):
    return {"kind": "tabular_v2", "draft": {"source": "default", "code": code, **draft}}


def usage(status, code=None, mode="on", **values):
    return task_rag._usage(mode, status, code, **values)


class RecommendationNoticeTest(unittest.TestCase):
    def test_change_invalid_shows_code_and_every_item_rule_without_claiming_a_cause(self):
        text = code_text(record(None), "RECOMMENDATION_CHANGE_INVALID")
        self.assertTrue(text.endswith("(RECOMMENDATION_CHANGE_INVALID)"))
        for words in ("항목 형식 오류", "허용되지 않은 경로·값", "같은 경로 중복", "하위 원인은 기록되지 않았습니다"):
            self.assertIn(words, text)

    def test_unregistered_code_is_not_copied_from_the_record(self):
        self.assertEqual(code_text(record(None), "<모델 값>"), "사유 코드 확인 불가")
        self.assertEqual(code_text(record(None), None), "사유 코드 확인 불가")

    def test_guidance_only_follows_model_response_rejections(self):
        for code in task_recommend.REJECTION_CODES:
            self.assertEqual(recommendation_guidance(record(code)), [REJECTED_GUIDANCE], code)
        for code in ("OLLAMA_CONNECTION_FAILED", "MODEL_REQUEST_FAILED", "RECOMMENDATION_REQUEST_TOO_LARGE", "SPLIT_ROLE_MISMATCH", None):
            self.assertEqual(recommendation_guidance(record(code)), [], code)
        for words in ("모델 변경 0건", "자동으로 다시 요청하지 않습니다", "아래 질문과 계획 양식"):
            self.assertIn(words, REJECTED_GUIDANCE)

    def test_other_codes_keep_their_label_and_gain_the_code(self):
        for code, label in (("OLLAMA_CONNECTION_FAILED", task_recommend.CODE_LABELS["OLLAMA_CONNECTION_FAILED"]),
                            ("RECOMMENDATION_TEXT_INVALID", task_recommend.CODE_LABELS["RECOMMENDATION_TEXT_INVALID"])):
            self.assertEqual(code_label(record(code), code), label)
            self.assertEqual(recommendation_notice(record(code)), [f"로컬 LLM 권장안을 쓰지 못해 기본 계획에서 시작했습니다: {label} ({code})."])

    def test_default_plan_line_does_not_nest_the_code_brackets(self):
        line = default_plan_line(record("RECOMMENDATION_CHANGE_INVALID"))
        self.assertTrue(line.startswith("권장안: 기본 계획 — 로컬 LLM 권장안을 사용할 수 없었습니다: 모델 변경 항목이"))
        self.assertTrue(line.endswith("하위 원인은 기록되지 않았습니다) (RECOMMENDATION_CHANGE_INVALID)"))
        self.assertNotIn("))", line)
        self.assertEqual(line.count("("), line.count(")"))

    def test_model_draft_has_no_notice_or_guidance(self):
        draft = {"kind": "tabular_v2", "draft": {"source": "model", "code": None}}
        self.assertEqual((recommendation_notice(draft), recommendation_guidance(draft)), ([], []))


class RagFailureTextTest(unittest.TestCase):
    def test_failed_head_does_not_blame_embedding_and_keeps_the_stored_reason(self):
        lines = task_rag.view({"rag": usage("failed", "RAG_SOURCE_CHANGED")})["lines"]
        self.assertEqual(lines[0], "검색 근거(실험적 Graph RAG): 선택 ON · 실제 결과: 검색 근거를 쓰지 못함 — 근거 없이 기존 방식으로 요청")
        self.assertNotIn("임베딩", lines[0])
        self.assertEqual(lines[1], "사유: " + task_rag.CODE_LABELS["RAG_SOURCE_CHANGED"] + " (RAG_SOURCE_CHANGED)")

    def test_embedding_failure_is_named_only_by_its_own_reason(self):
        lines = task_rag.view({"rag": usage("failed", "RAG_EMBED_CONNECTION_FAILED", embed_attempts=1)})["lines"]
        self.assertNotIn("임베딩", lines[0])
        self.assertEqual(lines[1], "사유: " + task_rag.CODE_LABELS["RAG_EMBED_CONNECTION_FAILED"] + " (RAG_EMBED_CONNECTION_FAILED)")

    def test_off_and_other_statuses_are_unchanged(self):
        self.assertEqual(task_rag.view({"rag": usage("off", mode="off")})["lines"],
                         ["검색 근거(실험적 Graph RAG): 선택 OFF · 실제 결과: 사용 안 함(OFF)"])
        self.assertIn("관련 근거 없음 — 근거 없이 기존 방식으로 요청", task_rag.view({"rag": usage("no_evidence")})["lines"][0])


class RoleColumnTest(unittest.TestCase):
    def setUp(self):
        columns = [dict(column=name, dtype_group="numeric", missing_ratio=0, observed_unique=2, finite_count=4) for name in ("a", "2urvived")]
        self.record = {"diagnostic": {"columns": columns}}
        self.fields = tabular_v2_task.plan_fields(self.record)
        self.plan = tabular_v2_task.base_plan(self.record)

    def render(self):
        st = self.st = MagicMock()
        st.columns.side_effect = lambda count: [st] * count
        st.number_input.side_effect = lambda label, *values, **_: values[2] if len(values) > 2 else 1
        st.selectbox.side_effect = lambda label, options, **_: list(options)[0]
        st.expander.return_value.__enter__.return_value = st

        def edit(table, **options):
            self.options, self.table = options, table
            return table.assign(role=["수치형", "target"])

        st.data_editor.side_effect = edit
        return tabular_task_form.render(st, self.record, self.plan, self.fields, "k")

    def test_role_follows_column_name_and_edited_target_is_returned(self):
        result = self.render()
        self.assertEqual(self.options["column_order"], ["column", "role", "dtype_group", "missing_ratio", "observed_unique", "finite_count"])
        self.assertEqual(self.options["disabled"], list(tabular_task_form.SHOWN))
        self.assertIn("target", self.st.column_config.SelectboxColumn.call_args.kwargs["options"])
        # 표 자료의 열은 그대로이고 편집한 역할이 계획으로 돌아온다.
        self.assertIsInstance(self.table, pd.DataFrame)
        self.assertEqual(list(self.table.columns), [*tabular_task_form.SHOWN, "role"])
        self.assertEqual(result["roles"], {"a": "수치형", "2urvived": "target"})


if __name__ == "__main__":
    unittest.main()

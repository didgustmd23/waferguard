"""LLM 권장안의 허용 역할·실행 계약 일치와 역할×분할 교차 거부 하위 진단. 모델 응답은 합성 문자열만 쓴다.

- v2: 수치 자료형이 아닌 컬럼은 요청 허용 값·응답 형식·병합 검사 모두에서 수치형 역할이 빠진다(v1 명세는 그대로).
- SPLIT_ROLE_MISMATCH: 원인별 고정 하위 코드와 분할 값만 기존 진단 구조로 남기고, 기존 우선순위·전체 거부·기본 계획 대체를 유지한다.
"""

import copy
import json
import unittest

from src.agents import task_adapters, task_recommend
from src.agents.llm_config import load_local_config
from src.agents.llm_task_assist import validate_plan
from src.agents.task_plan import PlanError
from src.agents.task_report import recommendation_notice
from tests.agents import test_task_orchestrator as base
from tests.agents.test_tabular_task_v2 import ROLES as V2_ROLES, TabularTaskV2Fixture


NOT_RECORDED = "세부 원인: 확인되지 않음(이 기록에는 하위 진단이 없습니다)"
SECRETS = ("의미에 맞는 역할", "비밀 이유 문장", "120행", "label을 target")


def _allowed(allowed, path):
    """개별 경로가 있으면 그 허용 값, 없으면 '구획.*' 허용 값(요청 본문의 우선순위 안내와 같은 규칙)."""
    return allowed[path] if path in allowed else allowed[path.split(".", 1)[0] + ".*"]


def _role_paths(branch):
    path = branch["properties"]["path"]
    return [path["const"]] if "const" in path else path["enum"]


class V2RoleSpecTest(TabularTaskV2Fixture):
    def test_string_columns_lose_numeric_role_in_request_schema_and_merge(self):
        task = self.diagnosed()
        # 허용 명세 안의 유효한 권장안(group 역할 + group 분할)은 그대로 채택된다.
        self.assertEqual((task["draft"]["source"], task["draft"].get("code")), ("model", None))
        self.assertEqual(task["draft"]["plan"]["roles"], V2_ROLES)

        columns = task["diagnostic"]["columns"]
        strings = sorted(row["column"] for row in columns if row["dtype_group"] != "numeric")
        numbers = sorted(row["column"] for row in columns if row["dtype_group"] == "numeric")
        self.assertTrue(strings and numbers, columns)
        self.assertIn("id", strings)

        payload = self.requests[0]
        content = json.loads(payload["messages"][1]["content"])
        allowed = content["allowed_values"]
        for name in strings:
            self.assertNotIn("수치형", _allowed(allowed, "roles." + name))
            self.assertIn("범주형", _allowed(allowed, "roles." + name))
        for name in numbers:
            self.assertIn("수치형", _allowed(allowed, "roles." + name))
        # 대상별 역할은 다수 허용 값을 'roles.*'로 묶고 다른 컬럼만 개별로 적는다. 이때만 우선순위 안내가 붙는다.
        self.assertIn("roles.*", allowed)
        self.assertEqual(sum(key.startswith("roles.") for key in allowed), 1 + min(len(strings), len(numbers)))
        self.assertEqual(content["allowed_values_note"], task_recommend.ALLOWED_OVERRIDE_NOTE)

        branches = payload["format"]["properties"]["changes"]["items"]["anyOf"]
        for branch in branches:
            options = branch["properties"]["value"].get("enum", [])
            for path in _role_paths(branch):
                if path.startswith("roles."):
                    self.assertEqual("수치형" in options, path[len("roles."):] in numbers, (path, options))

        system = payload["messages"][0]["content"]
        self.assertIn("숫자처럼 보이는 문자열도 숫자로 자동 변환하지 않으므로", system)
        self.assertIn("병합한 계획 전체가 이 제약을 만족해야 하며", system)

        # 병합 검사도 같은 명세를 쓴다. 문자열 컬럼의 수치형 제안은 계획 검증 전에 권장안 전체가 거부된다.
        stored = copy.deepcopy(task)
        draft = base.request_draft(stored, lambda payload, config: base.model_reply(dict(V2_ROLES, id="수치형")))
        adapter = task_adapters.adapter("tabular_v2")
        self.assertEqual((draft["source"], draft["code"], draft["changes"]), ("default", "RECOMMENDATION_CHANGE_INVALID", []))
        self.assertEqual(draft["plan"], adapter.base_plan(stored))
        self.assertNotIn("rejection", draft)
        # 원본 자료형·진단 값은 바꾸지 않는다.
        self.assertEqual(stored["diagnostic"], task["diagnostic"])
        text = json.dumps(draft, ensure_ascii=False)
        for secret in SECRETS:
            self.assertNotIn(secret, text)

        # v1 명세·지시문은 그대로다(모든 컬럼에 전체 역할).
        v1 = task_adapters.adapter("tabular_csv")
        v1_fields = {field["path"]: field for field in v1.plan_fields(stored)}
        self.assertIn("수치형", v1_fields["roles.id"]["options"])
        self.assertNotIn("숫자로 자동 변환하지 않", v1.recommendation_input(stored)["instructions"])
        v1_payload = task_recommend.plan_request(stored["goal"], v1.recommendation_input(stored), v1.base_plan(stored),
                                                 v1.plan_fields(stored), load_local_config())[0]
        v1_content = json.loads(v1_payload["messages"][1]["content"])
        self.assertNotIn("allowed_values_note", v1_content)
        self.assertEqual([key for key in v1_content["allowed_values"] if key.startswith("roles.")], ["roles.*"])
        v2_fields = {field["path"]: field for field in adapter.plan_fields(stored)}
        self.assertEqual(v2_fields["roles.id"]["options"], [o for o in v1_fields["roles.id"]["options"] if o != "수치형"])
        # 분할·전처리 명세는 같다. v2만 계획 단계에서 time 분할을 고를 수 있고(미지원 표시 없음), 사용자 전용 시간 설정 항목이 있다.
        shared = {path: field for path, field in v2_fields.items() if not path.startswith(("roles.", "time."))}
        self.assertEqual(shared, {path: field for path, field in v1_fields.items()
                                  if not path.startswith("roles.") and path != "split.strategy"} | {"split.strategy": shared["split.strategy"]})
        self.assertEqual(v1_fields["split.strategy"], dict(shared["split.strategy"], unsupported=["time"]))
        self.assertTrue(all(field["user_only"] for path, field in v2_fields.items() if path.startswith("time.")))
        self.assertFalse([path for path in v1_fields if path.startswith("time.")])

    def test_old_records_without_sub_diagnostic_keep_default_plan_notice(self):
        stored = self.diagnosed()
        draft = dict(stored["draft"], source="default")
        numeric = dict(stored, draft=dict(draft, code="NUMERIC_ROLE_TYPE_UNSUPPORTED"))
        lines = recommendation_notice(numeric)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("로컬 LLM 권장안을 쓰지 못해 기본 계획에서 시작했습니다"), lines)
        mismatch = recommendation_notice(dict(stored, draft=dict(draft, code="SPLIT_ROLE_MISMATCH")))
        self.assertTrue(mismatch[0].startswith("로컬 LLM 권장안 전체를 거부해 기본 계획에서 시작했습니다"), mismatch)
        self.assertEqual(mismatch[1:], [NOT_RECORDED])


class AllowedValuesTest(unittest.TestCase):
    """공통 허용 값 표기: 대상별 경로만 묶고, 대상이 없는 유형(배열)·균일 구획(v1)은 이전과 같은 표기다."""

    def enum(self, path, options):
        return {"path": path, "type": "enum", "options": list(options)}

    def test_subject_paths_collapse_to_majority_with_exceptions(self):
        fields = [self.enum("roles.a", "xyz"), self.enum("roles.b", "xyz"), self.enum("roles.c", "xz"),
                  self.enum("split.strategy", ("random", "group")), self.enum("split.other", ("random", "group"))]
        subjects = {"roles.a", "roles.b", "roles.c"}
        allowed, overridden = task_recommend._allowed_values(fields, subjects)
        self.assertEqual(allowed, {"roles.*": list("xyz"), "roles.c": list("xz"), "split.*": ["random", "group"]})
        self.assertTrue(overridden)
        # 대상 경로가 없거나 하나뿐이면 이전처럼 경로별로 적는다.
        self.assertEqual(task_recommend.allowed_values(fields[:3]), {"roles.a": list("xyz"), "roles.b": list("xyz"),
                                                                     "roles.c": list("xz")})
        self.assertFalse(task_recommend._allowed_values(fields[:3], ())[1])
        mixed = [self.enum("mapping.label", "ab"), self.enum("mapping.group", "ab"), self.enum("mapping.samples", "c")]
        self.assertEqual(task_recommend.allowed_values(mixed, ()), {"mapping.label": list("ab"), "mapping.group": list("ab"),
                                                                    "mapping.samples": ["c"]})
        # '*'이거나 '.*'로 끝나는 이름의 대상이 있으면 묶지 않고, 우선순위 안내도 붙이지 않는다.
        for odd in ("roles.*", "roles.x.*"):
            with self.subTest(path=odd):
                odd_fields = [*fields[:3], self.enum(odd, "xz")]
                allowed, overridden = task_recommend._allowed_values(odd_fields, {*subjects, odd})
                self.assertEqual(allowed, {field["path"]: list(field["options"]) for field in odd_fields})
                self.assertFalse(overridden)
        # 요약만 제공한 대상은 model_fields에서 빠지므로 묶음·예외·안내 어디에도 이름이 나오지 않는다.
        allowed, _ = task_recommend._allowed_values(fields[:2], subjects)
        self.assertNotIn("roles.c", allowed)


class RoleSplitDiagnosticTest(unittest.TestCase):
    setUp_csv = base.TaskOrchestratorTest.setUp_csv
    start_patch = base.TaskOrchestratorTest.start_patch
    transport = base.TaskOrchestratorTest.transport
    setUp = base.TaskOrchestratorTest.setUp
    requester = base.TaskOrchestratorTest.requester
    new_task = base.TaskOrchestratorTest.new_task

    def diagnosed(self):
        from src.agents import task_orchestrator as orchestrator

        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        return task

    def test_cross_errors_get_fixed_sub_codes_in_existing_priority(self):
        columns = self.diagnosed()["diagnostic"]["columns"]
        roles, split = base.ROLES, base.SPLIT
        grouped, no_target = dict(roles, time="group"), dict(roles, label="수치형")
        cases = {
            "required_missing": (roles, dict(split, strategy="group"), "SPLIT_ROLE_MISMATCH", "SPLIT_ROLE_REQUIRED_MISSING"),
            "group_strategy": (grouped, split, "SPLIT_ROLE_MISMATCH", "SPLIT_GROUP_STRATEGY"),
            # group 역할 불일치가 계층 조건보다 먼저다.
            "group_before_stratify": (dict(grouped, label="수치형"), dict(split, stratify=True), "SPLIT_ROLE_MISMATCH",
                                      "SPLIT_GROUP_STRATEGY"),
            "target_missing": (no_target, dict(split, stratify=True), "SPLIT_ROLE_MISMATCH", "SPLIT_STRATIFY_TARGET_MISSING"),
            # 계층 분할+비random은 분할 계약 단계가 먼저 거부한다(필요 역할 누락보다 앞).
            "stratify_strategy": (roles, dict(split, strategy="group", stratify=True), "SPLIT_ROLE_MISMATCH",
                                  "SPLIT_STRATIFY_STRATEGY"),
            # 비율 오류는 역할×분할 교차 검사보다 먼저이고 기존 분할 진단을 그대로 남긴다.
            "ratio_sum_first": (roles, dict(split, strategy="group", train_ratio=0.6), "SPLIT_INVALID", "SPLIT_RATIO_SUM"),
            "ratio_range_first": (grouped, dict(split, test_ratio=0.0, train_ratio=0.85), "SPLIT_INVALID", "SPLIT_RATIO_RANGE"),
        }

        for name, (case_roles, case_split, code, reason) in cases.items():
            with self.subTest(case=name), self.assertRaises(PlanError) as caught:
                validate_plan(columns, base.plan(case_roles, case_split))
            error = caught.exception
            self.assertEqual((error.code, error.detail["reason"], error.detail["section"]), (code, reason, "split"))
            self.assertEqual(set(error.detail), {"section", "reason", "values", "omitted"})
            self.assertEqual(set(error.detail["values"]) | set(error.detail["omitted"]),
                             {"train_ratio", "validation_ratio", "test_ratio", "strategy", "stratify"})
            rejection = task_recommend._rejection(error.detail, [])
            self.assertTrue(task_adapters.adapter("tabular_csv").rejection_valid(code, rejection), rejection)

        # 진단 없는 거부(역할 계약·분할 방식 없음·time 미지원)는 그대로 진단이 없다.
        plain = {"ROLES_INVALID": (dict(roles, id="target"), dict(split, strategy="group")),
                 "SPLIT_STRATEGY_MISSING": (roles, dict(split, strategy=None)),
                 "TIME_SPLIT_UNSUPPORTED": (roles, dict(split, strategy="time"))}
        for code, (case_roles, case_split) in plain.items():
            with self.subTest(code=code), self.assertRaises(PlanError) as caught:
                validate_plan(columns, base.plan(case_roles, case_split))
            self.assertEqual((caught.exception.code, caught.exception.detail), (code, None))

    def test_recommendation_is_rejected_whole_with_sub_code_and_valid_one_is_adopted(self):
        task = self.diagnosed()
        adapter = task_adapters.adapter("tabular_csv")
        base_plan = adapter.base_plan(task)
        grouped, no_target = dict(base.ROLES, time="group"), dict(base.ROLES, label="수치형")

        def change(path, value):
            return {"path": "split." + path, "value": value, "reason": "비밀 이유 문장"}

        cases = {
            "SPLIT_GROUP_STRATEGY": (grouped, [], []),
            "SPLIT_ROLE_REQUIRED_MISSING": (base.ROLES, [change("strategy", "group")], ["strategy"]),
            "SPLIT_STRATIFY_TARGET_MISSING": (no_target, [change("stratify", True)], ["stratify"]),
        }
        for reason, (roles, changes, from_model) in cases.items():
            with self.subTest(reason=reason):
                draft = base.request_draft(task, lambda payload, config, roles=roles, changes=changes:
                                           base.model_reply(roles, changes=changes))
                self.assertEqual((draft["source"], draft["code"], draft["changes"], draft["plan"]),
                                 ("default", "SPLIT_ROLE_MISMATCH", [], base_plan))
                rejection = draft["rejection"]
                self.assertEqual((rejection["reason"], rejection["from_model"], rejection["omitted"]), (reason, from_model, []))
                text = json.dumps(draft, ensure_ascii=False)
                for secret in SECRETS:
                    self.assertNotIn(secret, text)
                # 역할 컬럼명은 진단에 넣지 않는다.
                words = {*rejection, *rejection["values"], *rejection["omitted"], *rejection["from_model"],
                         *(value for value in rejection.values() if type(value) is str),
                         *(value for value in rejection["values"].values() if type(value) is str)}
                self.assertFalse(words & {"label", "id", "x", "time"}, words)
                lines = recommendation_notice(dict(task, draft=draft))
                self.assertIn("세부 원인: " + adapter.CODE_LABELS[reason], lines)
                self.assertTrue(any(line.startswith("거부한 분할 값(병합 결과):") for line in lines), lines)

        # 역할과 분할 방식을 함께 바꾼 유효한 권장안은 그대로 채택된다(자동 보정·부분 채택 아님).
        draft = base.request_draft(task, lambda payload, config: base.model_reply(grouped, changes=[change("strategy", "group")]))
        self.assertEqual((draft["source"], draft.get("code")), ("model", None))
        self.assertEqual((draft["plan"]["roles"]["time"], draft["plan"]["split"]["strategy"]), ("group", "group"))

    def test_tampered_role_split_diagnostics_are_not_displayed(self):
        task = self.diagnosed()
        adapter = task_adapters.adapter("tabular_csv")
        draft = base.request_draft(task, lambda payload, config: base.model_reply(dict(base.ROLES, time="group")))
        rejection = draft["rejection"]
        self.assertEqual(rejection["reason"], "SPLIT_GROUP_STRATEGY")
        self.assertTrue(adapter.rejection_valid("SPLIT_ROLE_MISMATCH", rejection))

        values = rejection["values"]
        broken = {
            "required_missing_on_random": dict(rejection, reason="SPLIT_ROLE_REQUIRED_MISSING"),
            "target_missing_without_stratify": dict(rejection, reason="SPLIT_STRATIFY_TARGET_MISSING"),
            "group_strategy_on_group": dict(rejection, values=dict(values, strategy="group")),
            "required_missing_on_time": dict(rejection, reason="SPLIT_ROLE_REQUIRED_MISSING", values=dict(values, strategy="time")),
            "required_missing_with_stratify": dict(rejection, reason="SPLIT_ROLE_REQUIRED_MISSING",
                                                   values=dict(values, strategy="group", stratify=True)),
            "group_strategy_bad_ratio_sum": dict(rejection, values=dict(values, train_ratio=0.6)),
            "unknown_reason": dict(rejection, reason="SPLIT_ROLE_COLUMN"),
            "role_name_added": dict(rejection, values=dict(values, group_column="time")),
            "free_text_added": dict(rejection, message="모델 원문"),
        }
        for name, item in broken.items():
            with self.subTest(case=name):
                self.assertFalse(adapter.rejection_valid("SPLIT_ROLE_MISMATCH", item))
                self.assertEqual(recommendation_notice(dict(task, draft=dict(draft, rejection=item)))[1:], [base.UNREADABLE_REJECTION])
        # 새 하위 코드는 SPLIT_ROLE_MISMATCH에서만 허용된다.
        self.assertFalse(adapter.rejection_valid("SPLIT_INVALID", rejection))
        # 배열 유형은 같은 진단을 해석하지 않는다.
        self.assertEqual(recommendation_notice({"kind": "array_npz", "draft": dict(draft)})[1:], [base.UNREADABLE_REJECTION])


if __name__ == "__main__":
    unittest.main()

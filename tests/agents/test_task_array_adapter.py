"""도메인 없는 합성 배열 어댑터로 공통 조정기·질문·위임 승인·보고서·화면이 표형에 기대지 않는지 검증한다.

표형과 같은 공개 함수(start_task·diagnose·review_state·check_plan·confirm_plan·approve_apply)와 같은 화면만 쓴다.
테스트만의 우회 경로는 없다. 모델 응답만 모의한다.
"""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents import task_adapters, task_consent, task_orchestrator as orchestrator, task_store
from src.agents.llm_config import load_local_config
from src.agents.task_plan import PlanError
from tests.agents import array_task_fixture as fixture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"wall_seconds": 60, "memory_mib": 2048, "processes": 2}


def samples(groups=6):
    items = []

    for index in range(24):
        # 첫 칸만 표본마다 다르게 두어 중복은 마지막에 덧붙인 표본 하나뿐이다.
        array = [[index if (row, column) == (0, 0) else (row + column) % 3 for column in range(4)] for row in range(4)]
        label = None if index in (5, 11) else ("a", "b", "c")[index % 3]
        items.append({"sample_id": f"s{index:02d}", "array": array, "label": label, "group": f"g{index % groups}"})

    items.append(dict(items[0], sample_id="s24"))
    return items


def reply(changes=(), questions=()):
    return json.dumps({"summary": ["표본 25개, 라벨 결측 2개"], "notes": [], "changes": list(changes),
                       "questions": list(questions)}, ensure_ascii=False)


class ArrayAdapterTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = self.root / "tasks"
        fixture.ROOT = self.root / "runs"
        patcher = patch.dict(task_adapters.ADAPTERS, {fixture.KIND: fixture})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.model = asdict(load_local_config())

    def source(self, groups=6):
        path = self.root / ("samples%d.arrays.json" % groups)
        path.write_text(json.dumps(samples(groups)), encoding="utf-8")
        return path

    def diagnosed(self, response, groups=6):
        task = orchestrator.start_task("라벨 분류용으로 표본을 정리해 주세요", self.source(groups), LIMITS, self.model, root=self.store)
        return orchestrator.diagnose(task["task_id"], root=self.store, requester=lambda payload, config: response)

    def test_same_common_path_from_request_to_report(self):
        task = self.diagnosed(reply([{"path": "options.group_split", "value": True, "reason": "그룹 누수 방지"}]))
        self.assertEqual((task["kind"], task["phase"], task["draft"]["source"]), ("array_synthetic", "awaiting_review", "model"))
        self.assertEqual(task["diagnostic"]["summary"]["label_missing"], 2)
        state = orchestrator.review_state(task)
        self.assertTrue(state["plan"]["options"]["group_split"])
        self.assertEqual([(item["id"], item["level"]) for item in state["questions"]],
                         [("options.missing_labels", "required"), ("options.drop_duplicates", "advisory")])

        # 필수 질문에 답하지 않으면 서버가 확정을 거부하고 아무것도 만들지 않는다.
        with self.assertRaises(PlanError) as caught:
            orchestrator.confirm_plan(task["task_id"], state["plan"], {}, root=self.store)

        self.assertEqual(caught.exception.code, "REQUIRED_QUESTION_UNANSWERED")
        self.assertFalse((fixture.ROOT / "array_review").exists())

        answers = {"options.missing_labels": "drop", "options.drop_duplicates": True}
        task = orchestrator.confirm_plan(task["task_id"], state["plan"], answers, root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(task["confirmed"]["plan"]["options"], {"drop_duplicates": True, "group_split": True, "missing_labels": "drop"})
        metrics = {item["label"]: item["value"] for item in task["candidate"]["facts"]["metrics"]}
        self.assertEqual(metrics, {"유지 표본": 22, "중복 제외": 1, "라벨 없음 제외": 2})
        # 위임 범위는 이 유형의 확정 계획에서 계산한다. 적용 단계는 C에 들어가지 않는다.
        self.assertEqual(task["consents"]["C"]["content"]["allowed"], [["array_review", "derive_preparation"]])
        self.assertEqual(task["consents"]["C"]["content"]["answers"],
                         [{"id": "options.drop_duplicates", "path": "options.drop_duplicates", "value": True},
                          {"id": "options.missing_labels", "path": "options.missing_labels", "value": "drop"}])

        with self.assertRaises(ValueError):
            task_consent.authorizer(task, "C", "array_apply", "apply_preparation", expected={})

        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        facts = task["report"]["facts"]
        self.assertEqual({item["label"]: item["value"] for item in facts["metrics"]}, {"입력 표본": 25, "정리 후 표본": 22})
        self.assertIn("미집계", task["report"]["text"])
        self.assertTrue(any("질문 답변으로 정한 항목: options.missing_labels = drop" in text for text in facts["cautions"]))

        # 공통 위임 발급 경로를 썼고 직접 확인 이벤트는 없다. 각 단계 승인은 1회씩 발급·소비됐다.
        events = "".join(path.read_text(encoding="utf-8") for path in fixture.ROOT.rglob("events.jsonl"))
        self.assertEqual(events.count("DELEGATED_APPROVAL_RECORDED"), 3)
        self.assertNotIn("LOCAL_REVIEW_CONFIRMED", events)
        self.assertEqual({key for key in task["consents"]}, {"A", "C", "D"})

    def test_recommendation_is_rejected_as_a_whole_when_merged_plan_fails(self):
        task = self.diagnosed(reply([{"path": "options.drop_duplicates", "value": True, "reason": "중복"},
                                     {"path": "options.group_split", "value": True, "reason": "그룹"}]), groups=1)
        self.assertEqual((task["draft"]["source"], task["draft"]["code"]), ("default", "GROUP_MISSING"))
        self.assertEqual(task["draft"]["plan"], fixture.base_plan(task))

    def test_apply_target_change_blocks_decision_d(self):
        task = self.diagnosed(reply())
        state = orchestrator.review_state(task)
        task = orchestrator.confirm_plan(task["task_id"], state["plan"], {"options.missing_labels": "keep"}, root=self.store)

        with patch.object(fixture, "review_signature", return_value="changed"):
            task = orchestrator.approve_apply(task["task_id"], root=self.store)

        self.assertEqual(task["problem"]["code"], "CANDIDATE_CHANGED")
        self.assertNotIn("D", task["consents"])


class ArrayScreenTest(unittest.TestCase):
    setUp = ArrayAdapterTest.setUp
    source = ArrayAdapterTest.source
    diagnosed = ArrayAdapterTest.diagnosed

    def test_generic_form_questions_and_result_on_same_screen(self):
        from streamlit.testing.v1 import AppTest

        task = self.diagnosed(reply())

        with patch("src.agents.task_store.STORE_ROOT", new=self.store), patch(
                "src.ui.runtime_notice.server_permissions", return_value={"elevated": True, "pid": 1, "python": "synthetic"}):
            app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=120).run()
            self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
            app.session_state["ui_task_id"] = task["task_id"]
            app.run()
            self.assertFalse(app.exception, [item.value for item in app.exception])
            self.assertTrue(any("[필수]" in item.value and "라벨이 없는 표본" in item.value for item in app.text))
            # 전용 부품이 없는 유형은 항목 명세로 그린 일반 양식을 쓴다.
            self.assertTrue(any(item.value == "항목 2: options.group_split" for item in app.text))
            self.assertTrue(any(item.label == "항목 2" for item in app.checkbox))
            answer = next(item for item in app.selectbox if item.label == "질문 1 답변")
            answer.set_value("제외")
            # 공통 화면은 최종 계획을 먼저 미리 본 뒤 동의·확정한다(2026-10-22).
            next(item for item in app.button if item.label == "최종 계획 미리보기").click().run()
            self.assertFalse(app.exception, [item.value for item in app.exception])
            next(item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")).check()
            next(item for item in app.button if item.label == "확인하고 진행").click().run()
            self.assertFalse(app.exception, [item.value for item in app.exception])
            self.assertTrue(any("필요한 결정 — 적용" in item.value for item in app.subheader))
            app.checkbox(key="task_consent_d").check()
            next(item for item in app.button if item.label == "적용하고 마무리").click().run()
            self.assertFalse(app.exception, [item.value for item in app.exception])
            self.assertTrue(any(item.value == "최종 결과" for item in app.subheader))
            # 유형 어댑터가 새 절을 주지 않아도 공통 결과 틀(2026-10-22)로 그린다. 이전 형식 보고서로 보지 않는다.
            self.assertIn("#### 실제 전후 비교", [item.value for item in app.markdown])
            self.assertFalse(any(item.value.startswith("이전 형식의 보고서") for item in app.info))

        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["phase"], "completed")



class CommonBoundaryTest(unittest.TestCase):
    """공통 조정기·화면이 유형별 해석·계산을 모르고 등록부와 어댑터 hook으로만 진행하는지 확인한다."""

    HOOKS = ("accepts", "consent_a_content", "consent_a_text", "diagnose", "base_plan", "plan_fields", "validate_plan", "questions",
             "prepare", "create_step", "service", "check_lineage", "review", "review_signature", "report_facts", "images",
             "consent_c_content", "consent_d_content", "expected", "approval_stages", "blocked_problem", "failure_hint")

    def test_registered_adapters_provide_common_hooks_and_input_resume_hooks_come_together(self):
        for kind, item in task_adapters.ADAPTERS.items():
            with self.subTest(kind=kind):
                self.assertEqual(item.KIND, kind)
                self.assertFalse([name for name in self.HOOKS if not callable(getattr(item, name, None))])
                resume = [callable(getattr(item, name, None)) for name in ("check_input_selection", "input_consent_content",
                                                                           "resume_input", "input_review")]
                self.assertIn(resume, ([True] * 4, [False] * 4))

    def test_each_generation_maps_one_format_to_one_kind_and_rejects_unknown_generations(self):
        for generation in ("legacy", "current"):
            kinds = [item["kind_id"] for item in task_adapters.input_specs(generation=generation)]
            for name in ("input.csv", "input.npz"):
                with self.subTest(generation=generation, name=name):
                    self.assertEqual(sum(task_adapters.adapter(kind).accepts(Path(name)) for kind in kinds), 1)
        for generation in ("", "v2", None, ["current"]):
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                task_adapters.input_specs(generation=generation)

    def test_common_coordinator_and_screen_have_no_format_parser_or_kind_branches(self):
        names = ("tabular_csv", "tabular_v2", "array_npz", "CsvReadSpec", "tabular_loaders", "tabular_input_flow", "read_csv",
                 "header_repair", "empty_field", "wafer", "WM-811K")
        for path in ("src/agents/task_orchestrator.py", "src/ui/task_view.py"):
            text = (PROJECT_ROOT / path).read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertFalse([name for name in names if name in text])


if __name__ == "__main__":
    unittest.main()

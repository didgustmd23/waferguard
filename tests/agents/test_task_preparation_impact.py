"""전처리 옵션 설명·처리 영향(예상/실측)·최종 계획 확정·결과 보고서를 합성 입력으로 검사한다(설계 7.22).

OS transport(AppContainer·Job)와 로컬 모델 응답만 모의한다. 후보·적용 worker 계산과 부모 검증기는 실제 코드다.
"""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.agents import task_orchestrator as orchestrator, task_store
from src.agents.preparation_candidate_report import candidate_markdown
from src.agents.preparation_candidate_results import read_candidate
from src.agents.task_plan import PlanError
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.transform import PreparationRules, TabularPreprocessor, impact_totals
from tests.agents import test_preparation_apply as apply_fixtures
from tests.agents import test_preparation_candidate as candidate_fixtures
from tests.agents import test_task_orchestrator as flow


def flag_frame(rows=120):
    """대부분 0이고 일부만 1인 정상 0/1 특징(flag)과 연속 특징(x). 실제 데이터가 아니다."""
    return pd.DataFrame({"x": [index * 0.37 % 11 for index in range(rows)],
                         "flag": [1 if index % 10 == 0 else 0 for index in range(rows)]})


class ImpactCalculationTest(unittest.TestCase):
    """후보·적용 worker가 쓰는 같은 영향 계산. 저장 기준만 쓰고 기준을 바꾸지 않는다."""

    def fit(self, frame, **rules):
        model = TabularPreprocessor(ColumnRoles(numeric=tuple(frame.columns)), PreparationRules(**rules))
        return model.fit(frame)

    def test_binary_feature_becomes_constant_under_iqr_clip(self):
        frame = flag_frame()
        model = self.fit(frame, outlier="iqr_clip")
        before = json.dumps(model.state_, sort_keys=True)
        items = {item["column"]: item for item in model.impact(frame)}
        self.assertEqual(json.dumps(model.state_, sort_keys=True), before)
        # 독립 계산: q1=q3=0이라 경계가 [0, 0]이고 1인 12개가 0으로 바뀐다.
        self.assertEqual(model.state_["records"][1]["clip"], [0.0, 0.0])
        self.assertEqual((items["flag"]["clipped"], items["flag"]["distinct_before"], items["flag"]["constant_after"]), (12, 2, True))
        self.assertFalse(items["x"]["constant_after"])
        self.assertEqual(impact_totals(model.impact(frame))["constant_features"], 1)

    def test_skewed_binary_quantile_binning_collapses_to_one_bin(self):
        frame = flag_frame()
        model = self.fit(frame, binning="quantile", n_bins=5, bin_encoding="onehot")
        items = {item["column"]: item for item in model.impact(frame)}
        self.assertEqual(model.state_["records"][1]["edges"], [])
        self.assertTrue(items["flag"]["constant_after"])
        self.assertIsNone(items["flag"]["clipped"])

    def test_negative_zero_output_counts_as_constant_for_worker_and_parent(self):
        """%.17g 표기에서 -0과 0은 글자가 다르지만 같은 값이다. 부모 대조가 정상 출력을 차단하지 않는다."""
        from src.agents.preparation_apply_results import _ImpactTally

        frame = pd.DataFrame({"z": [0.0, -0.0, 0.0, -0.0], "x": [1.0, 2.0, 3.0, 4.0]})
        model = self.fit(frame)
        totals = impact_totals(model.impact(frame))
        tally = _ImpactTally(model.state_["records"])

        for row in model.transform(frame).to_numpy():
            tally.add(["%.17g" % value for value in row])

        self.assertEqual(totals["constant_features"], 1)
        self.assertTrue(tally.matches(totals, model.state_["records"], len(frame)))

    def test_train_totals_must_match_candidate_impact(self):
        """적용 Train 합계는 같은 기준으로 센 후보 Train 합계와 같아야 한다. 이전 v1 후보는 대조할 값이 없다."""
        import tempfile

        from src.agents.preparation_apply_results import _matches_candidate

        frame = flag_frame()
        model = self.fit(frame, outlier="iqr_clip")
        impact = model.impact(frame)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.json"
            path.write_text(json.dumps({"schema_version": "preparation_candidate_report_v2", "impact": impact}), encoding="utf-8")
            self.assertTrue(_matches_candidate(path, impact_totals(impact)))
            self.assertFalse(_matches_candidate(path, dict(impact_totals(impact), clipped_values=impact_totals(impact)["clipped_values"] + 1)))
            path.write_text(json.dumps({"schema_version": "preparation_candidate_report_v1"}), encoding="utf-8")
            self.assertTrue(_matches_candidate(path, impact_totals(impact)))

    def test_missing_after_matches_imputation(self):
        frame = flag_frame().astype(float)
        frame.loc[[3, 7], "x"] = np.nan
        kept = {item["column"]: item for item in self.fit(frame, numeric_imputation="none").impact(frame)}
        filled = {item["column"]: item for item in self.fit(frame).impact(frame)}
        self.assertEqual((kept["x"]["missing"], kept["x"]["imputed"], kept["x"]["missing_after"]), (2, 0, 2))
        self.assertEqual((filled["x"]["imputed"], filled["x"]["missing_after"]), (2, 0))


class CandidateImpactValidationTest(unittest.TestCase):
    """후보 부모 검증기: 영향 기록의 형식·범위·순서·결정적 값이 다르면 후보 전체를 받지 않는다. 이전 v1 후보는 받는다."""

    # 기존 테스트를 다시 실행하지 않도록 상속 대신 준비 함수만 가져온다.
    fixture_commit = candidate_fixtures.PreparationCandidateTest.fixture_commit
    complete = candidate_fixtures.PreparationCandidateTest.complete
    setUp = candidate_fixtures.PreparationCandidateTest.setUp
    setup_run = candidate_fixtures.PreparationCandidateTest.setup_run
    transport = candidate_fixtures.PreparationCandidateTest.transport
    run_candidate = candidate_fixtures.PreparationCandidateTest.run_candidate

    def rewrite(self, change):
        def tampered(request, root, signal):
            outcome = self.transport(request, root, signal)
            folder = Path(request.output_directory) / "candidate"
            report = json.loads((folder / "candidate.json").read_text(encoding="utf-8"))
            change(report)
            (folder / "candidate.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            # 안내 문서도 바꾼 기록에 맞춰 다시 써서, 문서 불일치가 아니라 영향 검사로 거부되는지 본다.
            (folder / "candidate.md").write_bytes(candidate_markdown(report).encode("utf-8"))
            return outcome

        return tampered

    def test_v2_candidate_records_impact_and_parent_accepts_it(self):
        self.setup_run()
        result, output = self.run_candidate()
        self.assertEqual(result.status, "completed")
        report = read_candidate(output / "candidate", self.tool.contract, self.tool.code_id)
        self.assertEqual(report["schema_version"], "preparation_candidate_report_v2")
        self.assertEqual([item["column"] for item in report["impact"]], report["retained"])
        self.assertIn("예상 처리 영향", (output / "candidate/candidate.md").read_text(encoding="utf-8"))

    def tampered_candidate_is_rejected(self, change):
        self.setup_run()
        result, output = self.run_candidate(self.rewrite(change))
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_impact_count_out_of_range_is_rejected(self):
        self.tampered_candidate_is_rejected(lambda report: report["impact"][0].update(missing=10 ** 6))

    def test_impact_order_or_column_change_is_rejected(self):
        self.tampered_candidate_is_rejected(lambda report: report["impact"][0].update(column="other"))

    def test_impact_missing_after_inconsistent_with_imputation_is_rejected(self):
        self.tampered_candidate_is_rejected(lambda report: report["impact"][0].update(missing_after=report["impact"][0]["missing_after"] + 1))

    def test_impact_extra_field_is_rejected(self):
        self.tampered_candidate_is_rejected(lambda report: report["impact"][0].update(rows=[1, 2]))

    def test_impact_wrong_type_is_rejected(self):
        self.tampered_candidate_is_rejected(lambda report: report["impact"][0].update(constant_after="no"))

    def test_v1_downgrade_with_current_code_is_rejected(self):
        """현재 코드의 worker는 항상 v2를 쓴다. 현재 지문으로 묶인 v1은 영향 검증을 피하는 기록으로 보고 거부한다."""
        def downgrade(report):
            report["schema_version"] = "preparation_candidate_report_v1"
            del report["impact"]

        self.tampered_candidate_is_rejected(downgrade)

    def test_v1_candidate_recorded_by_previous_code_is_still_read(self):
        """이전 코드로 실행한 후보(기록된 다른 code_id)는 v1 그대로 읽어 영향을 미산출로 둔다."""
        self.setup_run()
        result, output = self.run_candidate()
        self.assertEqual(result.status, "completed")
        folder = output / "candidate"
        report = json.loads((folder / "candidate.json").read_text(encoding="utf-8"))
        previous = "0" * 64
        report.update(schema_version="preparation_candidate_report_v1",
                      binding=json.loads(json.dumps(self.tool.contract.packet(previous))))
        del report["impact"]
        (folder / "candidate.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        (folder / "candidate.md").write_bytes(candidate_markdown(report).encode("utf-8"))
        self.assertNotIn("impact", read_candidate(folder, self.tool.contract, previous))
        self.assertIsNone(read_candidate(folder, self.tool.contract, self.tool.code_id))


class ApplyImpactValidationTest(unittest.TestCase):
    """적용 부모 검증기: 출력 파일로 정해지는 실측 합계(상수 특징·새 범주·결측 칸)를 대조한다."""

    fixture_commit = apply_fixtures.PreparationApplyTest.fixture_commit
    complete = apply_fixtures.PreparationApplyTest.complete
    setUp = apply_fixtures.PreparationApplyTest.setUp
    setup_apply = apply_fixtures.PreparationApplyTest.setup_apply
    transport = apply_fixtures.PreparationApplyTest.transport
    run_apply = apply_fixtures.PreparationApplyTest.run_apply

    def test_applied_impact_is_recorded(self):
        self.setup_apply()
        result, output = self.run_apply()
        self.assertEqual(result.status, "completed")
        report = json.loads((output / "prepared/application.json").read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], "preparation_application_v2")
        self.assertEqual(set(report["impact"]), {"train", "validation"})
        # 합성 text는 행마다 달라 Validation 값은 모두 Train에 없던 범주다(unknown ignore).
        self.assertGreater(report["impact"]["validation"]["unknown_categories"], 0)

    def tampered_application_is_rejected(self, key):
        self.setup_apply()

        def tampered(request, root, signal):
            outcome = self.transport(request, root, signal)
            path = Path(request.output_directory) / "prepared/application.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["impact"]["validation"][key] += 1
            path.write_text(json.dumps(value), encoding="utf-8")
            return outcome

        result, output = self.run_apply(tampered)
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_tampered_constant_feature_count_is_rejected(self):
        self.tampered_application_is_rejected("constant_features")

    def test_tampered_unknown_category_count_is_rejected(self):
        self.tampered_application_is_rejected("unknown_categories")

    def test_tampered_missing_value_count_is_rejected(self):
        self.tampered_application_is_rejected("missing_values")


class PlanPreviewAndImpactFlowTest(unittest.TestCase):
    """옵션 설명 → 최종 계획 미리보기(답변 반영) → 확정 일치 → 예상 영향 → 적용 실측 → 보고서."""

    ROLES = {"x": "수치형", "flag": "수치형", "label": "target", "id": "식별자", "time": "제외"}
    setUp_csv = flow.TaskOrchestratorTest.setUp_csv
    start_patch = flow.TaskOrchestratorTest.start_patch
    transport = flow.TaskOrchestratorTest.transport
    new_task = flow.TaskOrchestratorTest.new_task

    def setUp(self):
        flow.TaskOrchestratorTest.setUp(self)
        frame = pd.read_csv(self.source)
        frame["flag"] = [1 if index % 10 == 0 else 0 for index in range(len(frame))]
        self.source = self.source.with_name("flag_input.csv")
        frame.to_csv(self.source, index=False)

    def recommending(self, changes):
        def requester(payload, config):
            self.requests.append(payload)
            return flow.model_reply(self.ROLES, changes=changes)

        return requester

    def reviewed(self, changes=()):
        task = self.new_task()
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.recommending(list(changes)))
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        return task

    def test_binary_clip_is_warned_before_approval_and_binning_turns_off_without_question(self):
        task = self.reviewed([{"path": "preparation.outlier", "value": "iqr_clip", "reason": "극단값 영향을 줄입니다"},
                              {"path": "preparation.binning", "value": "quantile", "reason": "분포를 단순화합니다"}])
        state = orchestrator.review_state(task)
        # 질문은 없어도 모든 전처리 옵션에 설명이 있고, 모델 의견과 계산 사실이 나뉜다.
        self.assertFalse([item for item in state["questions"] if item["path"] == "preparation.binning"])
        guide = {item["name"]: item for item in state["guide"]}
        self.assertEqual(guide["binning"]["opinion"], "분포를 단순화합니다")
        self.assertEqual(guide["binning"]["source"], "모델 권장")
        self.assertTrue(all(item["what"] and item["off"] and item["caution"] for item in state["guide"]))
        self.assertTrue(any("flag" in text and "상수" in text for text in guide["outlier"]["warnings"]))

        # 질문 없이 양식에서 구간화를 끈다. 미리 본 최종 계획이 그대로 확정·실행된다.
        plan = json.loads(json.dumps(state["plan"]))
        plan["preparation"]["binning"] = "none"
        preview = orchestrator.preview_plan(task, plan, {})
        self.assertEqual(preview["origins"]["preparation.binning"]["source"], "user")
        self.assertEqual(preview["origins"]["preparation.outlier"], {"source": "model", "reason": "극단값 영향을 줄입니다"})
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["phase"], "awaiting_review")

        with self.assertRaises(PlanError) as caught:
            orchestrator.confirm_plan(task["task_id"], plan, {}, root=self.store, expected_digest="0" * 64)

        self.assertEqual(caught.exception.code, "PLAN_PREVIEW_CHANGED")
        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["steps"], {})
        task = orchestrator.confirm_plan(task["task_id"], plan, {}, root=self.store, expected_digest=preview["digest"])
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(task["consents"]["C"]["content"]["plan_digest"], preview["plan_digest"])
        self.assertEqual(task["confirmed"]["plan"], preview["plan"])
        rules = next(table for table in task["candidate"]["facts"]["tables"] if table["title"].startswith("실제 실행 설정"))
        executed = {row["설정 이름"]: row["값"] for row in rules["rows"]}
        self.assertEqual((executed["binning"], executed["outlier"]), ("사용 안 함", "IQR 경계로 자르기"))

        # 결정 D 전: 0/1 특징의 상수화를 Train 후보 계산으로 확인해 경고하고 설정은 바꾸지 않는다.
        facts = task["candidate"]["facts"]
        metrics = {item["label"]: item["value"] for item in facts["metrics"]}
        self.assertGreaterEqual(metrics["정보 손실 특징 (예상)"], 1)
        self.assertTrue(any("flag" in text and "정보 손실" in text for text in facts["warnings"]))
        self.assertTrue(any("예상치" in text for text in facts["notes"]))
        impact = next(table for table in facts["tables"] if table["title"].startswith("예상 처리 영향"))
        flag = next(row for row in impact["rows"] if row["특징"] == "flag")
        self.assertEqual((flag["변환 후 상수"], flag["정보 손실"]), ("예", "예"))
        self.assertEqual(task["confirmed"]["plan"]["preparation"]["outlier"], "iqr_clip")

        # 적용 후 보고서: 예상과 실측을 나눠 보이고, 제외 사유·경고·활용 방법을 담는다.
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        facts = task["report"]["facts"]
        constant = {row["대상 구간"]: row for row in facts["comparison"] if row["항목"] == "변환 후 상수 특징"}
        self.assertEqual(set(constant), {"train", "validation"})
        self.assertIn("후보 예상", constant["train"]["근거"])
        self.assertNotIn("후보 예상", constant["validation"]["근거"])
        output = [row["처리 후"] for row in facts["comparison"] if row["항목"] == "출력 결측"]
        self.assertTrue(output and all(type(value) is int for value in output))
        processing = {row["컬럼"]: row for row in facts["processing"]}
        self.assertEqual(processing["time"]["제외 사유"], "모델 권장으로 제외(모델 의견: 의미에 맞는 역할)")
        self.assertIn("보존", processing["label"]["제외 사유"])
        self.assertIn("정보 손실", processing["flag"]["처리"])
        kinds = {item["구분"] for item in facts["warnings"]}
        self.assertEqual(kinds, {"정보 손실", "미처리", "미검증"})
        text = task["report"]["text"]

        for title in ("## 핵심 결과와 주요 문제", "## 컬럼별 처리와 제외 사유", "## 실제 전후 비교", "## 경고: 정보 손실·미처리·미검증",
                      "## 적용 구간과 결과 파일 활용", "<details>"):
            self.assertIn(title, text)

        self.assertTrue(any(row["파일"] == "prepared/train_features.csv" and row["용도"].startswith("학습용") for row in facts["files"]))

    def test_form_and_answer_conflict_is_not_resolved_automatically(self):
        roles = dict(self.ROLES, label="범주형")
        task = self.new_task()
        task = orchestrator.diagnose(task["task_id"], root=self.store,
                                     requester=lambda payload, config: flow.model_reply(roles))
        state = orchestrator.review_state(task)
        question = next(item for item in state["questions"] if item["id"] == "target")
        plan = json.loads(json.dumps(state["plan"]))
        plan["roles"]["label"] = "제외"

        with self.assertRaises(PlanError) as caught:
            orchestrator.preview_plan(task, plan, {"target": "label"})

        self.assertEqual((caught.exception.code, caught.exception.subjects), ("ANSWER_FORM_CONFLICT", ("roles.label",)))

        with self.assertRaises(PlanError):
            orchestrator.confirm_plan(task["task_id"], plan, {"target": "label"}, root=self.store)

        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["steps"], {})
        # 사용자가 답변 쪽을 고르면(양식 값을 답변 값으로) 충돌 없이 미리볼 수 있다. 답변이 반영된 최종 계획이다.
        self.assertIn("label", [option["value"] for option in question["options"]])
        plan["roles"]["label"] = "target"
        preview = orchestrator.preview_plan(task, plan, {"target": "label"})
        self.assertEqual((preview["plan"]["roles"]["label"], preview["origins"]["roles.label"]["source"]), ("target", "answer"))

    def test_plan_can_be_revised_from_apply_review_as_new_attempt(self):
        task = self.reviewed([{"path": "preparation.outlier", "value": "iqr_clip", "reason": "극단값 영향을 줄입니다"}])
        state = orchestrator.review_state(task)
        task = orchestrator.confirm_plan(task["task_id"], state["plan"], {}, root=self.store)
        # 간헐 실패(여러 모듈 동시 실행에서만 1회 발생, 단독·모듈 실행 재현 안 됨) 원인 확인용으로 apply_revisable의 판정 근거를 모두 남긴다.
        self.assertTrue(orchestrator.apply_revisable(task),
                        {"phase": task["phase"], "problem": task.get("problem"), "steps": task.get("steps"),
                         "candidate": task.get("candidate") is not None, "consents": sorted(task.get("consents", {})),
                         "step_states": {key: (item.get("decision"), item.get("status")) for key, item in task.get("steps", {}).items()}})
        previous = task["consents"]["C"]["consent_id"]
        state = orchestrator.review_state(task)
        self.assertEqual(state["plan"]["preparation"]["outlier"], "iqr_clip")
        plan = json.loads(json.dumps(state["plan"]))
        plan["preparation"]["outlier"] = "none"
        preview = orchestrator.preview_plan(task, plan, {})
        task = orchestrator.confirm_plan(task["task_id"], plan, {}, root=self.store, expected_digest=preview["digest"])
        self.assertEqual((task["phase"], task["attempt"]), ("awaiting_apply", 2), task.get("problem"))
        self.assertNotEqual(task["consents"]["C"]["consent_id"], previous)
        self.assertEqual(task["attempts"][0]["consent_c"]["consent_id"], previous)
        self.assertEqual({item["label"]: item["value"] for item in task["candidate"]["facts"]["metrics"]}["clipping으로 바뀌는 값 (예상)"], 0)
        # 같은 확정 요청을 다시 보내도(다른 탭·재전송) 새 시도를 만들지 않는다. 미리보기 없이 적용 단계 계획 수정도 거부한다.
        for digest in (preview["digest"], None):
            with self.assertRaises(PlanError) as caught:
                orchestrator.confirm_plan(task["task_id"], plan, {}, root=self.store, expected_digest=digest)

            self.assertEqual(caught.exception.code, "PLAN_PREVIEW_CHANGED")

        stored = task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["attempt"], stored["consents"]["C"]["consent_id"]), (2, task["consents"]["C"]["consent_id"]))
        # 다른 탭의 오래된 적용 화면(이전 후보 서명·시도)으로는 적용하지 않는다.
        old = task["attempts"][0]

        for values in ({"expected_attempt": old["attempt"]}, {"expected_signature": "0" * 64}):
            with self.assertRaises(ValueError):
                orchestrator.approve_apply(task["task_id"], root=self.store, **values)

        stored = task_store.load_task(task["task_id"], root=self.store)
        self.assertEqual((stored["phase"], "D" in stored["consents"]), ("awaiting_apply", False))
        self.assertEqual(self.calls.count("apply_preparation"), 0)
        # 적용 동의 이후에는 계획 수정 경로가 없다.
        self.assertFalse(orchestrator.apply_revisable(dict(task, consents=dict(task["consents"], D={}))))


if __name__ == "__main__":
    unittest.main()

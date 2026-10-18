"""데이터·모델·worker 없이 LLM 제안의 허용 범위와 기존 계획 검증을 확인한다."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from src.agents.llm_proposal_contract import validate_proposal


ROOT = Path(__file__).resolve().parents[2]
SOURCE = {"contract_id": "tabular_data", "contract_version": 1, "partition": "unsplit"}
ALLOWED = (("tabular_quality", "1.0.0"), ("eda_missing", "1.0.0"),
           ("eda_correlations", "1.0.0"))


class LlmProposalContractTest(unittest.TestCase):
    """모델 응답은 검증 입력일 뿐 승인이나 실행 명령이 아님을 확인한다."""

    def setUp(self):
        pipeline = json.loads((ROOT / "configs/agents/pipelines/tabular_eda_example.json").read_text(encoding="utf-8"))
        self.proposal = {
            "schema_version": "llm_proposal_v1", "status": "proposed",
            "pipeline": pipeline, "questions": [],
        }

    def validate(self, value, *, allowed=ALLOWED):
        return validate_proposal(
            json.dumps(value, ensure_ascii=False),
            expected_input=SOURCE,
            allowed_modules=allowed,
        )

    def test_valid_proposal_has_no_authority(self):
        """유효한 실행안도 worker 실행·승인 상태로 바뀌지 않는다."""
        result = self.validate(self.proposal)
        view = result.to_dict()
        self.assertEqual(len(view["plan_signature"]), 64)
        self.assertTrue(view["proposal_only"])
        self.assertFalse(view["execution_approved"])
        self.assertFalse(view["worker_started"])
        view["proposal"]["status"] = "completed"
        self.assertEqual(result.to_dict()["proposal"]["status"], "proposed")

    def test_clarification_has_no_pipeline(self):
        """모호한 요청은 실행안 대신 질문으로 반환할 수 있다."""
        self.proposal.update(status="needs_clarification", pipeline=None, questions=["제거 기준을 지정해 주세요."])
        self.assertIsNone(self.validate(self.proposal).plan_signature)

    def test_invalid_question_combinations(self):
        """질문 누락·과다·잘못된 타입·실행안과 질문 혼합을 거부한다."""
        for questions in ([], [""], [1], ["가" * 501], ["질문"] * 6):
            with self.subTest(questions=questions):
                candidate = dict(self.proposal, status="needs_clarification", pipeline=None, questions=questions)
                with self.assertRaises(ValueError):
                    self.validate(candidate)

        self.proposal["questions"] = ["질문"]
        with self.assertRaises(ValueError):
            self.validate(self.proposal)

    def test_unknown_fields_and_authority_rejected(self):
        """모델이 승인·경로·명령을 덧붙여도 허용하지 않는다."""
        for field in ("approved", "command", "source_path", "limits"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(dict(self.proposal, **{field: True}))

    def test_module_and_settings_rejected(self):
        """가짜 모듈·다른 버전·등록 외 설정·미허용 모듈을 거부한다."""
        for field, value in (("module_id", "fake_module"), ("version", "99.0.0"),
                             ("settings", {"shell": "anything"})):
            candidate = deepcopy(self.proposal)
            candidate["pipeline"]["steps"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(candidate)

        with self.assertRaises(ValueError):
            self.validate(self.proposal, allowed=ALLOWED[:1])

    def test_source_change_rejected(self):
        """모델이 Train·Test 또는 입력 종류를 임의로 선택하지 못한다."""
        for field, value in (("partition", "test"), ("contract_id", "wm_canonical")):
            candidate = deepcopy(self.proposal)
            candidate["pipeline"]["input"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(candidate)

    def test_order_and_quality_gate_rejected(self):
        """미래 단계 의존성과 품질 검사 우회를 모두 거부한다."""
        candidate = deepcopy(self.proposal)
        candidate["pipeline"]["steps"][0]["depends_on"] = ["missing"]
        with self.assertRaises(ValueError):
            self.validate(candidate)

        candidate = deepcopy(self.proposal)
        candidate["pipeline"]["steps"][1]["depends_on"] = []
        with self.assertRaises(ValueError):
            self.validate(candidate)

    def test_strict_json_and_size(self):
        """중복 키·NaN·설명문·과대 응답·깊은 중첩을 거부한다."""
        for response in ('{"status":1,"status":2}', '{"value":NaN}',
                         '```json\n{}\n```', "가" * 30000, "[" * 2000 + "]" * 2000):
            with self.subTest(response=response[:30]), self.assertRaises(ValueError):
                validate_proposal(response, expected_input=SOURCE, allowed_modules=ALLOWED)

    def test_wm_plan_reuses_existing_validator(self):
        """표형 전용이 아니라 기존 WM 증거 계획도 정적으로 검증한다."""
        pipeline = json.loads((ROOT / "configs/agents/pipelines/wm_development_example.json").read_text(encoding="utf-8"))
        allowed = tuple((step["module_id"], step["version"]) for step in pipeline["steps"])
        candidate = dict(self.proposal, pipeline=pipeline)
        result = validate_proposal(
            json.dumps(candidate), expected_input=pipeline["input"], allowed_modules=allowed,
        )
        self.assertIsNotNone(result.plan_signature)


if __name__ == "__main__":
    unittest.main()

"""실험적 Graph RAG 2단계: 작업 설정·동의 A·권장 요청·기록·보고서·화면 연결을 합성 CSV·합성 지식·모의 임베더/생성기로 검사한다.

OS transport·로컬 모델·임베딩만 모의한다. 진단·계획 검증·승인·결과 검증은 실제 코드다. 실제 지식집·색인·모델을 쓰지 않는다.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.agents import (array_task_transport, module_pipeline_transport, task_adapters, task_consent,
                        qwen_tokens, task_orchestrator as orchestrator, task_rag, task_recommend, task_report, task_store)
from src.agents.llm_config import LocalModelConfig
from src.agents.local_ollama import LocalProposalError
from src.agents.rag_embed import EmbedError
from tests.agents import rag_fixtures as fx
from tests.agents.test_qwen_tokens import FakeTokenizer
from tests.agents import test_task_orchestrator as flow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_K01 = fx.unit(1.0, 0.1)
RAG_MODULES = ("task_rag", "rag_embed", "rag_knowledge", "rag_index", "rag_tabular")


def content(payload):
    return json.loads(payload["messages"][1]["content"])


class RagFlowTest(unittest.TestCase):
    setUp_flow = flow.TaskOrchestratorTest.setUp
    setUp_csv = flow.TaskOrchestratorTest.setUp_csv
    start_patch = flow.TaskOrchestratorTest.start_patch
    transport = flow.TaskOrchestratorTest.transport
    requester = flow.TaskOrchestratorTest.requester

    def setUp(self):
        self.setUp_flow()
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.rag_root = Path(temp.name)
        self.embedders = []
        # 기본은 실제 서버 상태 조회 없이 로컬 토큰 계산을 쓸 수 없는 상태(기존 바이트 예산). 토큰 경로 시험은 따로 바꾼다.
        self.prepare = self.start_patch("src.agents.qwen_tokens.prepare",
                                        side_effect=qwen_tokens.TokenCountUnavailable("RAG_TOKENS_STATUS_FAILED"))

    def knowledge(self, values=None, query=QUERY_K01, error=None):
        """합성 지식집·색인을 임시 폴더에 만들고 표형 프로필 경로와 임베더 생성을 그쪽으로 돌린다."""
        profile, _ = fx.prepared(self.rag_root, values)

        for name in ("REPO_ROOT", "PACK_DIR", "INDEX_ROOT"):
            self.start_patch("src.agents.rag_tabular." + name, new=getattr(profile, name))

        def make(config):
            self.assertIsInstance(config, LocalModelConfig)
            embedder = fx.FakeEmbedder(query=query, error=error)
            self.embedders.append(embedder)
            return embedder

        self.start_patch("src.agents.task_rag.OllamaEmbedder", side_effect=make)

    def embeds(self):
        return sum(len(embedder.texts) for embedder in self.embedders)

    def new_task(self, rag_mode=None):
        options = {} if rag_mode is None else {"rag_mode": rag_mode}
        return orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", self.source, flow.LIMITS, self.model,
                                       root=self.store, **options)

    def baseline(self, task):
        """같은 작업의 근거 없는 요청(OFF 경로와 같은 계산)."""
        adapter = task_adapters.adapter(task["kind"])
        chosen = task_recommend.plan_request(task["goal"], dict(adapter.recommendation_input(task)), adapter.base_plan(task),
                                             adapter.plan_fields(task), LocalModelConfig(**task["model"]))
        return chosen[0]

    # ==========================================
    # OFF·이전 기록
    # ==========================================

    def test_off_is_default_and_keeps_request_without_rag_calls(self):
        self.knowledge()
        task = self.new_task()
        self.assertEqual(task["rag"], {"mode": "off"})
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["consents"]["A"]["content"]["rag"], {"mode": "off"})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0], self.baseline(task))
        self.assertNotIn("evidence", content(self.requests[0]))
        self.assertEqual((self.embedders, task["draft"]["rag"]["status"]), ([], "off"))
        self.assertEqual(task_rag.view(task["draft"])["lines"], ["검색 근거(실험적 Graph RAG): 선택 OFF · 실제 결과: 사용 안 함(OFF)"])

    def test_record_without_setting_is_read_as_off(self):
        self.knowledge()
        task = self.new_task()
        task.pop("rag")
        task = task_store.save_task(task, expected_revision=task["revision"], root=self.store)
        self.assertNotIn("rag", task_store.load_task(task["task_id"], root=self.store))
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["consents"]["A"]["content"]["rag"], {"mode": "off"})
        self.assertEqual((len(self.requests), self.embeds(), task["draft"]["rag"]["status"]), (1, 0, "off"))
        # 사용 기록이 없는 이전 초안은 검색 근거 절을 만들지 않는다.
        self.assertIsNone(task_rag.view({"source": "model"}))

    # ==========================================
    # ON 정상 경로
    # ==========================================

    def test_on_sends_one_embed_and_one_generation_and_keeps_diagnostic_scope(self):
        self.knowledge()
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        consent = task["consents"]["A"]["content"]["rag"]
        self.assertEqual((consent["mode"], consent["embedding_model"], consent["query_scope"]),
                         ("on", "bge-m3:567m", task_rag.QUERY_SCOPE))
        self.assertEqual((len(self.requests), self.embeds()), (1, 1))
        sent, base = content(self.requests[0]), content(self.baseline(task))
        # 진단 전달 범위(상세·이름만·요약)와 나머지 본문은 근거 없는 요청과 같다.
        self.assertEqual({key: value for key, value in sent.items() if key != "evidence"}, base)
        self.assertTrue(self.requests[0]["messages"][0]["content"].endswith(task_rag.EVIDENCE_INSTRUCTIONS))
        self.assertEqual(self.requests[0]["format"], self.baseline(task)["format"])
        self.assertTrue(task_recommend._fits(self.requests[0], LocalModelConfig(**task["model"])))
        # 질의는 목표와 고정 단서만 담고 목표에 없는 컬럼명·값을 담지 않는다.
        query = self.embedders[0].texts[0]
        self.assertTrue(query.startswith("목표: label 예측용으로 분할·전처리해 주세요\n데이터 특성: "))
        for name in ("time", "120"):
            self.assertNotIn(name, query)
        usage = task["draft"]["rag"]
        self.assertEqual(usage["status"], "graph")
        self.assertEqual(usage["included_items"], [item["id"] for item in sent["evidence"]["it"]])
        self.assertEqual(usage["included_relations"], [relation["id"] for relation in sent["evidence"]["rel"]])
        self.assertIn("K01", usage["included_items"])
        self.assertEqual(usage["retrieved"][0]["id"], "K01")
        self.assertEqual([row["ID"] for row in task_rag.view(task["draft"])["items"]], usage["included_items"])
        self.assertEqual([row["ID"] for row in task_rag.view(task["draft"])["relations"]], usage["included_relations"])
        # 계획 검증·질문 필터는 그대로다(모델 권장안은 기존 병합 검증을 통과한 것만).
        self.assertEqual(task["draft"]["source"], "model")
        self.assertEqual(task["draft"]["plan"]["roles"], flow.ROLES)
        stored = json.dumps(task["draft"]["rag"], ensure_ascii=False)
        for forbidden in (query, "데이터 특성", str(self.source)):
            self.assertNotIn(forbidden, stored)

    def test_on_evidence_appears_in_completed_report_without_new_search(self):
        self.knowledge()
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        task = orchestrator.confirm_plan(task["task_id"], flow.plan(), root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertEqual(self.embeds(), 1)
        text = task["report"]["text"].split("## 기술 상세")[0]
        self.assertIn("## 검색 근거 (실험적 Graph RAG)", text)
        self.assertIn("실제 결과: 근거와 그래프 관계를 전달함", text)
        self.assertIn(task_rag.EXPERIMENTAL_NOTICE, text)
        for iid in task["draft"]["rag"]["included_items"] + task["draft"]["rag"]["included_relations"]:
            self.assertIn("| " + iid + " |", text)
        # 보고서 Markdown은 입력 유래 문자열의 특수문자를 escape한다(_ → \_).
        self.assertIn(r"S\_RULE 합성 규칙 — 합성 절", text)
        self.assertIn("작성자 확인 · 사람 검토 대기", text)
        self.assertEqual(task["report"]["facts"]["technical"]["draft"]["rag"]["status"], "graph")

    def test_only_vector_evidence_is_not_reported_as_graph_success(self):
        values = fx.pack_values()
        values["graph"]["edges"] = [edge for edge in values["graph"]["edges"] if "K01" not in edge["knowledge"]]
        self.knowledge(values)
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        usage = task["draft"]["rag"]
        self.assertEqual((usage["status"], usage["included_relations"]), ("vector_only", []))
        self.assertEqual(content(self.requests[0])["evidence"]["rel"], [])
        lines = "\n".join(task_rag.view(task["draft"])["lines"])
        self.assertIn("벡터 검색 근거만 전달함", lines)
        self.assertNotIn("그래프 관계를 전달함", lines)

    # ==========================================
    # 실패 정책: 생성 요청 전에 결정, 근거 없는 기존 요청 1회
    # ==========================================

    def assert_baseline_once(self, task, status, code, embeds):
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0], self.baseline(task))
        self.assertEqual((task["draft"]["rag"]["status"], task["draft"]["rag"]["code"], self.embeds()), (status, code, embeds))
        self.assertEqual(task["draft"]["rag"]["included_items"], [])

    def test_no_relevant_evidence(self):
        self.knowledge(query=fx.unit(0, 0, 0, 0, 1))
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assert_baseline_once(task, "no_evidence", "RAG_NO_RELEVANT", 1)

    def test_embedding_failure(self):
        self.knowledge(error=EmbedError("RAG_EMBED_CONNECTION_FAILED"))
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assert_baseline_once(task, "failed", "RAG_EMBED_CONNECTION_FAILED", 1)

    def test_invalid_query_vector(self):
        self.knowledge(query=[float("nan")] * fx.DIMENSIONS)
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assert_baseline_once(task, "failed", "RAG_EMBED_RESPONSE_INVALID", 1)

    def test_missing_knowledge_and_index_mismatch_send_no_embedding(self):
        self.start_patch("src.agents.rag_tabular.PACK_DIR", new=self.rag_root / "none")
        self.start_patch("src.agents.task_rag.OllamaEmbedder", side_effect=AssertionError("임베딩을 만들면 안 됨"))
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assert_baseline_once(task, "failed", "RAG_KNOWLEDGE_UNAVAILABLE", 0)

    def test_changed_index_is_not_reused(self):
        self.knowledge()
        index = next(path for path in (self.rag_root / "models/rag/tabular").iterdir())
        manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
        manifest["model"]["digest"] = "b" * 64
        (index / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assert_baseline_once(task, "failed", "RAG_INDEX_MODEL_MISMATCH", 0)

    def test_changed_product_source_rejects_knowledge(self):
        self.knowledge()
        (self.rag_root / fx.SOURCE_PATH).write_bytes(b"# changed\n")
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assert_baseline_once(task, "failed", "RAG_SOURCE_CHANGED", 0)

    def test_budget_shortfall_keeps_diagnostic_scope_and_sends_baseline(self):
        self.knowledge()
        self.start_patch("src.agents.task_rag.EVIDENCE_INSTRUCTIONS", new=" " * 70000)
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assert_baseline_once(task, "budget_omitted", "RAG_BUDGET_EXHAUSTED", 1)
        self.assertGreater(task["draft"]["rag"]["omitted_items"], 0)
        self.assertIn("예산 부족으로 근거 미사용", task_rag.view(task["draft"])["lines"][0])

    def test_generation_failure_uses_default_plan_without_retry(self):
        self.knowledge()

        def failing(payload, config):
            self.requests.append(payload)
            raise LocalProposalError("OLLAMA_CONNECTION_FAILED")

        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=failing)
        self.assertEqual((len(self.requests), self.embeds()), (1, 1))
        self.assertIn("evidence", content(self.requests[0]))
        self.assertEqual((task["draft"]["source"], task["draft"]["code"]), ("default", "OLLAMA_CONNECTION_FAILED"))
        self.assertEqual(task["draft"]["rag"]["status"], "graph")

    def test_contract_rejection_keeps_whole_rejection_and_no_second_request(self):
        self.knowledge()
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store,
                                     requester=lambda payload, config: self.requests.append(payload) or "응답 아님")
        self.assertEqual((len(self.requests), task["draft"]["source"]), (1, "default"))
        self.assertEqual(task["draft"]["code"], "RECOMMENDATION_RESPONSE_INVALID")

    def test_discarded_evidence_is_not_recorded_as_sent_even_on_unexpected_error(self):
        self.knowledge()

        def oversized(payload, config, items_by_id, hits, expansion, fits):
            # 예산 검사를 통과하지 못하는 근거 요청을 돌려줘 권장 요청 쪽의 최종 검사가 버리게 한다.
            final = task_rag.with_evidence(payload, [{"id": "K01", "s": "x" * 70000}], [], [0, 0])
            return final, [{"id": "K01", "o": "v"}], [], [0, 0]

        self.start_patch("src.agents.task_rag.fit", side_effect=oversized)

        def broken(payload, config):
            self.requests.append(payload)
            raise RuntimeError("unexpected")

        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=broken)
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("evidence", content(self.requests[0]))
        self.assertEqual((task["draft"]["code"], task["draft"]["rag"]["status"]), ("MODEL_REQUEST_FAILED", "failed"))
        self.assertEqual(task["draft"]["rag"]["included_items"], [])

    def test_unexpected_rag_error_falls_back_to_baseline(self):
        self.knowledge()
        self.start_patch("src.agents.task_rag.expand", side_effect=RuntimeError("secret detail C:\\x"))
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assert_baseline_once(task, "failed", "RAG_INTERNAL_ERROR", 1)
        self.assertNotIn("secret", json.dumps(task["draft"]["rag"], ensure_ascii=False))

    # ==========================================
    # 동의·설정 변조·지원 유형
    # ==========================================

    def test_changed_setting_after_consent_refuses_embedding(self):
        self.knowledge()
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        self.requests.clear()
        adapter = task_adapters.adapter(task["kind"])

        for change in ("setting", "consent_missing", "consent_scope"):
            with self.subTest(change):
                record = task_store.load_task(task["task_id"], root=self.store)
                record["rag"] = {"mode": "on"}

                if change == "consent_missing":
                    del record["consents"]["A"]
                elif change == "consent_scope":
                    consent = record["consents"]["A"]
                    consent["content"]["rag"] = task_rag.consent_content(record, adapter)
                    # 내용만 바꾸고 해시를 다시 계산하지 않은 동의는 유효하지 않다.

                augment = task_rag.augmenter(record, adapter)
                payload = self.baseline(record)
                # 근거 없는 생성 요청도 보내지 않도록 요청 자체를 멈추는 신호를 낸다.
                with self.assertRaises(task_recommend.RequestWithheld) as caught:
                    augment(payload, LocalModelConfig(**record["model"]))
                usage = caught.exception.rag
                self.assertEqual((usage["status"], usage["code"], self.embeds()), ("not_attempted", "RAG_CONSENT_INVALID", 0))
                self.assertEqual(task_rag.blocked(record, adapter)["code"], "RAG_CONSENT_INVALID")

    # ==========================================
    # 실행 코드 지문(동의 A의 rag 범위)
    # ==========================================

    def code_root(self):
        """지문 대상 코드의 사본을 임시 폴더에 두고 지문 계산 기준을 그쪽으로 돌린다. 제품 소스는 바꾸지 않는다."""
        root = self.rag_root / "code"

        for name in task_rag.CODE_FILES:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes((PROJECT_ROOT / name).read_bytes())

        self.start_patch("src.agents.task_rag.CODE_ROOT", new=root)
        return root

    def test_code_id_follows_each_rag_execution_file(self):
        real = task_rag.code_id()
        root = self.code_root()
        self.assertEqual(task_rag.code_id(), real)
        self.assertEqual(task_rag.code_id(), task_rag.code_id())
        # 관리 전용 색인 구축 명령과 지식·색인 데이터는 실행 코드 지문 대상이 아니다.
        self.assertNotIn("src/agents/rag_index_cli.py", task_rag.CODE_FILES)
        self.assertFalse(any(name.startswith(("configs/", "models/")) for name in task_rag.CODE_FILES))

        for module in RAG_MODULES:
            self.assertIn(f"src/agents/{module}.py", task_rag.CODE_FILES)

        for name in task_rag.CODE_FILES:
            with self.subTest(name):
                original = (root / name).read_bytes()
                (root / name).write_bytes(original + b"\n# changed\n")
                self.assertNotEqual(task_rag.code_id(), real)
                (root / name).write_bytes(original)
                self.assertEqual(task_rag.code_id(), real)

    def test_on_consent_binds_code_id(self):
        task = self.new_task("on")
        self.assertEqual(orchestrator.consent_a_scope(task)["rag"]["code_id"], task_rag.code_id())
        self.assertNotIn("code_id", orchestrator.consent_a_scope(self.new_task())["rag"])

    def change_code_after_consent(self, name):
        """실제 동의 발급 직후 지문 대상 사본을 바꾼다(동의 후 실행 전 코드 변경)."""
        root, real = self.code_root(), task_consent.grant

        def grant(*args, **kwargs):
            consent = real(*args, **kwargs)
            (root / name).write_bytes((root / name).read_bytes() + b"\n# changed after consent\n")
            return consent

        self.start_patch("src.agents.task_consent.grant", side_effect=grant)

    def assert_blocked_after_code_change(self, name):
        self.knowledge()
        self.change_code_after_consent(name)
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assertNotEqual(task["consents"]["A"]["content"]["rag"]["code_id"], task_rag.code_id())
        self.assertEqual((len(self.requests), self.embeds()), (0, 0))
        draft = task["draft"]
        self.assertEqual((draft["source"], draft["code"]), ("default", "RAG_CONSENT_INVALID"))
        self.assertEqual(draft["plan"], task_adapters.adapter(task["kind"]).base_plan(task))
        self.assertEqual((draft["rag"]["status"], draft["rag"]["code"], draft["rag"]["embed_attempts"]),
                         ("not_attempted", "RAG_CONSENT_INVALID", 0))
        self.assertIn(task_report.code_label(task, "RAG_CONSENT_INVALID"), task_report.recommendation_notice(task)[0])
        # 근거를 붙이는 함수를 직접 불러도 같은 대조로 거부한다.
        adapter = task_adapters.adapter(task["kind"])
        with self.assertRaises(task_recommend.RequestWithheld) as caught:
            task_rag.augmenter(task, adapter)(self.baseline(task), LocalModelConfig(**task["model"]))
        self.assertEqual((caught.exception.rag["code"], self.embeds()), ("RAG_CONSENT_INVALID", 0))

    def test_embed_code_change_after_consent_sends_no_embedding_or_generation(self):
        self.assert_blocked_after_code_change("src/agents/rag_embed.py")

    def test_rag_code_change_after_consent_sends_no_embedding_or_generation(self):
        self.assert_blocked_after_code_change("src/agents/task_rag.py")

    def test_consent_check_after_precheck_withholds_generation(self):
        """권장 요청 전 대조를 통과한 뒤(임베딩 직전) 코드가 바뀌어도 임베딩·생성 요청을 보내지 않는다."""
        self.knowledge()
        root = self.code_root()
        real = task_rag.blocked

        def blocked_then_change(record, adapter):
            result = real(record, adapter)
            (root / "src/agents/rag_index.py").write_bytes((root / "src/agents/rag_index.py").read_bytes() + b"\n# changed\n")
            return result

        self.start_patch("src.agents.task_rag.blocked", side_effect=blocked_then_change)
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assertEqual((len(self.requests), self.embeds()), (0, 0))
        self.assertEqual((task["draft"]["source"], task["draft"]["code"], task["draft"]["rag"]["status"]),
                         ("default", "RAG_CONSENT_INVALID", "not_attempted"))

    def test_resolve_problem_keeps_original_rag_consent_after_code_change(self):
        """해결 동작으로 동의 A를 다시 발급해도 처음 동의한 rag 범위(코드 지문)를 이어 코드 변경 뒤 요청하지 않는다."""
        import pandas as pd

        self.knowledge()
        root = self.code_root()
        frame = pd.read_csv(self.source)
        frame.insert(1, "zero", frame["x"] * 2)
        frame.insert(2, "zero_", frame["x"] * 3)
        source = Path(self.store).parent / "duplicate.csv"
        frame.to_csv(source, index=False, header=["x", "zero", "zero", "label", "id", "time"])
        task = orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", source, flow.LIMITS, self.model, root=self.store,
                                       rag_mode="on")
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "needs_attention", task.get("problem"))
        first = task["consents"]["A"]["content"]["rag"]
        self.assertEqual(first["code_id"], task_rag.code_id())
        self.assertEqual(orchestrator.consent_a_scope(task, "header_dedupe")["rag"], first)

        (root / "src/agents/rag_embed.py").write_bytes((root / "src/agents/rag_embed.py").read_bytes() + b"\n# changed\n")
        task = orchestrator.resolve_problem(task["task_id"], "header_dedupe", root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        self.assertEqual(task["consents"]["A"]["content"]["rag"], first)
        self.assertEqual((len(self.requests), self.embeds()), (0, 0))
        self.assertEqual((task["draft"]["code"], task["draft"]["rag"]["code"]), ("RAG_CONSENT_INVALID", "RAG_CONSENT_INVALID"))

    def test_renewed_consent_without_previous_rag_is_off(self):
        task = self.new_task()
        self.assertEqual(task_rag.renewed(task, task_adapters.adapter(task["kind"])), {"mode": "off"})
        previous = dict(task, consents={"A": {"content": {"rag": {"mode": "on", "code_id": "0" * 64}}}}, rag={"mode": "on"})
        self.assertEqual(task_rag.renewed(previous, task_adapters.adapter(task["kind"]))["code_id"], "0" * 64)

    def test_rag_code_change_does_not_touch_off_or_other_fingerprints(self):
        self.knowledge()
        self.change_code_after_consent("src/agents/rag_embed.py")
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["consents"]["A"]["content"]["rag"], {"mode": "off"})
        self.assertEqual((len(self.requests), self.embeds(), task["draft"]["rag"]["status"]), (1, 0, "off"))
        self.assertEqual(self.requests[0], self.baseline(task))

        # RAG 코드는 진단·배열 worker 지문 목록에 일괄 추가하지 않는다(동의 A의 rag 범위에서만 대조).
        for files in (module_pipeline_transport.CONTROL_FILES, array_task_transport.CONTROL_FILES):
            self.assertFalse({f"src/agents/{name}.py" for name in RAG_MODULES} & set(files))

    # ==========================================
    # 근거 예산: 로컬 토큰 계산(검증 구성)과 기존 바이트 추정
    # ==========================================

    def token_counter(self, *, fail=None):
        """UTF-8 4바이트 = 1토큰인 대역 토크나이저로 실제 Counter 규칙(형식 검사·렌더·64 KiB)을 쓴다. fail(payload)가 참이면 오류."""
        counter = qwen_tokens.Counter(LocalModelConfig(**self.model), FakeTokenizer())
        real_count = counter.count

        def count(payload):
            if fail is not None and fail(payload):
                raise fail.error
            return real_count(payload)

        counter.count = count
        self.prepare.side_effect, self.prepare.return_value = None, counter
        return counter

    def padded(self):
        # 근거 안내문을 늘려 기존 바이트 예산(8,832 B)은 넘고 토큰 예산(5,888)은 남게 한다.
        self.start_patch("src.agents.task_rag.EVIDENCE_INSTRUCTIONS", new=task_rag.EVIDENCE_INSTRUCTIONS + " x" * 1500)

    def test_local_token_budget_admits_evidence_that_byte_budget_omits(self):
        self.knowledge()
        self.padded()
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        before = task["draft"]["rag"]
        self.assertEqual((before["status"], before["budget"]["method"], before["budget"]["fallback_code"]),
                         ("budget_omitted", "bytes", "RAG_TOKENS_STATUS_FAILED"))
        self.assertEqual(before["included_items"], [])

        self.requests.clear()
        counter = self.token_counter()
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        usage, sent = task["draft"]["rag"], self.requests[0]
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(usage["status"], "graph")
        self.assertEqual(usage["budget"], {"method": "local_tokens", "count_id": qwen_tokens.COUNT_ID,
                                           "allowed_tokens": task_recommend.token_budget(LocalModelConfig(**self.model)),
                                           "input_tokens": counter.count(sent), "byte_estimate_tokens": None, "fallback_code": None})
        self.assertGreater(task_recommend._messages_bytes(sent) / task_recommend.BYTES_PER_TOKEN, usage["budget"]["allowed_tokens"])
        # 실제 보낸 근거와 기록이 같고, 진단·기본 계획·허용 값은 근거 없는 요청과 같다.
        evidence = content(sent).pop("evidence")
        self.assertEqual(usage["included_items"], [item["id"] for item in evidence["it"]])
        self.assertEqual(usage["included_relations"], [relation["id"] for relation in evidence["rel"]])
        self.assertEqual({key: value for key, value in content(sent).items() if key != "evidence"}, content(self.baseline(task)))
        self.assertIn("요청 예산: 로컬 토큰 계산", "\n".join(task_rag.view(task["draft"])["lines"]))
        stored = json.dumps(usage, ensure_ascii=False)
        for forbidden in ("<|im_start|>", "<think>", sent["messages"][0]["content"][:40], '"ids"'):
            self.assertNotIn(forbidden, stored)

    def test_unverified_configuration_uses_byte_budget_with_fixed_reason(self):
        self.knowledge()

        for code in ("RAG_TOKENS_SERVER_MISMATCH", "RAG_TOKENS_MODEL_MISMATCH", "RAG_TOKENS_FILE_MISMATCH",
                     "RAG_TOKENS_LIBRARY_MISMATCH", "RAG_TOKENS_CONFIG_MISMATCH"):
            with self.subTest(code):
                self.requests.clear()
                self.prepare.side_effect = qwen_tokens.TokenCountUnavailable(code)
                task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
                budget = task["draft"]["rag"]["budget"]
                self.assertEqual((task["draft"]["rag"]["status"], budget["method"], budget["fallback_code"]), ("graph", "bytes", code))
                self.assertTrue(task_recommend._fits(self.requests[0], LocalModelConfig(**self.model)))
                self.assertIn(task_rag.BUDGET_CODE_LABELS[code], "\n".join(task_rag.view(task["draft"])["lines"]))

    def test_unsupported_candidate_or_count_error_falls_back_to_bytes_for_whole_request(self):
        self.knowledge()
        self.padded()

        for name, error, code in (("unsupported", qwen_tokens.TokenCountUnavailable("RAG_TOKENS_INPUT_UNSUPPORTED"),
                                   "RAG_TOKENS_INPUT_UNSUPPORTED"),
                                  ("error", RuntimeError("secret detail C:\\x"), "RAG_TOKENS_COUNT_FAILED")):
            with self.subTest(name):
                self.requests.clear()
                fail = lambda payload: "evidence" in content(payload)  # noqa: E731
                fail.error = error
                self.token_counter(fail=fail)
                task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
                usage = task["draft"]["rag"]
                self.assertEqual((usage["status"], usage["budget"]["method"], usage["budget"]["fallback_code"]),
                                 ("budget_omitted", "bytes", code))
                self.assertEqual(self.requests, [self.baseline(task)])
                self.assertNotIn("secret", json.dumps(usage, ensure_ascii=False))

    def test_off_does_not_prepare_token_count(self):
        self.knowledge()
        task = orchestrator.diagnose(self.new_task()["task_id"], root=self.store, requester=self.requester)
        self.assertEqual((self.requests, task["draft"]["rag"]["budget"]), ([self.baseline(task)], None))
        self.prepare.assert_not_called()

    def test_unsupported_base_request_skips_server_status(self):
        self.knowledge()
        with patch.object(qwen_tokens, "supported", side_effect=qwen_tokens.TokenCountUnavailable("RAG_TOKENS_INPUT_UNSUPPORTED")):
            task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["draft"]["rag"]["budget"]["fallback_code"], "RAG_TOKENS_INPUT_UNSUPPORTED")
        self.prepare.assert_not_called()

    def test_no_evidence_does_not_prepare_token_count(self):
        self.knowledge(query=fx.unit(0, 0, 0, 0, 1))
        task = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        self.assertEqual((task["draft"]["rag"]["status"], task["draft"]["rag"]["budget"]), ("no_evidence", None))
        self.prepare.assert_not_called()

    def test_final_recheck_uses_same_budget_and_byte_limit(self):
        self.knowledge()
        self.padded()
        self.token_counter()
        record = orchestrator.diagnose(self.new_task("on")["task_id"], root=self.store, requester=self.requester)
        augment = task_rag.augmenter(record, task_adapters.adapter(record["kind"]))
        config = LocalModelConfig(**record["model"])
        augmented, usage = augment(self.baseline(record), config)
        self.assertEqual(usage["budget"]["method"], "local_tokens")
        self.assertTrue(augment.final_fits(augmented, config))
        self.assertFalse(task_recommend._fits(augmented, config))
        oversized = dict(augmented, format={"type": "object", "description": "x" * 70000})
        self.assertFalse(augment.final_fits(oversized, config))

    def test_consent_binds_budget_configuration(self):
        consent = orchestrator.consent_a_scope(self.new_task("on"))["rag"]
        self.assertEqual(consent["budget"], {"method": "local_tokens_or_bytes", "count_id": qwen_tokens.COUNT_ID})
        self.assertIn("src/agents/qwen_tokens.py", task_rag.CODE_FILES)
        self.assertTrue(any("요청 예산" in line for line in task_rag.consent_text("on", "qwen3.5:9b")))

    def test_invalid_setting_value_is_not_treated_as_on(self):
        self.knowledge()
        task = self.new_task()
        for value in ({"mode": "ON"}, {"mode": "on", "extra": 1}, "on"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    task_rag.mode(dict(task, rag=value))
                usage = task_rag.finalize(dict(task, rag=value), None)
                self.assertEqual(usage["status"], "not_attempted")

    def test_unsupported_kind_is_rejected_before_task_creation(self):
        source = Path(self.store).parent / "arrays.npz"
        np.savez(source, x=np.zeros((4, 3), dtype=np.float32))

        with self.assertRaises(task_rag.RagUnsupported):
            orchestrator.start_task("배열 전처리", source, flow.LIMITS, self.model, root=self.store, rag_mode="on")

        self.assertFalse(self.store.exists() and any(self.store.iterdir()))
        task = orchestrator.start_task("배열 전처리", source, flow.LIMITS, self.model, root=self.store)
        self.assertEqual((task["kind"], task["rag"]), ("array_npz", {"mode": "off"}))
        self.assertEqual(orchestrator.consent_a_scope(task)["rag"], {"mode": "off"})
        # 기록을 바꿔 ON으로 만들어도 프로필이 없으면 검색하지 않는다.
        adapter = task_adapters.adapter(task["kind"])
        with self.assertRaises(task_rag.RagUnsupported):
            task_rag.consent_content(dict(task, rag={"mode": "on"}), adapter)
        sent, usage = task_rag.augmenter(dict(task, rag={"mode": "on"}), adapter)({"messages": []}, None)
        self.assertEqual((usage["status"], usage["code"]), ("failed", "RAG_KIND_UNSUPPORTED"))

    def test_v2_tabular_accepts_on(self):
        task = orchestrator.start_task("label 예측", self.source, flow.LIMITS, self.model, root=self.store,
                                       input_generation="current", rag_mode="on")
        self.assertEqual((task["kind"], task["rag"]), ("tabular_v2", {"mode": "on"}))
        self.assertEqual(orchestrator.consent_a_scope(task)["rag"]["profile"], "tabular_v1")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.new_task("yes")


class RagScreenTest(unittest.TestCase):
    """요청 화면의 선택·동의 문구와 계획 확인 화면의 실제 사용 결과 표시(v2 표형, Streamlit AppTest)."""

    setUp_flow = flow.TaskOrchestratorTest.setUp
    setUp_csv = flow.TaskOrchestratorTest.setUp_csv
    start_patch = flow.TaskOrchestratorTest.start_patch
    transport = flow.TaskOrchestratorTest.transport
    knowledge = RagFlowTest.knowledge

    def setUp(self):
        from tests.agents.test_tabular_task_v2 import add_v2_patches

        self.setUp_flow()
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.rag_root, self.embedders = Path(temp.name), []
        self.start_patch("src.agents.task_store.STORE_ROOT", new=self.store)
        self.start_patch("src.ui.runtime_notice.server_permissions",
                         return_value={"elevated": True, "pid": 1, "python": "synthetic"})
        self.model_calls = self.start_patch("src.agents.task_recommend.request_local_proposal",
                                            side_effect=lambda payload, config: flow.model_reply())
        add_v2_patches(self, self.store, self.handoff_runs)
        self.start_patch("src.agents.qwen_tokens.prepare",
                         side_effect=qwen_tokens.TokenCountUnavailable("RAG_TOKENS_STATUS_FAILED"))

    def test_default_off_then_on_shows_scope_and_actual_evidence(self):
        from streamlit.testing.v1 import AppTest

        self.knowledge()
        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        radio = app.radio(key="task_rag_mode")
        self.assertEqual(radio.value, "off")
        self.assertTrue(any("검색 근거(실험적 Graph RAG): OFF(기본)" in item.value for item in app.info))
        radio.set_value("on").run()
        self.assertTrue(any("bge-m3:567m" in item.value and "권장 정확성 개선은 확인되지 않았습니다" in item.value
                            for item in app.info))
        self.assertTrue(any("지식집·색인: 검사 통과" in item.value for item in app.text))
        self.assertEqual(self.embedders, [])
        app.text_input(key="_widget_task_source").set_value(str(self.source))
        app.text_area(key="task_goal").set_value("label 예측용으로 분할·전처리해 주세요")
        app.checkbox(key="task_consent_a").check()
        next(item for item in app.button if item.label == "작업 시작").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        texts = [item.value for item in app.text]
        self.assertIn("검색 근거(실험적 Graph RAG): 선택 ON · 실제 결과: 근거와 그래프 관계를 전달함", texts)
        self.assertIn(task_rag.EXPERIMENTAL_NOTICE, texts)
        self.assertTrue(any(item.label.startswith("생성 요청에 포함한 지식") for item in app.expander))
        self.assertEqual((self.model_calls.call_count, sum(len(item.texts) for item in self.embedders)), (1, 1))


if __name__ == "__main__":
    unittest.main()

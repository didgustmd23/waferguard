"""실험적 Graph RAG 1단계: 지식 계약·로딩·색인·질의·검색·그래프 확장·근거 구성을 합성 지식과 모의 임베더로 검사한다."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.agents import rag_embed, rag_index, rag_knowledge, rag_tabular, task_rag
from src.agents.llm_config import load_local_config
from tests.agents import rag_fixtures as fx


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Temp(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)


class KnowledgeContractTest(Temp):
    def load(self, values):
        profile = fx.Profile(self.root)
        fx.write_repo(self.root)
        fx.write_pack(profile.PACK_DIR, values)
        return rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)

    def code(self, values):
        with self.assertRaises(rag_knowledge.KnowledgeError) as caught:
            self.load(values)
        return caught.exception.code

    def test_valid_pack_loads_with_identity(self):
        pack = self.load(fx.pack_values())
        self.assertEqual([item["id"] for item in pack["knowledge"]["items"]], ["K01", "K02", "K03", "K04"])
        self.assertRegex(pack["sha256"], r"^[a-f0-9]{64}$")
        self.assertEqual(pack["changed_sources"], [])

    def test_missing_pack_is_unavailable(self):
        profile = fx.Profile(self.root)
        with self.assertRaises(rag_knowledge.KnowledgeError) as caught:
            rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
        self.assertEqual(caught.exception.code, "RAG_KNOWLEDGE_UNAVAILABLE")

    def test_invalid_references_and_fields_are_rejected(self):
        base = fx.pack_values()
        cases = {
            "unknown source": fx.changed(base, ["knowledge", "items", 0, "sources"], [{"id": "S_NONE", "section": "x"}]),
            "unsupported kind": fx.changed(base, ["knowledge", "items", 0, "task_kinds"], ["npz_array"]),
            "unknown cue": fx.changed(base, ["knowledge", "items", 1, "cues"], ["made_up"]),
            "missing condition": fx.changed(base, ["knowledge", "items", 0, "applies_when"], []),
            "case with support": fx.changed(base, ["knowledge", "items", 3, "support"], "supported"),
            "extra field": fx.changed(base, ["knowledge", "items", 0, "concepts"], ["C_X"]),
            "unknown edge node": fx.changed(base, ["graph", "edges", 0, "target"], "N_NONE"),
            "unknown edge knowledge": fx.changed(base, ["graph", "edges", 0, "knowledge"], ["K99"]),
            "self loop": fx.changed(base, ["graph", "edges", 0, "target"], "N_OPT_A"),
            "scope evidence mismatch": fx.changed(base, ["graph", "edges", 1, "scope"], "product"),
            "absolute path text": fx.changed(base, ["knowledge", "items", 0, "body"], "C:\\Users\\x\\data.csv 참고"),
            "workspace text": fx.changed(base, ["knowledge", "items", 0, "body"], "_workspace/graph_rag_eval 결과"),
            "token text": fx.changed(base, ["knowledge", "items", 0, "body"], "토큰 " + "ab" * 16),
            "workspace source": fx.changed(base, ["sources", "sources", 0, "path"], "_workspace/x.py"),
            "parent source": fx.changed(base, ["sources", "sources", 0, "path"], "src/../x.py"),
            "missing source hash": fx.changed(base, ["sources", "sources", 0, "sha256"], None),
            "wrong embed model": fx.changed(base, ["pack", "embedding", "model"], "other:1b"),
            "wrong profile": fx.changed(base, ["pack", "profile"], "array_v1"),
        }
        contradiction = fx.pack_values()
        contradiction["graph"]["edges"].append(dict(contradiction["graph"]["edges"][0], id="E05", type="conflicts_with"))
        contradiction["graph"]["edges"][0]["type"] = "requires"
        cases["contradiction"] = contradiction

        for name, values in cases.items():
            with self.subTest(name):
                self.assertEqual(self.code(values), "RAG_KNOWLEDGE_INVALID")

    def official(self, title, *, kind="official_doc", version="scikit-learn 1.9.1 documentation"):
        values = fx.pack_values()
        values["sources"]["sources"].append({"id": "S_GUIDE", "kind": kind, "title": title, "path": None, "version": version,
                                             "sha256": None, "enforce": False})
        return values

    def test_official_doc_https_url_is_kept_and_local_paths_stay_rejected(self):
        """공식 문서 출처 제목의 https 주소는 그대로 허용하고, 로컬 경로·UNC·금지 경로·토큰은 주소와 함께 있어도 거부한다.

        문자열 검사만 한다. 경로에 접근하거나 주소에 접속하지 않는다.
        """
        allowed = {
            "https url": "Common pitfalls 12.2 Data leakage (https://scikit-learn.org/1.9/common_pitfalls.html)",
            "fragment": "3.1.2.4 Grouped data (https://scikit-learn.org/1.9/modules/cross_validation.html"
                        "#cross-validation-iterators-for-grouped-data)",
            "query and fragment": "re 모듈 https://docs.python.org/3/library/re.html?highlight=fullmatch&v=3#re.fullmatch 참고",
            "two urls": "https://scikit-learn.org/1.9/modules/impute.html#imputation-of-missing-values 및 "
                        "https://scikit-learn.org/1.9/common_pitfalls.html#data-leakage",
        }

        for name, title in allowed.items():
            with self.subTest(name):
                sources = self.load(self.official(title))["sources"]["sources"]
                self.assertEqual(sources[-1]["title"], title)

        rejected = {
            "drive path": self.official("C:\\Users\\x\\guide.html"),
            "drive slash": self.official("사본 D:/docs/guide.html"),
            "path mid sentence": self.official("공식 문서 사본(c:\\temp\\x.html)을 참고"),
            "unc": self.official("\\\\server\\share\\guide.html"),
            "unix home": self.official("https://scikit-learn.org/1.9/ 사본 /home/user/x.html"),
            "workspace in url": self.official("https://scikit-learn.org/_workspace/x.html"),
            "logs in url": self.official("https://example.org/logs/agents/run"),
            "token in query": self.official("https://example.org/doc?t=" + "ab" * 16),
            "url with drive path": self.official("https://scikit-learn.org/1.9/impute.html 및 C:\\Users\\x\\copy.html"),
            "drive inside url path": self.official("https://scikit-learn.org/C:/x.html"),
            "host without dot": self.official("https://C:/Users/x.html"),
            "http scheme": self.official("http://scikit-learn.org/1.9/impute.html"),
            "file scheme": self.official("file:///C:/docs/x.html"),
            "upper case scheme": self.official("HTTPS://scikit-learn.org/1.9/impute.html"),
            "git+https scheme": self.official("git+https://scikit-learn.org/x"),
            "view-source scheme": self.official("view-source:https://scikit-learn.org/x"),
            "jar scheme": self.official("jar:https://scikit-learn.org/x"),
            "glued prefix": self.official("xhttps://scikit-learn.org/x"),
            "url in product doc title": self.official("https://scikit-learn.org/1.9/impute.html", kind="product_doc"),
            "url in version": self.official("공식 문서", version="https://scikit-learn.org/1.9/"),
            "url in knowledge body": fx.changed(fx.pack_values(), ["knowledge", "items", 0, "body"],
                                                "https://scikit-learn.org/1.9/impute.html 참고"),
        }

        for name, values in rejected.items():
            with self.subTest(name):
                self.assertEqual(self.code(values), "RAG_KNOWLEDGE_INVALID")

    def test_duplicate_json_key_is_rejected(self):
        profile = fx.Profile(self.root)
        fx.write_repo(self.root)
        fx.write_pack(profile.PACK_DIR)
        (profile.PACK_DIR / "pack.json").write_text('{"schema_version": "rag_pack_v1", "schema_version": "x"}', encoding="utf-8")
        with self.assertRaises(rag_knowledge.KnowledgeError) as caught:
            rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
        self.assertEqual(caught.exception.code, "RAG_KNOWLEDGE_INVALID")

    def test_enforced_source_change_rejects_pack_and_doc_change_is_reported(self):
        self.load(fx.pack_values())
        (self.root / fx.DOC_PATH).write_bytes(b"changed doc\n")
        profile = fx.Profile(self.root)
        self.assertEqual(rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)["changed_sources"], ["S_DOC"])
        (self.root / fx.SOURCE_PATH).write_bytes(b"# changed product rule\n")
        with self.assertRaises(rag_knowledge.KnowledgeError) as caught:
            rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
        self.assertEqual((caught.exception.code, caught.exception.targets), ("RAG_SOURCE_CHANGED", ["S_RULE"]))


class IndexTest(Temp):
    def test_build_then_load_and_never_overwrite(self):
        profile, pack = fx.prepared(self.root)
        index = rag_index.load_index(profile.INDEX_ROOT, pack)
        self.assertEqual(index["ids"], ["K01", "K02", "K03", "K04"])
        self.assertEqual(index["matrix"].dtype, np.float32)
        with self.assertRaises(rag_index.RagIndexError) as caught:
            rag_index.build_index(profile.INDEX_ROOT, pack, fx.FakeEmbedder())
        self.assertEqual(caught.exception.code, "RAG_INDEX_EXISTS")

    def test_no_index_is_unavailable_and_build_failure_publishes_nothing(self):
        profile = fx.Profile(self.root)
        fx.write_repo(self.root)
        fx.write_pack(profile.PACK_DIR)
        pack = rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
        with self.assertRaises(rag_index.RagIndexError) as caught:
            rag_index.load_index(profile.INDEX_ROOT, pack)
        self.assertEqual(caught.exception.code, "RAG_INDEX_UNAVAILABLE")
        failing = fx.FakeEmbedder(error=rag_embed.EmbedError("RAG_EMBED_CONNECTION_FAILED"))
        with self.assertRaises(rag_embed.EmbedError):
            rag_index.build_index(profile.INDEX_ROOT, pack, failing)
        self.assertEqual(len(failing.texts), 1)
        self.assertFalse((profile.INDEX_ROOT / rag_index.index_name(pack)).exists())

    def test_tampered_vectors_manifest_and_changed_knowledge_are_rejected(self):
        profile, pack = fx.prepared(self.root)
        directory = profile.INDEX_ROOT / rag_index.index_name(pack)
        original = (directory / "vectors.npy").read_bytes()
        (directory / "vectors.npy").write_bytes(original[:-4] + b"\x00\x00\x80\x7f")
        with self.assertRaises(rag_index.RagIndexError) as caught:
            rag_index.load_index(profile.INDEX_ROOT, pack)
        self.assertEqual(caught.exception.code, "RAG_INDEX_INVALID")
        (directory / "vectors.npy").write_bytes(original)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        (directory / "manifest.json").write_text(json.dumps(dict(manifest, model=dict(manifest["model"], digest="b" * 64))),
                                                 encoding="utf-8")
        with self.assertRaises(rag_index.RagIndexError) as caught:
            rag_index.load_index(profile.INDEX_ROOT, pack)
        self.assertEqual(caught.exception.code, "RAG_INDEX_MODEL_MISMATCH")
        (directory / "manifest.json").write_text(json.dumps(dict(manifest, pack_sha256="0" * 64)), encoding="utf-8")
        with self.assertRaises(rag_index.RagIndexError) as caught:
            rag_index.load_index(profile.INDEX_ROOT, pack)
        self.assertEqual(caught.exception.code, "RAG_INDEX_MISMATCH")
        # 지식이 바뀌면 색인 이름이 달라져 이전 색인을 조용히 열지 않는다.
        fx.write_pack(profile.PACK_DIR, fx.changed(fx.pack_values(), ["knowledge", "items", 0, "body"], "바뀐 본문"))
        changed = rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
        with self.assertRaises(rag_index.RagIndexError) as caught:
            rag_index.load_index(profile.INDEX_ROOT, changed)
        self.assertEqual(caught.exception.code, "RAG_INDEX_UNAVAILABLE")

    def test_pickled_or_wrong_shape_vectors_are_rejected(self):
        profile, pack = fx.prepared(self.root)
        directory = profile.INDEX_ROOT / rag_index.index_name(pack)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

        for name, array in (("object", np.array([{"a": 1}], dtype=object)), ("shape", np.ones((4, 8), dtype=np.float32)),
                            ("unnormalized", np.ones((4, fx.DIMENSIONS), dtype=np.float32))):
            with self.subTest(name):
                path = directory / "vectors.npy"
                path.unlink()
                np.save(path, array, allow_pickle=(name == "object"))
                raw = path.read_bytes()
                (directory / "manifest.json").write_text(json.dumps(dict(manifest, vectors_sha256=hashlib.sha256(raw).hexdigest())),
                                                         encoding="utf-8")
                with self.assertRaises(rag_index.RagIndexError) as caught:
                    rag_index.load_index(profile.INDEX_ROOT, pack)
                self.assertEqual(caught.exception.code, "RAG_INDEX_INVALID")

    def test_oversized_header_is_rejected_before_allocation(self):
        profile, pack = fx.prepared(self.root)
        directory = profile.INDEX_ROOT / rag_index.index_name(pack)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        cases = {"huge shape": {"descr": "<f4", "fortran_order": False, "shape": (250000, fx.DIMENSIONS)},
                 "big endian": {"descr": ">f4", "fortran_order": False, "shape": (4, fx.DIMENSIONS)},
                 "fortran": {"descr": "<f4", "fortran_order": True, "shape": (4, fx.DIMENSIONS)}}

        for name, header in cases.items():
            with self.subTest(name):
                import io

                buffer = io.BytesIO()
                np.lib.format.write_array_header_1_0(buffer, header)
                raw = buffer.getvalue()
                (directory / "vectors.npy").write_bytes(raw)
                (directory / "manifest.json").write_text(json.dumps(dict(manifest, vectors_sha256=hashlib.sha256(raw).hexdigest())),
                                                         encoding="utf-8")
                with patch("src.agents.rag_index.np.load", side_effect=AssertionError("헤더 검사 전에 읽으면 안 됨")):
                    with self.assertRaises(rag_index.RagIndexError) as caught:
                        rag_index.load_index(profile.INDEX_ROOT, pack)
                self.assertEqual(caught.exception.code, "RAG_INDEX_INVALID")

    def test_build_refuses_linked_index_root(self):
        profile = fx.Profile(self.root)
        fx.write_repo(self.root)
        fx.write_pack(profile.PACK_DIR)
        pack = rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
        with patch("src.agents.rag_index._is_link", side_effect=lambda path: path == profile.INDEX_ROOT.parent.absolute()):
            profile.INDEX_ROOT.mkdir(parents=True)
            with self.assertRaises(rag_index.RagIndexError) as caught:
                rag_index.build_index(profile.INDEX_ROOT, pack, fx.FakeEmbedder())
        self.assertEqual(caught.exception.code, "RAG_INDEX_PATH_INVALID")
        self.assertEqual(list(profile.INDEX_ROOT.iterdir()), [])

    def test_rebuild_reuses_only_same_model_and_same_embed_text(self):
        profile, pack = fx.prepared(self.root)
        previous = rag_index.index_name(pack)
        fx.write_pack(profile.PACK_DIR, fx.changed(fx.pack_values(), ["knowledge", "items", 0, "body"], "바뀐 본문"))
        changed = rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
        embedder = fx.FakeEmbedder()
        result = rag_index.build_index(profile.INDEX_ROOT, changed, embedder, previous=previous)
        self.assertEqual((result["embedded"], result["reused"]), (1, 3))
        self.assertTrue(embedder.texts[0].startswith("합성 지식 K01\n바뀐 본문"))
        other = fx.changed(fx.pack_values(), ["pack", "embedding", "digest"], "c" * 64)
        fx.write_pack(profile.PACK_DIR, other)
        other_pack = rag_knowledge.load_pack(profile.PACK_DIR, profile, profile.REPO_ROOT)
        embedder = fx.FakeEmbedder()
        result = rag_index.build_index(profile.INDEX_ROOT, other_pack, embedder, previous=previous)
        self.assertEqual((result["embedded"], result["reused"]), (4, 0))


def payload(goal="label 예측", rows=None, row_count=100):
    fields = ["id", "dtype_group", "missing_ratio", "observed_unique", "blank_string_count", "pos_inf_count", "neg_inf_count",
              "finite_count", "mean", "std", "outlier_candidate_count", "reasons"]
    rows = rows if rows is not None else [["SECRET_COL", "numeric", 0.0, 80, 0, 0, 0, 100, 5.0, 1.0, 3, []]]
    content = {"goal": goal, "row_count": row_count, "repeated_rows": 0,
               "subjects": {"detail_fields": fields, "detailed": rows, "named": ["NAMED_ONLY"], "summarized": {"count": 0, "buckets": {}}},
               "base_plan": {}, "allowed_values": {}, "scope_note": "x"}
    return {"model": "m", "messages": [{"role": "system", "content": "지시"},
                                       {"role": "user", "content": json.dumps(content, ensure_ascii=False)}],
            "stream": False, "format": {}, "options": {}}


class QueryTest(unittest.TestCase):
    def test_query_uses_goal_and_cues_without_column_names_or_values(self):
        query = rag_tabular.build_query(payload())
        self.assertEqual(query["cues"], ["outliers"])
        self.assertTrue(query["text"].startswith("목표: label 예측\n데이터 특성: "))
        for forbidden in ("SECRET_COL", "NAMED_ONLY", "5.0", "80"):
            self.assertNotIn(forbidden, query["text"])

    def test_named_only_columns_do_not_create_cues(self):
        self.assertEqual(rag_tabular.build_query(payload(rows=[]))["cues"], [])

    def test_invalid_or_too_long_query_has_fixed_code(self):
        for value, code in ((payload(goal="x" * 2000), "RAG_QUERY_TOO_LONG"),
                            ({"messages": [{}, {"content": "{}"}]}, "RAG_QUERY_INVALID"),
                            (payload(rows=[["c", "numeric"]]), "RAG_QUERY_INVALID")):
            with self.subTest(code):
                with self.assertRaises(rag_tabular.QueryError) as caught:
                    rag_tabular.build_query(value)
                self.assertEqual(caught.exception.code, code)


class SearchGraphTest(Temp):
    def setUp(self):
        super().setUp()
        self.profile, self.pack = fx.prepared(self.root)
        self.index = rag_index.load_index(self.profile.INDEX_ROOT, self.pack)
        self.items = self.pack["knowledge"]["items"]

    def test_threshold_band_and_top_k(self):
        hits, _ = task_rag.search(self.index, fx.unit(0.9, 0.85, 0.84, 0.83), self.items, "tabular_v2", ["outliers", "few_values"])
        self.assertLessEqual(len(hits), task_rag.TOP_K)
        self.assertTrue(all(hit["score"] >= task_rag.MIN_SCORE for hit in hits))
        self.assertTrue(all(hits[0]["score"] - hit["score"] <= task_rag.RELATIVE_BAND for hit in hits))
        # 1위 대비 0.05 밖이면 기준 이상이어도 넣지 않는다.
        hits, _ = task_rag.search(self.index, fx.unit(0.9, 0.7), self.items, "tabular_csv", ["outliers"])
        self.assertEqual([hit["id"] for hit in hits], ["K01"])

    def test_unrelated_query_has_no_hits(self):
        hits, _ = task_rag.search(self.index, fx.unit(0, 0, 0, 0, 1), self.items, "tabular_csv", [])
        self.assertEqual(hits, [])

    def test_kind_and_cue_conditions_are_not_assumed(self):
        hits, filtered = task_rag.search(self.index, fx.unit(0.1, 1.0, 1.0, 1.0), self.items, "tabular_csv", [])
        # K02(outliers 필요)·K04(few_values 필요)는 단서가 없어서, K03은 v2 전용이라 제외된다.
        self.assertEqual((hits, filtered), ([], 3))

    def test_wrong_query_vector_is_rejected(self):
        for vector in ([0.0] * fx.DIMENSIONS, [1.0] * 8, [float("nan")] * fx.DIMENSIONS):
            with self.assertRaises(ValueError):
                task_rag.search(self.index, vector, self.items, "tabular_csv", [])

    def test_expansion_depth_cycle_kind_and_relation_cap(self):
        graph = self.pack["graph"]
        result = task_rag.expand(graph, self.items, ["K01"], "tabular_csv", ["outliers"])
        # 깊이 0: E01(K01 근거), 깊이 1: E01 끝 노드에 닿는 E02·E03. E04는 v2 전용 근거라 제외. 순환은 한 번만.
        self.assertEqual([relation["id"] for relation in result["relations"]], ["E01", "E02", "E03"])
        self.assertEqual((result["added"], result["skipped"]), (["K02"], 1))
        no_cue = task_rag.expand(graph, self.items, ["K01"], "tabular_csv", [])
        self.assertEqual([relation["id"] for relation in no_cue["relations"]], ["E01"])
        v2 = task_rag.expand(graph, self.items, ["K01"], "tabular_v2", ["outliers"])
        self.assertTrue(next(relation for relation in v2["relations"] if relation["id"] == "E04")["to_unsupported"])
        many = dict(graph, edges=[dict(graph["edges"][0], id=f"E{index:02d}") for index in range(10, 30)])
        capped = task_rag.expand(many, self.items, ["K01"], "tabular_csv", [])
        self.assertEqual((len(capped["relations"]), capped["truncated"]), (task_rag.MAX_RELATIONS, True))

    def test_depth_one_does_not_reach_two_hops(self):
        graph = self.pack["graph"]
        nodes = graph["nodes"] + [{"id": "N_FAR_E", "type": "effect", "label": "먼 효과"}]
        # E09는 검색 항목(K01)을 근거로 두지 않고, 깊이 1에서 닿는 N_COND_C에서 한 번 더 떨어진 관계다.
        edges = graph["edges"] + [dict(graph["edges"][1], id="E09", source="N_COND_C", target="N_FAR_E")]
        result = task_rag.expand(dict(graph, nodes=nodes, edges=edges), self.items, ["K01"], "tabular_csv", ["outliers"])
        self.assertNotIn("E09", [relation["id"] for relation in result["relations"]])


class FitTest(Temp):
    def setUp(self):
        super().setUp()
        self.profile, self.pack = fx.prepared(self.root)
        self.by_id = {item["id"]: item for item in self.pack["knowledge"]["items"]}
        self.expansion = task_rag.expand(self.pack["graph"], self.pack["knowledge"]["items"], ["K01"], "tabular_csv", ["outliers"])
        self.hits = [{"id": "K01", "score": 0.9}]

    def limit_fits(self, limit):
        return lambda value, config: len(json.dumps(value["messages"], ensure_ascii=False).encode("utf-8")) <= limit

    def test_items_and_relations_are_whole_and_original_content_is_kept(self):
        base = payload()
        final, items, relations, omitted = task_rag.fit(base, None, self.by_id, self.hits, self.expansion, self.limit_fits(10 ** 6))
        self.assertEqual([view["id"] for view in items], ["K01", "K02"])
        self.assertEqual([view["id"] for view in relations], ["E01", "E02", "E03"])
        self.assertEqual(omitted, [0, 0])
        content, original = json.loads(final["messages"][1]["content"]), json.loads(base["messages"][1]["content"])
        self.assertEqual({key: value for key, value in content.items() if key != "evidence"}, original)
        self.assertTrue(final["messages"][0]["content"].endswith(task_rag.EVIDENCE_INSTRUCTIONS))
        self.assertEqual(content["evidence"]["it"][0]["w"], ["K01 적용 조건"])

    def test_tight_budget_omits_whole_units_and_never_sends_relations_without_items(self):
        base = payload()
        size = len(json.dumps(base["messages"], ensure_ascii=False).encode("utf-8"))
        self.assertIsNone(task_rag.fit(base, None, self.by_id, self.hits, self.expansion, self.limit_fits(size + 100)))
        one = task_rag.fit(base, None, self.by_id, self.hits, self.expansion, self.limit_fits(size + 1250))
        self.assertIsNotNone(one)
        final, items, relations, omitted = one
        self.assertEqual([view["id"] for view in items], ["K01"])
        self.assertGreater(sum(omitted), 0)
        text = json.loads(final["messages"][1]["content"])["evidence"]
        for relation in text["rel"]:
            self.assertTrue(set(relation["kb"]) & {view["id"] for view in text["it"]})


class EmbedTransportTest(unittest.TestCase):
    def test_check_vector_rejects_bad_responses(self):
        good = {"model": "bge-m3:567m", "embeddings": [[1.0] + [0.0] * 1023], "prompt_eval_count": 10}
        self.assertEqual(len(rag_embed.check_vector(good, "bge-m3:567m", 1024)), 1024)
        for bad in (dict(good, model="other"), dict(good, embeddings=[[0.0] * 1024]), dict(good, embeddings=[[1.0] * 8]),
                    dict(good, embeddings=[[float("nan")] * 1024]), dict(good, prompt_eval_count=5000), dict(good, error="x")):
            with self.assertRaises(rag_embed.EmbedError):
                rag_embed.check_vector(bad, "bge-m3:567m", 1024)

    def test_single_embed_per_task_object_and_no_retry(self):
        embedder = rag_embed.OllamaEmbedder(load_local_config())
        calls = []

        def request(method, path, payload, limit):
            calls.append(path)
            if path == "/api/tags":
                return {"models": [{"name": "bge-m3:567m", "digest": fx.DIGEST}]}
            raise rag_embed.EmbedError("RAG_EMBED_CONNECTION_FAILED")

        with patch.object(embedder, "_request", side_effect=request):
            for _ in range(2):
                with self.assertRaises(rag_embed.EmbedError):
                    embedder.embed("목표: x", model="bge-m3:567m", digest=fx.DIGEST, dimensions=1024)
        self.assertEqual(calls, ["/api/tags", "/api/embed"])

    def test_digest_mismatch_sends_no_embed(self):
        embedder = rag_embed.OllamaEmbedder(load_local_config())
        calls = []

        def request(method, path, payload, limit):
            calls.append(path)
            return {"models": [{"name": "bge-m3:567m", "digest": "f" * 64}]}

        with patch.object(embedder, "_request", side_effect=request):
            with self.assertRaises(rag_embed.EmbedError) as caught:
                embedder.embed("목표: x", model="bge-m3:567m", digest=fx.DIGEST, dimensions=1024)
        self.assertEqual((caught.exception.code, calls), ("RAG_EMBED_MODEL_MISMATCH", ["/api/tags"]))

    def test_transport_uses_configured_loopback_without_proxy(self):
        embedder = rag_embed.OllamaEmbedder(load_local_config())
        with patch("src.agents.rag_embed.http.client.HTTPConnection") as connection:
            connection.return_value.getresponse.return_value.status = 500
            with self.assertRaises(rag_embed.EmbedError) as caught:
                embedder.identity("bge-m3:567m", fx.DIGEST)
        config = load_local_config()
        connection.assert_called_once_with(config.host, config.port, timeout=config.timeout_seconds)
        self.assertEqual(caught.exception.code, "RAG_EMBED_HTTP_FAILED")


class ProductPackTest(unittest.TestCase):
    """제품 경로의 실제 지식집과(있으면) 이 PC에서 구축한 색인을 제품 로더로 연다. 모델을 호출하지 않는다."""

    def test_product_pack_loads_from_product_paths_only(self):
        pack = rag_knowledge.load_pack(rag_tabular.PACK_DIR, rag_tabular, rag_tabular.REPO_ROOT)
        self.assertEqual((pack["pack"]["pack_id"], pack["pack"]["profile"]), ("tabular-knowledge", rag_tabular.PROFILE_ID))
        self.assertEqual((len(pack["knowledge"]["items"]), len(pack["graph"]["edges"]), len(pack["sources"]["sources"])), (38, 65, 20))
        self.assertEqual(pack["changed_sources"], [])
        self.assertTrue(all(item["review"]["human"] == "pending" for item in pack["knowledge"]["items"]))
        # 출처는 저장소 제품 폴더 또는 경로 없는 공식 문서(https 주소)만이다.
        for source in pack["sources"]["sources"]:
            if source["path"] is None:
                self.assertEqual(source["kind"], "official_doc")
            else:
                self.assertTrue(source["path"].startswith(rag_knowledge.SOURCE_ROOTS))
        self.assertEqual(sum("(https://scikit-learn.org/1.9/" in source["title"] for source in pack["sources"]["sources"]), 3)
        raw = "".join((rag_tabular.PACK_DIR / name).read_text(encoding="utf-8") for name in rag_knowledge.FILES.values())
        self.assertNotIn("_workspace", raw)

    @unittest.skipUnless(any(rag_tabular.INDEX_ROOT.glob("tabular-knowledge-*")), "이 PC에서 색인을 구축하지 않았음(Git 제외 생성물)")
    def test_built_index_matches_product_pack(self):
        pack = rag_knowledge.load_pack(rag_tabular.PACK_DIR, rag_tabular, rag_tabular.REPO_ROOT)
        index = rag_index.load_index(rag_tabular.INDEX_ROOT, pack)
        self.assertEqual(index["ids"], [item["id"] for item in pack["knowledge"]["items"]])
        self.assertEqual(index["matrix"].shape, (38, rag_tabular.DIMENSIONS))


class IndependenceTest(unittest.TestCase):
    def test_product_modules_do_not_reference_workspace_code(self):
        for name in ("rag_knowledge", "rag_index", "rag_embed", "rag_tabular", "task_rag", "rag_index_cli"):
            text = (PROJECT_ROOT / "src/agents" / (name + ".py")).read_text(encoding="utf-8")
            code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            with self.subTest(name):
                self.assertNotIn("import ragkit", code)
                self.assertNotIn("from ragkit", code)
                self.assertNotIn("graph_rag_eval/", code.replace("_workspace/graph_rag_eval/phase2/params.json", ""))
                self.assertNotIn("allow_pickle=True", code)


if __name__ == "__main__":
    unittest.main()

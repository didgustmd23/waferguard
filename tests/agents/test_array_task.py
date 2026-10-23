"""배열 표본 묶음(.npz) 통합 작업: 진단 → 권장안·질문 → 최종 계획 → 분할·Train 기준 → 적용 → 결과를 합성 입력으로 검사한다.

OS transport(AppContainer·Job)와 로컬 모델 응답만 모의한다. 격리 worker 계산·부모 검증·승인 저장·위임 발급은 실제 코드다.
"""

from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from src.agents import task_adapters, task_orchestrator as orchestrator, task_store
from src.agents.array_task_worker import execute_request
from src.agents.isolated_eda import _valid_png
from src.agents.llm_config import load_local_config
from src.agents.task_plan import PlanError
from src.preprocessing import array_dataset as core


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"wall_seconds": 60, "memory_mib": 2048, "processes": 2}
LABELS = ["alpha", "beta", "gamma"]


def synthetic(path, *, nonfinite=True, duplicates=True, unlabeled=2, rows=60):
    """합성 3채널 8×8 표본, 라벨(일부 비어 있음)·그룹·ID. 채널 2는 0/1 값만 갖는다. 실제 데이터가 아니다."""
    rng = np.random.default_rng(7)
    x = rng.normal(10.0, 3.0, size=(rows, 8, 8, 3)).astype(np.float32)
    x[..., 2] = (rng.random((rows, 8, 8)) < 0.05).astype(np.float32)

    if nonfinite:
        x[3, 0, 0, 0] = np.nan

    if duplicates:
        x[11] = x[10]
        x[41] = x[40]

    labels = np.array([LABELS[index % 3] for index in range(rows)])
    labels[:unlabeled] = ""
    groups = np.array([index // 4 for index in range(rows)], dtype=np.int64)
    np.savez(path, x=x, y=labels, g=groups, extra=np.arange(5))
    return x


def model_reply(payload, config):
    return json.dumps({"summary": ["표본 60개, 3채널"], "notes": [], "questions": [],
                       "changes": [{"path": "mapping.channel_axis", "value": "last", "reason": "마지막 축 길이가 3입니다"}]},
                      ensure_ascii=False)


class ArrayCoreTest(unittest.TestCase):
    """계산 코어: npz 헤더 검사·매핑·미리보기·묶기·Train 기준. 에이전트 없이 호출한다."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_headers_reject_object_arrays_non_npy_members_and_oversized_archives(self):
        np.savez(self.root / "object.npz", x=np.array([{"a": 1}, None], dtype=object))

        with zipfile.ZipFile(self.root / "text.npz", "w") as archive:
            archive.writestr("note.txt", "hello")

        np.savez(self.root / "ok.npz", x=np.zeros((4, 2, 2)))

        for name, code in (("object.npz", "NPY_DTYPE_UNSUPPORTED"), ("text.npz", "NPZ_MEMBER_INVALID")):
            with self.subTest(name=name), self.assertRaises(core.ArrayInputError) as caught:
                core.read_headers(self.root / name)

            self.assertEqual(caught.exception.code, code)

        with self.assertRaises(core.ArrayInputError) as caught:
            core.read_headers(self.root / "ok.npz", max_total=64)

        self.assertEqual(caught.exception.code, "NPZ_TOO_LARGE")

    def test_zero_size_dtype_negative_or_huge_shapes_and_npy_suffix_keys_are_rejected(self):
        def archive(name, member, header):
            buffer = io.BytesIO()
            np.lib.format.write_array_header_1_0(buffer, header)

            with zipfile.ZipFile(self.root / name, "w") as stream:
                stream.writestr(member, buffer.getvalue())

            return self.root / name

        cases = (("u0.npz", "y.npy", {"descr": "<U0", "fortran_order": False, "shape": (10 ** 12,)}, "NPY_DTYPE_UNSUPPORTED"),
                 ("huge.npz", "z.npy", {"descr": "|b1", "fortran_order": False, "shape": (10 ** 9,)}, "NPY_HEADER_INVALID"))

        for name, member, header, code in cases:
            with self.subTest(name=name), self.assertRaises(core.ArrayInputError) as caught:
                core.read_headers(archive(name, member, header))

            self.assertIn(caught.exception.code, (code, "NPY_HEADER_INVALID"))

        np.savez(self.root / "keys.npz", **{"a": np.zeros((2, 2)), "a.npy": np.ones((2, 2))})

        with self.assertRaises(core.ArrayInputError) as caught:
            core.read_headers(self.root / "keys.npz")

        self.assertEqual(caught.exception.code, "NPZ_MEMBER_INVALID")

    def test_mapping_is_shape_based_and_checks_lengths(self):
        synthetic(self.root / "a.npz")
        headers = core.read_headers(self.root / "a.npz")
        # 이름 단서로 라벨·그룹을 추정하지 않는다.
        self.assertEqual(core.base_mapping(headers), {"samples": "x", "channel_axis": "none", "label": None, "group": None, "sample_id": None})
        self.assertEqual(core.check_mapping(headers, dict(core.base_mapping(headers), channel_axis="last", label="y")), (60, 3))

        for change, code in (({"label": "extra"}, "MAPPING_META_INVALID"), ({"group": "y", "label": "y"}, "MAPPING_META_INVALID")):
            with self.subTest(change=change), self.assertRaises(core.ArrayInputError) as caught:
                core.check_mapping(headers, dict(core.base_mapping(headers), **change))

            self.assertEqual(caught.exception.code, code)

        # 채널 축 길이는 1~16만 허용한다(두 번째 축 길이 32).
        wide = {"z": {"dtype": "<f4", "kind": "f", "shape": [5, 32, 32]}}

        with self.assertRaises(core.ArrayInputError) as caught:
            core.check_mapping(wide, dict(core.base_mapping(wide), channel_axis="first"))

        self.assertEqual(caught.exception.code, "MAPPING_CHANNEL_INVALID")

    def test_previews_are_limited_valid_png_without_changing_data(self):
        x = synthetic(self.root / "a.npz")
        before = x.copy()
        shown = core.previews(x)
        self.assertEqual([position for position, _ in shown], list(range(12)))
        self.assertTrue(all(_valid_png(data) for _, data in shown))
        np.testing.assert_array_equal(x, before)
        self.assertEqual(core.previews(np.zeros((3, 2, 2, 2, 2))), [])

    def test_duplicate_grouping_merges_group_and_hash_components(self):
        self.assertEqual(core._components([["a", "b", "c", "d"], ["h1", "h2", "h1", "h3"]]), [0, 1, 0, 3])
        # 0-1은 그룹 a, 0-3은 해시 h로 연결되고 2는 따로 남는다.
        self.assertEqual(len(set(core._components([["a", "a", "b", "c"], ["h", "x", "y", "h"]]))), 2)

    def test_train_basis_and_binary_channel_clipping_collapse(self):
        train = np.zeros((20, 2, 4, 4))
        train[:, 0] = np.arange(20)[:, None, None]
        train[:2, 1, 0, 0] = 1.0  # 대부분 0인 0/1 채널
        options = {"nonfinite": "block", "clip": "percentile", "clip_low": 1.0, "clip_high": 99.0, "scaling": "standard", "fixed_divisor": 1.0}
        state = core.derive_state(train, options)
        _, impact = core.apply_state(train, state, options)
        self.assertEqual(state["channels"][1]["clip"], [0.0, 0.0])
        self.assertEqual((impact[1]["clipped"], impact[1]["constant_after"]), (2, True))
        # 저장 기준만 적용한다. 다른 배치를 넣어도 기준은 바뀌지 않는다.
        other = train * 100
        core.apply_state(other, state, options)
        self.assertEqual(core.derive_state(train, options), state)

        with self.assertRaises(core.ArrayInputError) as caught:
            core.derive_state(np.full((2, 1, 2), np.nan), dict(options, nonfinite="block"))

        self.assertEqual(caught.exception.code, "NONFINITE_PRESENT")


class ArrayMetadataValidationTest(unittest.TestCase):
    def test_json_scalar_types_include_bool_without_accepting_nested_values(self):
        from src.agents.array_task_results import _scalar

        for value in (True, False, None, 0, 1, 1.5, "label", "x" * 200):
            with self.subTest(value=value):
                self.assertTrue(_scalar(value))

        for value in ([], {}, [True], {"label": True}, "x" * 201, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertFalse(_scalar(value))


class ArrayTaskFlowTest(unittest.TestCase):
    """공통 조정기 + 배열 어댑터 + 격리 worker(모의 transport) + 부모 검증."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = self.root / "tasks"
        self.source = self.root / "samples.npz"
        self.x = synthetic(self.source)
        self.calls, self.requests, self.tamper = [], [], None

        for target, kwargs in (("src.agents.array_task_service._base", {"return_value": self.root / "array_runs"}),
                               ("src.agents.array_task_service.run_array_worker", {"side_effect": self.transport})):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.model = asdict(load_local_config())

    def transport(self, request, root, signal):
        self.calls.append(request.stage)
        reply = execute_request(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")

        if self.tamper is not None and reply.status == "succeeded":
            self.tamper(request.stage, Path(request.output_directory))

        outcome = {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes}}}
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
        return outcome

    def requester(self, payload, config):
        self.requests.append(json.dumps(payload, ensure_ascii=False))
        return model_reply(payload, config)

    def reviewed(self):
        task = orchestrator.start_task("y 라벨 분류용으로 배열을 분할·전처리해 주세요", self.source, LIMITS, self.model, root=self.store)
        self.assertEqual(task["kind"], "array_npz")
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        return task

    def plan(self, task, **preparation):
        state = orchestrator.review_state(task)
        plan = json.loads(json.dumps(state["plan"]))
        plan["mapping"].update(label="y", group="g")
        plan["preparation"].update({"nonfinite": "median", "missing_labels": "exclude", "duplicate_grouping": True,
                                    "clip": "percentile", "scaling": "standard", **preparation})
        return plan

    def confirm(self, task, plan, answers=None):
        preview = orchestrator.preview_plan(task, plan, answers or {})
        return orchestrator.confirm_plan(task["task_id"], plan, answers or {}, root=self.store, expected_digest=preview["digest"])

    def test_npz_from_request_to_result_with_train_only_basis_and_no_leakage(self):
        before = self.source.read_bytes()
        task = self.reviewed()
        self.assertEqual(self.calls, ["audit"])
        self.assertEqual(task["draft"]["source"], "model")
        self.assertEqual(task["draft"]["plan"]["mapping"]["channel_axis"], "last")
        adapter = task_adapters.adapter(task["kind"])
        previews = adapter.preview_images(task)
        self.assertEqual(len(previews), 12)
        self.assertTrue(all(_valid_png(item["data"]) and "처리 결과 아님" in item["label"] for item in previews))

        # 모델 요청에는 라벨 이름·최솟값·최댓값·경로·미리보기가 없다. 순번별 개수만 있다.
        sent = "".join(self.requests)

        for forbidden in (*LABELS, str(self.source), '\\"min\\"', '\\"max\\"', "PNG", "preview"):
            self.assertNotIn(forbidden, sent)

        self.assertIn("class_counts", sent)

        # 질문 없이도 확정 전에는 아무것도 실행하지 않는다. 라벨을 고르면 결측 처리가 필수 질문이 된다.
        state = orchestrator.review_state(task)
        ids = {item["id"]: item["level"] for item in state["questions"]}
        self.assertEqual(ids.get("mapping.label"), "advisory")
        self.assertEqual(ids.get("preparation.nonfinite"), "required")
        # 채널 2는 약 5%만 1이다. 상한 분위수 90이면 하한·상한이 모두 0이 되어 상수가 된다(독립 계산).
        plan = self.plan(task, missing_labels=None, clip_high=90.0)

        with self.assertRaises(PlanError) as caught:
            orchestrator.preview_plan(task, plan, {})

        self.assertEqual(caught.exception.code, "REQUIRED_QUESTION_UNANSWERED")
        self.assertEqual(self.calls, ["audit"])
        # 질문은 표시한 계획(라벨 미지정)에서 계산된다. 라벨을 고른 사용자는 양식의 '라벨 없는 표본' 항목으로 정한다.
        plan["preparation"]["missing_labels"] = "exclude"
        task = self.confirm(task, plan)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        self.assertEqual(self.calls, ["audit", "derive_preparation"])
        content = task["consents"]["C"]["content"]
        self.assertEqual(content["preparation"]["missing_labels"], "exclude")
        self.assertIn(["array_prepare", "derive_preparation"], content["allowed"])
        self.assertNotIn(["array_apply", "apply_preparation"], content["allowed"])

        # 후보: 라벨 없는 2개 제외, 같은 표본은 한 구간, 그룹 겹침 없음, 기준은 Train에서만.
        verified = adapter._candidate(task)
        assignment = verified["assignment"]
        self.assertEqual(assignment["excluded"], [0, 1])
        self.assertEqual((assignment["cross_split_duplicates"], assignment["group_overlap"]), (0, 0))
        owner = {position: name for name, values in assignment["partitions"].items() for position in values}
        self.assertEqual(owner[10], owner[11])
        train = np.moveaxis(self.x[assignment["partitions"]["train"]], -1, 1).astype(np.float64)
        expected = float(np.nanmedian(train[:, 0]))
        self.assertAlmostEqual(verified["state"]["channels"][0]["impute"], expected, places=4)
        facts = task["candidate"]["facts"]
        self.assertTrue(any("채널 2" in text and "정보 손실" in text for text in facts["warnings"]))
        self.assertTrue(any("예상치" in text for text in facts["notes"]))
        self.assertEqual(self.calls.count("apply_preparation"), 0)

        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        folder, application = adapter.service("array_apply").verified(task["steps"]["apply"]["run_name"])
        names = {path.name for path in folder.iterdir()}
        self.assertNotIn("test_samples.npy", names)
        train_out = np.load(folder / "train_samples.npy", allow_pickle=False)
        self.assertEqual(train_out.shape, (len(assignment["partitions"]["train"]), 8, 8, 3))
        self.assertEqual(train_out.dtype, np.float32)
        self.assertEqual(application["impact"]["train"]["constant_channels"], 1)
        meta = json.loads((folder / "validation_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["positions"], assignment["partitions"]["validation"])
        report = task["report"]
        text = report["text"]

        for title in ("## 핵심 결과와 주요 문제", "## 실제 전후 비교", "## 경고: 정보 손실·미처리·미검증", "## 적용 구간과 결과 파일 활용"):
            self.assertIn(title, text)

        comparison = {row["항목"]: row for row in report["facts"]["comparison"]}
        self.assertEqual(comparison["Test 구간"]["후 (실측)"], "변환 적용 안 함")
        self.assertTrue(comparison["변환 후 상수 채널"]["예상 (후보)"].startswith("train "))
        self.assertTrue(any("클래스 의미" in item["내용"] for item in report["facts"]["warnings"]))
        self.assertEqual(self.source.read_bytes(), before)

    def test_bool_labels_survive_diagnosis_candidate_and_application(self):
        # 원본은 합성 fixture다. bool 라벨이 입력 계약부터 최종 메타데이터까지 같은 타입·값·순서로 남아야 한다.
        with np.load(self.source, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        labels = np.arange(len(self.x)) % 2 == 0
        data["y"] = labels
        np.savez(self.source, **data)
        before = self.source.read_bytes()

        task = self.reviewed()
        task = self.confirm(task, self.plan(task))
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        adapter = task_adapters.adapter(task["kind"])
        assignment = adapter._candidate(task)["assignment"]
        self.assertEqual(assignment["excluded"], [])

        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        folder, _ = adapter.service("array_apply").verified(task["steps"]["apply"]["run_name"])
        for split in ("train", "validation"):
            meta = json.loads((folder / f"{split}_meta.json").read_text(encoding="utf-8"))
            positions = assignment["partitions"][split]
            self.assertEqual(meta["positions"], positions)
            self.assertTrue(all(type(value) is bool for value in meta["label"]))
            self.assertEqual(meta["label"], [bool(labels[position]) for position in positions])

        self.assertEqual(self.calls, ["audit", "derive_preparation", "apply_preparation"])
        self.assertEqual(self.source.read_bytes(), before)

    def test_duplicates_without_grouping_are_reported_as_leakage_not_moved(self):
        task = self.reviewed()
        plan = self.plan(task, duplicate_grouping=False)
        plan["split"]["strategy"] = "random"
        plan["mapping"]["group"] = None
        task = self.confirm(task, plan)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        assignment = task_adapters.adapter(task["kind"])._candidate(task)["assignment"]
        warned = any("누수" in text for text in task["candidate"]["facts"]["warnings"])
        self.assertEqual(warned, assignment["cross_split_duplicates"] > 0)
        self.assertEqual(task["confirmed"]["plan"]["preparation"]["duplicate_grouping"], False)

    def test_plan_checks_are_explained_without_automatic_changes(self):
        task = self.reviewed()

        for change, code in (({"split": {"strategy": "group"}, "mapping": {"group": None}}, "GROUP_REQUIRED"),
                             ({"split": {"stratify": True}}, "STRATIFY_INVALID"),
                             ({"preparation": {"clip_low": 40.0, "clip_high": 60.0, "clip": "percentile"}, "split": {"strategy": "random"}}, None),
                             ({"mapping": {"label": "extra"}}, "MAPPING_META_INVALID")):
            plan = self.plan(task)

            for section, values in change.items():
                plan[section].update(values)

            with self.subTest(change=change):
                if code is None:
                    orchestrator.preview_plan(task, plan, {})
                else:
                    with self.assertRaises(PlanError) as caught:
                        orchestrator.preview_plan(task, plan, {})

                    self.assertEqual(caught.exception.code, code)

        self.assertEqual(task_store.load_task(task["task_id"], root=self.store)["steps"], {})

    def test_tampered_worker_outputs_are_not_registered(self):
        def leak(stage, output):
            if stage == "derive_preparation":
                path = output / "candidate/assignment.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["cross_split_duplicates"] = 1
                path.write_text(json.dumps(value), encoding="utf-8")

        task = self.reviewed()
        self.tamper = leak
        task = self.confirm(task, self.plan(task))
        self.assertEqual(task["phase"], "needs_attention")
        self.assertIsNone(task["candidate"])

    def test_tampered_application_constant_count_is_rejected(self):
        def tamper(stage, output):
            if stage == "apply_preparation":
                path = output / "prepared/application.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["impact"]["validation"]["constant_channels"] += 1
                path.write_text(json.dumps(value), encoding="utf-8")

        task = self.reviewed()
        task = self.confirm(task, self.plan(task))
        self.tamper = tamper
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "needs_attention")
        self.assertIsNone(task.get("report"))

    def test_invalid_preview_blocks_diagnosis_with_next_step(self):
        def tamper(stage, output):
            if stage == "audit":
                (output / "profile/previews/preview_0.png").write_bytes(b"not a png")

        self.tamper = tamper
        task = orchestrator.start_task("배열 전처리", self.source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "needs_attention")
        self.assertIn("새 작업", task["problem"]["message"])
        self.assertEqual(self.requests, [])

    def test_unsupported_npz_is_blocked_before_model_with_cause(self):
        np.savez(self.source, x=np.array(["a", "b"]))
        task = orchestrator.start_task("배열 전처리", self.source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual((task["phase"], task["problem"]["code"]), ("needs_attention", "NO_SAMPLE_ARRAY"))
        self.assertEqual(self.requests, [])

    def test_group_overlap_tamper_in_group_split_is_rejected(self):
        def tamper(stage, output):
            if stage == "derive_preparation":
                path = output / "candidate/assignment.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["group_overlap"] = 1
                path.write_text(json.dumps(value), encoding="utf-8")

        task = self.reviewed()
        plan = self.plan(task, duplicate_grouping=False)
        plan["split"]["strategy"] = "group"
        self.tamper = tamper
        task = self.confirm(task, plan)
        self.assertEqual((task["phase"], task["candidate"]), ("needs_attention", None))

    def test_committed_candidate_changed_after_review_is_not_applied(self):
        task = self.reviewed()
        task = self.confirm(task, self.plan(task))
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        adapter = task_adapters.adapter(task["kind"])
        folder, _ = adapter.service("array_prepare").verified(task["steps"]["prepare"]["run_name"])
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        state["channels"][0]["center"] += 1.0
        (folder / "state.json").write_text(json.dumps(state), encoding="utf-8")

        # 형식이 맞아도 완료 표식 해시가 달라 결과를 다시 읽지 않는다. 적용 동의·적용 실행은 생기지 않는다.
        with self.assertRaises(ValueError):
            adapter.service("array_prepare").verified(task["steps"]["prepare"]["run_name"])

        result = None

        try:
            result = orchestrator.approve_apply(task["task_id"], root=self.store)
        except Exception:
            pass

        stored = task_store.load_task(task["task_id"], root=self.store)
        self.assertNotIn("D", stored["consents"])
        self.assertNotEqual((result or {}).get("phase"), "completed")
        self.assertEqual(self.calls.count("apply_preparation"), 0)

    def test_preview_order_or_count_tamper_blocks_diagnosis(self):
        def tamper(stage, output):
            if stage == "audit":
                path = output / "profile/summary.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["previews"] = list(reversed(value["previews"]))
                path.write_text(json.dumps(value), encoding="utf-8")

        self.tamper = tamper
        task = orchestrator.start_task("배열 전처리", self.source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "needs_attention")
        self.assertEqual(self.requests, [])

    def test_constant_or_tiny_arrays_do_not_send_mean_or_std(self):
        np.savez(self.source, x=np.full((6, 4, 4), 255.0, dtype=np.float32), y=np.array(list("abcabc")))
        task = orchestrator.start_task("배열 전처리", self.source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        detail = next(item for item in task_adapters.adapter(task["kind"]).recommendation_input(task)["items"] if item["id"] == "x")["detail"]
        self.assertEqual((detail["mean_of_finite_values"], detail["std_of_finite_values"]), (None, None))
        # 실제 요청 본문의 해당 배열 행에도 평균·표준편차가 없다(기본 계획의 고정 나눗값 255.0과는 무관).
        content = json.loads(json.loads(self.requests[0])["messages"][1]["content"])
        fields = content["subjects"]["detail_fields"]
        row = next(item for item in content["subjects"]["detailed"] if item[0] == "x")
        self.assertEqual((row[fields.index("mean_of_finite_values")], row[fields.index("std_of_finite_values")]), (None, None))

    def test_zero_and_one_other_value_arrays_do_not_send_mean_or_std_but_keep_local_values(self):
        mask = np.zeros((8, 4, 4), dtype=np.float32)
        mask[:, 0, 0] = 255.0  # 값은 0과 255 두 종류. 0 비율과 평균으로 255를 복원할 수 있다.
        np.savez(self.source, x=mask)
        task = orchestrator.start_task("배열 전처리", self.source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        self.assertEqual(task["phase"], "awaiting_review", task.get("problem"))
        content = json.loads(json.loads(self.requests[0])["messages"][1]["content"])
        fields = content["subjects"]["detail_fields"]
        row = next(item for item in content["subjects"]["detailed"] if item[0] == "x")
        self.assertEqual((row[fields.index("mean_of_finite_values")], row[fields.index("std_of_finite_values")], row[fields.index("distinct_values_capped_257")]), (None, None, 2))
        # 로컬 진단에는 그대로 남는다.
        self.assertAlmostEqual(task["diagnostic"]["summary"]["samples"]["x"]["overall"]["mean"], 255.0 / 16, places=4)

    def test_float32_storage_is_explained_in_options_review_and_report(self):
        task = self.reviewed()
        guide = {item["path"]: item for item in orchestrator.review_state(task)["guide"]}
        self.assertIn("float32", guide["preparation.scaling"]["caution"])
        self.assertFalse(any("정밀한 자료형" in text for text in guide["preparation.scaling"]["warnings"]))
        task = self.confirm(task, self.plan(task, scaling="none", clip="none"))
        self.assertTrue(any("float32" in text for text in task["candidate"]["facts"]["notes"]))
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertTrue(any(item["구분"] == "정보 손실" and "float32" in item["내용"] for item in task["report"]["facts"]["warnings"]))

        # float64 입력은 값과 관계없이 사전 경고한다. 설정은 바꾸지 않는다.
        np.savez(self.source, x=np.arange(64, dtype=np.float64).reshape(4, 4, 4) / 3)
        task = orchestrator.start_task("배열 전처리", self.source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)
        guide = {item["path"]: item for item in orchestrator.review_state(task)["guide"]}
        self.assertTrue(any("정밀한 자료형" in text for text in guide["preparation.scaling"]["warnings"]))
        self.assertEqual(task["draft"]["plan"]["preparation"]["scaling"], "none")

    def test_runtime_failure_in_diagnosis_does_not_claim_input_problem_or_raise_limits(self):
        original = self.transport

        def exceeded(request, root, signal):
            outcome = original(request, root, signal)
            outcome["launch"]["status"] = "resource_exceeded"
            (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
            return outcome

        with patch("src.agents.array_task_service.run_array_worker", side_effect=exceeded):
            task = orchestrator.start_task("배열 전처리", self.source, LIMITS, self.model, root=self.store)
            task = orchestrator.diagnose(task["task_id"], root=self.store, requester=self.requester)

        self.assertEqual((task["phase"], task["problem"]["code"]), ("needs_attention", "DIAGNOSTIC_RUNTIME_FAILED"))
        self.assertIn("메모리", task["problem"]["message"])
        self.assertIn("자동으로 늘리지 않았습니다", task["problem"]["message"])
        self.assertEqual(task["limits"], LIMITS)
        self.assertEqual(self.requests, [])

    def test_real_run_draft_errors_are_marked_and_label_mapping_needs_confirmation(self):
        """작업 94197a03…의 실제 모델 응답 형태 재현: 그룹 배열을 라벨로 권함, 기본값과 같은 변경, 틀린 비율의 중복 질문."""
        def reply(payload, config):
            self.requests.append(json.dumps(payload, ensure_ascii=False))
            return json.dumps({"summary": ["y는 문자열이라 수치 전처리 불가"], "notes": ["nonfinite 비율 0.3423"],
                               "changes": [{"path": "mapping.samples", "value": "x", "reason": "기본 계획과 동일"},
                                           {"path": "mapping.label", "value": "g", "reason": "g 라벨이 균일 분포"},
                                           {"path": "preparation.nonfinite", "value": "block", "reason": "nonfinite 비율 0.3423"}],
                               "questions": [{"path": "preparation.nonfinite", "text": "nonfinite 값(비율 0.3423)을 어떻게 처리할까요?",
                                              "options": ["block", "median"]},
                                             {"path": "preparation.scaling", "text": "y는 문자열이므로 수치 전처리를 적용하지 않을까요?",
                                              "options": ["none", "standard"]}]}, ensure_ascii=False)

        task = orchestrator.start_task("y 라벨 분류용 전처리", self.source, LIMITS, self.model, root=self.store)
        task = orchestrator.diagnose(task["task_id"], root=self.store, requester=reply)
        self.assertEqual(task["draft"]["source"], "model")
        # 기본 계획과 같은 값은 권장 변경으로 남지 않는다. 형식 검사를 통과한 변경은 라벨 매핑 하나뿐이다.
        self.assertEqual([item["path"] for item in task["draft"]["changes"]], ["mapping.label"])
        state = orchestrator.review_state(task)
        self.assertTrue(state["recommended"][0]["check"].startswith("확인 필요"))
        ids = [item["id"] for item in state["questions"]]
        # 프로그램 질문(비유한 값)이 있는 항목의 모델 질문은 빠지고, 남은 모델 질문은 출처가 model이다.
        self.assertNotIn("model:0", ids)
        self.assertTrue(all(item["source"] == "model" for item in state["questions"] if item["id"].startswith("model:")))
        self.assertIn("mapping.label.confirm", ids)
        confirm = next(item for item in state["questions"] if item["id"] == "mapping.label.confirm")
        self.assertEqual((confirm["level"], confirm["subjects"]), ("required", ["g"]))

        plan = json.loads(json.dumps(state["plan"]))
        plan["preparation"].update(nonfinite="median", missing_labels=None)

        with self.assertRaises(PlanError) as caught:
            orchestrator.preview_plan(task, plan, {})

        self.assertIn("mapping.label.confirm", caught.exception.subjects)
        # 사용자가 확인하면 진행할 수 있고, 양식에서 라벨을 바꾸면 확인 질문 자체가 사라진다.
        orchestrator.preview_plan(task, plan, {"mapping.label.confirm": "confirmed"})
        plan["mapping"]["label"] = "y"
        plan["preparation"]["missing_labels"] = "exclude"
        self.assertNotIn("mapping.label.confirm", [item["id"] for item in task_adapters.adapter(task["kind"]).questions(task, plan)])
        orchestrator.preview_plan(task, plan, {})
        # 재시도에서 표시한 계획(직전 확정)이 y인데 사용자가 양식에서 g를 직접 고르면 화면에 없던 필수 확인 질문으로 막지 않는다.
        retry = dict(task, confirmed={"plan": plan})
        chosen = json.loads(json.dumps(plan))
        chosen["mapping"]["label"] = "g"
        self.assertNotIn("mapping.label.confirm", [item["id"] for item in task_adapters.adapter(task["kind"]).questions(retry, chosen)])

        # 모델 지시문: 수치 전처리는 표본 배열에만, 집계 이름에 분모를 적는다.
        content = json.loads(self.requests[0])
        system, user = content["messages"][0]["content"], json.loads(content["messages"][1]["content"])
        self.assertIn("표본 배열에만 적용", system)
        self.assertIn("라벨이 아닙니다", system)
        fields = user["subjects"]["detail_fields"]

        for name in ("values_total", "nonfinite_or_missing_count", "nonfinite_or_missing_ratio_of_values",
                     "zero_count_of_finite_values", "zero_ratio_of_finite_values", "duplicate_sample_count_of_samples_n"):
            self.assertIn(name, fields)

        row = next(item for item in user["subjects"]["detailed"] if item[0] == "x")
        self.assertEqual(row[fields.index("nonfinite_or_missing_count")], 1)
        self.assertEqual(row[fields.index("values_total")], 60 * 8 * 8 * 3)

    def test_csv_and_npz_select_their_own_adapters_only(self):
        self.assertEqual(task_adapters.select(Path("a.csv")), "tabular_csv")
        self.assertEqual(task_adapters.select(Path("a.NPZ")), "array_npz")

        with self.assertRaises(ValueError):
            task_adapters.select(Path("a.pkl"))


class ArrayScreenTest(unittest.TestCase):
    """통합 작업 화면에서 npz 하나를 요청부터 결과까지 진행한다(모의 transport·모델)."""

    setUp_flow = ArrayTaskFlowTest.setUp
    transport = ArrayTaskFlowTest.transport

    def setUp(self):
        self.setUp_flow()

        for target, kwargs in (("src.agents.task_store.STORE_ROOT", {"new": self.store}),
                               ("src.ui.runtime_notice.server_permissions", {"return_value": {"elevated": True, "pid": 1, "python": "synthetic"}}),
                               ("src.agents.task_recommend.request_local_proposal", {"side_effect": model_reply})):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_screen_flow_with_previews_mapping_and_result(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"), default_timeout=180).run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any("배열 표본 파일(.npz)" in item.label for item in app.text_input))
        app.text_input(key="_widget_task_source").set_value(str(self.source))
        app.text_area(key="task_goal").set_value("y 라벨 분류용으로 배열을 분할·전처리해 주세요")
        app.checkbox(key="task_consent_a").check()
        next(item for item in app.button if item.label == "작업 시작").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any("계획 확인" in item.value for item in app.subheader))
        texts = [item.value for item in app.text]
        self.assertTrue(any("모델은 클래스 의미를 해석하지 못하며" in text for text in texts))

        def pick(label, value):
            box = next(item for item in app.selectbox if item.label == label)
            box.set_value(value)

        pick("라벨 배열", "y")
        pick("비유한 값(NaN·무한대)", "median")
        pick("라벨 없는 표본", "exclude")
        next(item for item in app.button if item.label == "최종 계획 미리보기").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        next(item for item in app.checkbox if item.label.startswith("위 최종 계획과 진행 범위에 동의")).check()
        next(item for item in app.button if item.label == "확인하고 진행").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any("필요한 결정 — 적용" in item.value for item in app.subheader), [item.value for item in app.error])
        app.checkbox(key="task_consent_d").check()
        next(item for item in app.button if item.label == "적용하고 마무리").click().run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(any(item.value == "최종 결과" for item in app.subheader))
        self.assertIn("#### 실제 전후 비교", [item.value for item in app.markdown])
        # 배열 이름·라벨 값 같은 입력 유래 문자열은 마크다운 해석 위치에 두지 않는다.
        self.assertFalse(any(label in item.value for item in app.markdown for label in LABELS))


if __name__ == "__main__":
    unittest.main()

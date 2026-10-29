"""선택 실행하는 v2 표형 작업의 실제 AppContainer·Job 검사.

`TabularTaskV2Fixture`의 6행 합성 CSV로 공통 조정기의 A→진단→계획→C→D를 진행한다. 입력 검사·생성, 진단, 분할,
내보내기, 후보, 적용 worker는 모의 없이 실제 transport로 실행한다. 로컬 모델 응답은 모의를 유지하며
(실제 모델 호출 없음) 전달 단계는 worker 없이 부모가 수행한다. 네트워크 차단 시험은 하지 않는다.
"""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents import task_orchestrator as orchestrator
from src.agents.csv_handoff_contract import load_handoff_partition
from src.agents.input_snapshot import verify_snapshot
from src.ui.runtime_notice import server_permissions
from tests.agents.test_tabular_task_v2 import PREPARATION, TabularTaskV2Fixture


# 실제 OS 실행 대상 transport. 각 worker 경계가 한 번씩 실제로 실행되어야 한다.
TRANSPORTS = {
    "input": "src.agents.tabular_input_service.run_input_worker",
    "diagnostic": "src.agents.module_pipeline.run_pipeline_worker",
    "split": "src.agents.snapshot_split.run_split_worker",
    "export": "src.agents.partition_export.run_export_worker",
    "candidate": "src.agents.preparation_candidate.run_candidate_worker",
    "apply": "src.agents.preparation_apply.run_apply_worker",
}
# 실제 실행에서는 fixture의 모의 운송·런타임 안내를 쓰지 않는다.
REAL = set(TRANSPORTS.values()) | {"src.agents.csv_workflow_service.CsvWorkflowService.runtime",
                                   "src.agents.tabular_workflow_service.TabularWorkflowService.runtime",
                                   "src.agents.tabular_input_flow.current_runtime_id"}
MINIMUM_RUNS = {"input": 2, "diagnostic": 1, "split": 4, "export": 1, "candidate": 1, "apply": 1}
LIMITS = {"wall_seconds": 120, "memory_mib": 2048, "processes": 2}


def skip_reason(environ=None, permissions=server_permissions):
    """실제 격리 실행 조건이 모두 갖춰지면 None, 아니면 생략 사유를 돌려준다."""
    environ = os.environ if environ is None else environ
    if os.name != "nt":
        return "Windows 전용 실제 격리 검사"
    if environ.get("WAFERGUARD_TEST_APPCONTAINER") != "1":
        return "실제 Windows 격리 검사 선택 실행(WAFERGUARD_TEST_APPCONTAINER=1 필요)"
    if permissions()["elevated"] is not True:
        return "실제 격리 폴더 권한 설정에는 관리자 PowerShell이 필요합니다"
    return None


def _resolve(target):
    module, _, name = target.rpartition(".")
    return getattr(__import__(module, fromlist=[name]), name)


class RealRunGateTest(unittest.TestCase):
    """일반 권한·환경변수 미설정이면 실제 실행 없이 생략한다."""

    def test_requires_switch_and_elevation(self):
        elevated, normal = (lambda: {"elevated": True}), (lambda: {"elevated": False})
        self.assertIsNone(skip_reason({"WAFERGUARD_TEST_APPCONTAINER": "1"}, elevated))
        self.assertIn("관리자", skip_reason({"WAFERGUARD_TEST_APPCONTAINER": "1"}, normal))
        self.assertIn("관리자", skip_reason({"WAFERGUARD_TEST_APPCONTAINER": "1"}, lambda: {"elevated": None}))
        for environ in ({}, {"WAFERGUARD_TEST_APPCONTAINER": "0"}, {"WAFERGUARD_TEST_APPCONTAINER": "true"}):
            with self.subTest(environ=environ):
                self.assertIn("WAFERGUARD_TEST_APPCONTAINER=1", skip_reason(environ, elevated))


REASON = skip_reason()


@unittest.skipIf(REASON is not None, REASON or "")
class TabularV2AppContainerTest(TabularTaskV2Fixture):
    def start_patch(self, target, **kwargs):
        if target in REAL:
            return None
        return super().start_patch(target, **kwargs)

    def setUp(self):
        super().setUp()
        self.outcomes, self.errors = [], []
        for stage, target in TRANSPORTS.items():
            real = _resolve(target)

            def record(request, root, cancel, real=real, stage=stage, **kwargs):
                try:
                    outcome = real(request, root, cancel, **kwargs)
                except Exception as error:
                    self.errors.append((stage, f"{type(error).__name__}: {error}"[:500]))
                    raise
                self.outcomes.append((stage, outcome))
                return outcome

            patcher = patch(target, side_effect=record)
            patcher.start()
            self.addCleanup(patcher.stop)

    def new_task(self, *, generation="current"):
        return orchestrator.start_task("label 예측용으로 분할·전처리해 주세요", self.source, LIMITS, self.model, root=self.store,
                                       input_generation=generation)

    def check_launches(self):
        """각 worker가 실제 AppContainer·Job에서 정상 종료하고 프로세스·프로필이 정리됐는지 확인한다."""
        # 한 경계가 여러 실행을 가진다: 입력은 검사·생성, 분할은 audit·run_eda·propose_split·commit_split.
        counts = {stage: sum(1 for item, _ in self.outcomes if item == stage) for stage in TRANSPORTS}
        for stage, minimum in MINIMUM_RUNS.items():
            self.assertGreaterEqual(counts[stage], minimum, (stage, counts))
        for index, (stage, outcome) in enumerate(self.outcomes):
            with self.subTest(stage=stage, run=index):
                launch = outcome.get("launch") or {}
                resources = launch.get("resources") or {}
                self.assertEqual((launch.get("status"), launch.get("exit_code")), ("exited", 0), outcome)
                self.assertEqual(resources.get("status"), "succeeded", outcome)
                self.assertEqual(resources.get("resource_messages"), [], outcome)
                self.assertTrue(resources.get("cleanup_confirmed"), outcome)
                self.assertEqual(resources.get("active_processes"), 0, outcome)
                self.assertEqual(resources.get("configured_process_limit"), LIMITS["processes"], outcome)
                self.assertEqual(resources.get("configured_job_memory_bytes"), LIMITS["memory_mib"] * 1024 ** 2, outcome)
                self.assertTrue(outcome.get("profile_cleanup"), outcome)
                self.assertNotIn("diagnostic", outcome)

    def evidence(self):
        """실패 원인 확인용: 각 실행의 종료·자원 요약과 worker가 남긴 진단 표식(단계·코드만)."""
        runs = []
        for stage, outcome in self.outcomes:
            launch = outcome.get("launch") or {}
            resources = launch.get("resources") or {}
            runs.append({"stage": stage, "status": launch.get("status"), "exit_code": launch.get("exit_code"),
                         "resources": resources.get("status"), "messages": resources.get("resource_messages"),
                         "cleanup": resources.get("cleanup_confirmed"), "profile_cleanup": outcome.get("profile_cleanup"),
                         "code": launch.get("code")})
        diagnostics = {str(path.relative_to(self.root)): path.read_text(encoding="utf-8", errors="replace")[:500]
                       for path in self.root.glob("**/workers/*/scratch/task/diagnostic.json")}
        return {"runs": runs, "errors": self.errors, "diagnostics": diagnostics}

    def test_real_isolated_run_matches_independent_train_based_calculation(self):
        try:
            self.check_flow()
        except AssertionError as error:
            raise AssertionError(f"{error}\n실행 증거: {self.evidence()}") from None

    def check_flow(self):
        source_before = self.source.read_bytes()
        task = self.diagnosed()
        plan = self.plan()
        plan["preparation"] = dict(PREPARATION, scaling="standard")
        task = orchestrator.confirm_plan(task["task_id"], plan, root=self.store)
        self.assertEqual(task["phase"], "awaiting_apply", task.get("problem"))
        task = orchestrator.approve_apply(task["task_id"], root=self.store)
        self.assertEqual(task["phase"], "completed", task.get("problem"))
        self.assertEqual(set(task["consents"]), {"A", "C", "D"})
        self.check_launches()

        # 원본 CSV와 작업 snapshot은 바뀌지 않는다.
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertTrue(verify_snapshot(self.root / "snapshot", self.snapshot))

        bundle = Path(task["report"]["facts"]["outputs"]["bundle"])
        parts = {name: load_handoff_partition(bundle, name) for name in ("train", "validation")}
        original = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        ids = {name: rows["id"].tolist() for name, (_, rows) in parts.items()}

        # 그룹 단위 분할: 구간끼리 행·그룹이 겹치지 않고 Test 행은 결과 파일에 없다.
        self.assertFalse(set(ids["train"]) & set(ids["validation"]))
        test_ids = set(original["id"]) - set(ids["train"]) - set(ids["validation"])
        self.assertTrue(test_ids)
        groups = {name: set(original.set_index("id").loc[ids[name], "g"]) for name in ids}
        test_groups = set(original.set_index("id").loc[sorted(test_ids), "g"])
        self.assertFalse(groups["train"] & groups["validation"] or (groups["train"] | groups["validation"]) & test_groups)

        # 원본에서 Train 행만으로 중앙값 대체·표준화 기준을 독립 계산하고 Validation에 같은 기준을 적용한다.
        x = pd.to_numeric(original.set_index("id")["x"].replace("", np.nan))
        train = x.loc[ids["train"]]
        median = train.median()
        filled = train.fillna(median)
        center, scale = filled.mean(), filled.std(ddof=0) or 1.0
        for name, (features, rows) in parts.items():
            with self.subTest(partition=name):
                expected = ((x.loc[ids[name]].fillna(median) - center) / scale).to_numpy()
                self.assertEqual(features.shape, (len(ids[name]), 1))
                np.testing.assert_allclose(features.iloc[:, 0].to_numpy(), expected, rtol=0, atol=1e-12)
                # 행 순서는 원본 순서를 따르고, 원래 값·자료형을 보존한다.
                order = [value for value in original["id"] if value in set(ids[name])]
                self.assertEqual(ids[name], order)
                self.assertTrue(all(type(value) is str for value in rows["id"]))
                # CSV 입력은 bool을 추론하지 않는다. 라벨은 원문 문자열 "True"/"False"로 보존된다.
                self.assertEqual(rows["label"].tolist(), original.set_index("id").loc[ids[name], "label"].tolist())
                self.assertTrue(all(type(value) is str for value in rows["label"]))
                self.assertTrue(set(rows["label"]) <= {"True", "False"})
                self.assertEqual(rows["flag"].tolist(), [int(value) for value in original.set_index("id").loc[ids[name], "flag"]])


if __name__ == "__main__":
    unittest.main()

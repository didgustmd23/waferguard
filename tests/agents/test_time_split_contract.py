"""time 분할 연결의 계약·부모 검증 단위 검사: 출처 계약 두 종류, worker 출처 packet, 분할 요약 구조 규칙, 부모의 시간 범위 재검증.

합성 값과 in-process 운송(입력·시간 복원 도구)만 쓴다. 실제 AppContainer·모델·실데이터는 쓰지 않는다.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest

import pandas as pd

from src.agents import tabular_worker_input as worker_input
from src.agents.split_evidence import read_split_files
from src.agents.tabular_source_contract import (ADAPTER_ID, SOURCE_ID_PATTERN, TabularEdaInputContract, TimeRestoredSource,
                                                source_from_dict)
from src.agents.tabular_split import TabularSplitTool
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.split import SplitRules, _digest, propose_split, time_partition_bounds
from src.preprocessing.tabular_bundle import load_bundle
from tests.agents.test_tabular_time import TimeServiceFixture


TIME_RULES = SplitRules(strategy="time", train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, require_all_classes=False,
                        timestamp_format="%Y-%m-%d")
ROLES = ColumnRoles(numeric=("x",), target="label", timestamp="t")


class TimePartitionBoundsTest(unittest.TestCase):
    def values(self):
        return pd.Series(["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24"])

    def test_bounds_match_the_ordered_partition(self):
        bounds = time_partition_bounds(self.values(), TIME_RULES, {"train": [0, 1], "validation": [2, 3], "test": [4, 5]})
        self.assertEqual(bounds["validation"], ("2026-10-21T00:00:00+00:00", "2026-10-22T00:00:00+00:00"))

    def test_same_time_across_partitions_or_reversed_order_is_rejected(self):
        same = pd.Series(["2026-10-19", "2026-10-20", "2026-10-20", "2026-10-22", "2026-10-23", "2026-10-24"])
        for values, partitions in ((same, {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}),
                                   (self.values(), {"train": [4, 5], "validation": [2, 3], "test": [0, 1]}),
                                   (self.values(), {"train": [0, 3], "validation": [1, 2], "test": [4, 5]}),
                                   (self.values(), {"train": [0, 1], "validation": [2, 3], "test": [4]})):
            with self.subTest(partitions=partitions), self.assertRaises(ValueError):
                time_partition_bounds(values, TIME_RULES, partitions)
        with self.assertRaises(ValueError):
            time_partition_bounds(self.values(), replace(TIME_RULES, strategy="random"), {"train": [0], "validation": [1], "test": [2]})

    def test_unreadable_value_error_does_not_carry_the_value(self):
        values = pd.Series(["2026-10-19", "19/10/2026-canary", "2026-10-21"])
        with self.assertRaises(ValueError) as caught:
            time_partition_bounds(values, TIME_RULES, {"train": [0], "validation": [1], "test": [2]})
        self.assertNotIn("canary", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_offset_values_order_by_real_instant(self):
        # 오프셋이 있는 값은 실제 시각 순서로 비교한다(벽시계 09:00+0900은 UTC 00:00으로 03:00+0100(UTC 02:00)보다 이르다).
        rules = replace(TIME_RULES, timestamp_format="%Y-%m-%d %H:%M%z")
        values = pd.Series(["2026-10-19 09:00+0900", "2026-10-19 03:00+0100", "2026-10-19 05:00+0100"])
        bounds = time_partition_bounds(values, rules, {"train": [0], "validation": [1], "test": [2]})
        self.assertEqual(bounds["train"][0], "2026-10-19T00:00:00+00:00")
        with self.assertRaises(ValueError):
            time_partition_bounds(values, rules, {"train": [1], "validation": [0], "test": [2]})

    def test_partition_failure_raises_without_fallback(self):
        # 서로 다른 시각이 두 개뿐이면 세 구간을 만들 수 없다. 다른 방식으로 대체하지 않고 실패한다.
        frame = pd.DataFrame({"x": [1, 2, 3, 4], "label": ["a", "b", "a", "b"],
                              "t": ["2026-10-19", "2026-10-19", "2026-10-20", "2026-10-20"]})
        with self.assertRaises(ValueError):
            propose_split(TabularInput(frame, ROLES), TIME_RULES, source_id="s", config_id="c")


class SplitSummaryRuleTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, proposal):
        folder = self.root / str(len(list(self.root.iterdir())))
        folder.mkdir()
        (folder / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
        (folder / "proposal.md").write_text("요약", encoding="utf-8")
        return folder

    def proposal(self, rules):
        frame = pd.DataFrame({"x": range(6), "label": ["a", "b"] * 3,
                              "t": ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24"]})
        roles = ROLES if rules.strategy == "time" else ColumnRoles(numeric=("x",), target="label")
        return propose_split(TabularInput(frame, roles), rules, source_id="s", config_id="c")

    def resealed(self, proposal, change):
        changed = json.loads(json.dumps(proposal))
        change(changed)
        changed.pop("proposal_id")
        changed["proposal_id"] = _digest(changed)
        return changed

    def test_time_summary_requires_ranges_and_other_strategies_forbid_them(self):
        time = self.proposal(TIME_RULES)
        self.assertEqual(read_split_files(self.write(time), "s", "c", preview=True)["proposal_id"], time["proposal_id"])
        missing = self.resealed(time, lambda value: value["summary"][1].pop("time_max"))
        with self.assertRaises(ValueError):
            read_split_files(self.write(missing), "s", "c", preview=True)
        typed = self.resealed(time, lambda value: value["summary"][0].update(time_min=1))
        with self.assertRaises(ValueError):
            read_split_files(self.write(typed), "s", "c", preview=True)
        random = self.proposal(SplitRules(require_all_classes=False))
        self.assertTrue(read_split_files(self.write(random), "s", "c", preview=True))
        extra = self.resealed(random, lambda value: value["summary"][0].update(time_min="2026-10-19T00:00:00+00:00"))
        with self.assertRaises(ValueError):
            read_split_files(self.write(extra), "s", "c", preview=True)


class DerivedSourceContractTest(TimeServiceFixture):
    """파생 표 출처의 계약 복원·worker packet·부모 시간 범위 재검증."""

    def derived(self):
        source, folder = self.source()
        self.run_preview(source, folder)
        service, state = self.run_apply(source, folder)
        self.assertEqual(state["status"], "completed")
        return source, service.derived_source("a1")

    def test_contract_interface_worker_packet_and_parent_time_recheck(self):
        original, derived = self.derived()
        # 출처 계약: schema_version으로만 복원, 공통 인터페이스
        self.assertEqual(source_from_dict(json.loads(json.dumps(derived.to_dict()))), derived)
        self.assertEqual(source_from_dict(original.to_dict()), original)
        with self.assertRaises(ValueError):
            source_from_dict(dict(derived.to_dict(), schema_version="tabular_time_source_v0"))
        self.assertEqual((derived.kind, derived.origin, derived.commit), ("time_restored", original, derived.application))
        self.assertEqual(derived.protected_directories, (Path(derived.bundle_directory), *original.protected_directories))
        self.assertEqual((original.kind, original.origin, original.run_directory),
                         ("generated", original, original.materialization.run_directory))
        profile = DatasetProfile("p", ADAPTER_ID, "split_review_v1", ColumnRoles(numeric=("x",), target="label", timestamp="t"),
                                 split_rules=TIME_RULES)
        contract = TabularEdaInputContract(derived, json.dumps(profile.to_dict(), ensure_ascii=False), derived.limits)
        self.assertEqual(TabularEdaInputContract.from_dict(json.loads(json.dumps(contract.to_dict()))), contract)
        self.assertTrue(contract.check_input())
        # 하위 구간·전달 계약의 부모 출처 형식: 원래·파생 출처만 허용
        import re
        self.assertTrue(re.fullmatch(SOURCE_ID_PATTERN, derived.source_id) and re.fullmatch(SOURCE_ID_PATTERN, original.source_id))
        self.assertIsNone(re.fullmatch(SOURCE_ID_PATTERN, "tabular_partition:" + "0" * 64))
        self.assertEqual(SOURCE_ID_PATTERN, worker_input.SOURCE_ID_PATTERN)

        # worker 쪽 출처 packet: 파생 packet의 ID·키·계보
        packet = derived.packet()
        self.assertEqual(worker_input.source_id(packet), derived.source_id)
        for index, forged in enumerate((dict(packet, extra=1), dict(packet, restored_count=0), dict(packet, rules_signature="0" * 64),
                       dict(packet, parent_source_id="tabular_time_source:" + "0" * 64), dict(packet, policy_version="x"))):
            with self.subTest(case=index), self.assertRaises(ValueError):
                worker_input.validate_source(forged)
        staged = self.root / "staged" / "w"
        (staged / "input").mkdir(parents=True)
        shutil.copytree(derived.bundle_directory, staged / "input/table")
        self.assertEqual(worker_input.check_staged(staged, packet), derived.bundle)
        with self.assertRaises(ValueError):
            worker_input.check_staged(staged, dict(packet, preview_sha256="0" * 64))

        # 부모의 time 분할 결과 재검증: 출처 묶음의 시간 컬럼으로 구간 범위·순서를 다시 계산해 요약과 대조한다.
        frame = load_bundle(derived.bundle_directory, expected=derived.bundle)
        report = propose_split(TabularInput(frame, profile.roles), TIME_RULES, source_id=derived.source_id, config_id="c")
        tool = SimpleNamespace(contract=SimpleNamespace(source=derived, profile=profile))
        TabularSplitTool._check_time(tool, report, TIME_RULES)
        forged = json.loads(json.dumps(report))
        forged["summary"][0]["time_min"] = "1999-01-01T00:00:00+00:00"
        with self.assertRaises(ValueError):
            TabularSplitTool._check_time(tool, forged, TIME_RULES)
        swapped = json.loads(json.dumps(report))
        swapped["partitions"] = {"train": report["partitions"]["test"], "validation": report["partitions"]["validation"],
                                 "test": report["partitions"]["train"]}
        with self.assertRaises(ValueError):
            TabularSplitTool._check_time(tool, swapped, TIME_RULES)
        with self.assertRaises(ValueError):
            TabularSplitTool._check_time(tool, dict(report, rules=asdict(replace(TIME_RULES, timestamp_format="%d.%m.%Y"))),
                                         TIME_RULES)
        self.assertIs(type(derived), TimeRestoredSource)


if __name__ == "__main__":
    unittest.main()

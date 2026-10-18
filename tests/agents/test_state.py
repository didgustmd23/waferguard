"""실제 데이터 없이 에이전트 상태·감사 기록 경계를 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest

from src.agents.audit import append_event
from src.agents.contracts import ToolResult
from src.agents.state import LocalStateStore


# ==========================================
# 로컬 상태·감사 기록의 합성 테스트
# - 실제 canonical·캐시·원본 pickle을 열지 않는다.
# - 상태 변조와 상세 객체 주입이 차단되는지를 확인한다.
# ==========================================
class AgentStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "agent"

    def test_create_load_and_revision(self):
        """상태는 원자적으로 저장되고 오래된 revision 저장을 거부한다."""
        store = LocalStateStore(self.root)
        state = store.create("local_operator")
        self.assertEqual(state.revision, 1)
        loaded = store.load()
        self.assertEqual(loaded.run_id, state.run_id)
        loaded.status = "running"
        store.save(loaded, expected_revision=1)
        with self.assertRaisesRegex(ValueError, "오래된 상태 revision"):
            store.save(loaded, expected_revision=1)

    def test_rejects_tampered_state_and_detail_objects(self):
        """알 수 없는 상태 필드·비열거 상태·상세 지표 객체를 거부한다."""
        store = LocalStateStore(self.root)
        store.create("local_operator")
        raw = json.loads(store.path.read_text(encoding="utf-8"))
        raw["unexpected_detail"] = "CANARY"
        store.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "상태 파일 구조"):
            store.load()
        raw.pop("unexpected_detail")
        raw["artifact_refs"] = "상세 경로를 문자열로 주입"
        store.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "산출물 참조"):
            store.load()
        with self.assertRaises(ValueError):
            ToolResult(status="passed", metrics={"detail": 1.5})

    def test_audit_event_has_no_free_text_field(self):
        """감사 이벤트는 고정 필드만 저장하고 원문 입력을 받지 않는다."""
        self.root.mkdir()
        path = self.root / "events.jsonl"
        append_event(path, "action_denied", "ACTION_NOT_ALLOWED", stage="audit", revision=2)
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(record), {"event_type", "code", "stage", "revision", "recorded_at"})
        with self.assertRaises(ValueError):
            append_event(path, "unknown", "CODE", stage="audit", revision=2)


if __name__ == "__main__":
    unittest.main()

"""공통 UI→실제 WM 생성/승인/검증 통합. 합성 데이터·모의 OS 운송만 사용한다."""

from importlib.util import find_spec
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.agents.wm_module_plan import load_wm_modules
from tests.agents import test_wm_workflow_service as fixtures


@unittest.skipUnless(find_spec('streamlit'), 'Streamlit 미설치')
class UiWmIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WmWorkflowServiceTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.project = Path(__file__).resolve().parents[2]
        self.plan = self.project / 'configs/agents/pipelines/wm_development_example.json'
        self.bindings = self.fixture.root / 'empty_bindings.json'
        self.bindings.write_text(json.dumps({'schema_version': 'wm_run_bindings_v1',
            'pipeline_signature': load_wm_modules(self.plan).signature, 'runs': {}}), encoding='utf-8')

        def directory(name, **kwargs):
            return self.fixture.root / 'runs' / name

        def creation_root(*, create=False):
            root = self.fixture.root / 'creations'
            if create:
                root.mkdir(exist_ok=True)
            return root

        for target, replacement in (('src.agents.wm_canonical_cli._directory', directory),
                                     ('src.agents.wm_run_creation.creation_root', creation_root)):
            mock = patch(target, side_effect=replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def run_ui(self, *, review=False):
        from streamlit.testing.v1 import AppTest
        from src.agents.wm_canonical_cli import create_demo, open_run

        source = self.fixture.tool.contract.snapshot_directory
        if review:
            create_demo('review-source', self.fixture.tool.limits, review=True)
            source = open_run('review-source').contract.snapshot_directory

        app = AppTest.from_file(str(self.project / 'src/ui/app.py'), default_timeout=60).run()
        app.selectbox(key='ui_workflow_mode').select('등록 모듈').run()
        app.selectbox(key='workflow_module_select').select('wm_canonical_eda')
        app.button(key='FormSubmitter:workflow_select-이 모듈 선택').click().run()
        app.button(key='ui_step_2').click().run()
        app.button(key='workflow_review').click().run()
        app.button(key='ui_step_3').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['ui_screen'], '3 입력 선택')
        self.assertIn('_widget_workflow_wm_canonical_eda_pipeline', [t.key for t in app.text_input],
                      [(t.label, t.key) for t in app.text_input])

        prefix = 'workflow_wm_canonical_eda_'
        app.text_input(key='_widget_' + prefix + 'pipeline').input(str(self.plan))
        app.text_input(key='_widget_' + prefix + 'bindings').input(str(self.bindings))
        app.text_input(key='_widget_' + prefix + 'canonical_dir').input(str(source))
        app.text_input(key=prefix + 'name').input('ui-native-core')
        app.button(key='FormSubmitter:workflow_creation_wm_canonical_eda-생성 범위 검사·미리보기').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error, [e.value for e in app.error])
        next(c for c in app.checkbox if c.label == '위 설정으로 새 작업과 배정 기록만 생성합니다').check()
        app.button(key='FormSubmitter:workflow_create_confirm-새 작업 준비').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error, [e.value for e in app.error])

        app.button(key='ui_step_4').click().run()
        app.button(key='workflow_refresh').click().run()
        app.button(key='workflow_next').click().run()
        app.button(key='workflow_next').click().run()
        app.button(key='workflow_preview').click().run()
        app.checkbox[0].check()
        app.button(key='FormSubmitter:workflow_approve-이번 단계 승인').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error, [e.value for e in app.error])

        # 생성·승인까지 OS worker를 호출하지 않는다. 실행 제출에서 한 번만 호출한다.
        with patch('src.agents.wm_canonical.run_wm_worker', side_effect=self.fixture.transport) as worker:
            app.run()
            worker.assert_not_called()
            app.checkbox[0].check()
            app.button(key='FormSubmitter:workflow_execute-승인 단계 실행').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error, [e.value for e in app.error])
            self.assertEqual(worker.call_count, 1)
            expected = 'blocked_input' if review else 'completed'
            self.assertEqual(app.session_state['ui_workflow'].state['status'], expected)
            app.run()
            self.assertEqual(worker.call_count, 1)

        app.button(key='ui_step_5').click().run()
        app.button(key='workflow_results').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error, [e.value for e in app.error])
        loaded = app.session_state['ui_workflow_results'][2]
        self.assertEqual(loaded['status'], expected)
        return loaded

    def test_creation_to_approval_execution_and_local_results(self):
        loaded = self.run_ui()
        suffixes = {item['suffix'] for item in loaded['files']}
        self.assertTrue({'.png', '.csv', '.md'} <= suffixes)

    def test_quality_review_stops_analysis_and_preserves_local_report(self):
        loaded = self.run_ui(review=True)
        self.assertFalse(any(item['suffix'] == '.png' for item in loaded['files']))
        self.assertTrue(any(item['suffix'] == '.md' for item in loaded['files']))


if __name__ == '__main__':
    unittest.main()

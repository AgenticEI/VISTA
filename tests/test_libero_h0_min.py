"""Package behavior and snapshot boundary, without simulator or paid API calls."""
import json
from pathlib import Path
import shutil

from rpent.harness import CapabilityBinding, CapabilitySpec, bind_harness, load_harness
from rpent.tools.toolkit import ToolResult

ROOT = Path(__file__).resolve().parents[1] / 'harnesses/libero_pro_h0_min'

class Toolkit:
    def get_tools_spec(self):
        return [dict(name=n, description=n, input_schema={'type': 'object'})
                for n in ('finish', 'read_text_file', 'write_text_file', 'list_dir')]
    def execute_tool(self, name, args):
        raise AssertionError('Robot behavior must not use native tools')


def test_package_snapshot_calls_raw_capabilities_and_preserves_images(tmp_path):
    root = tmp_path / 'package'
    shutil.copytree(ROOT, root)
    original = load_harness(root)
    path = root / 'skills/vla_step/tool.py'
    path.write_text(path.read_text().replace('{"instruction": instruction}', '{"instruction": "changed: " + instruction}'))
    changed = load_harness(root)
    calls = []
    def invoke(args):
        calls.append(args)
        return {'_image_cam_bytes': b'image-evidence', 'episode_active': True}
    caps = {c['id']: CapabilityBinding(CapabilitySpec(**c), invoke)
            for c in json.loads((ROOT/'development/capabilities.json').read_text())['capabilities']}
    for package, expected in ((original, 'task'), (changed, 'changed: task')):
        bound = bind_harness(package, toolkit=Toolkit(), capabilities=caps, episode_dir=tmp_path/expected.replace(':',''))
        try:
            assert {s['name'] for s in bound.toolkit.get_tools_spec()} == {
                'finish', 'read_text_file', 'write_text_file', 'list_dir', 'observe', 'vla_step'}
            assert len(bound.policy_runtime.capability_specs) == 10
            result = bound.toolkit.execute_tool('vla_step', {'instruction': 'task'})
            assert calls[-1] == {'instruction': expected}
            assert result.result['_image_cam_bytes'] == b'image-evidence'
            assert any(b.get('type') == 'image' for b in result.content_blocks)
        finally:
            bound.close()

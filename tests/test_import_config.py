"""Validate installed Isaac config schemas without loading Kit or a GPU."""
import ast
import sys
import types
from pathlib import Path
from widowx_codex.common import ROOT
from widowx_codex.import_config import robot_import_config


def test_installed_urdf_gain_schema():
    import pytest
    source = ROOT.parent/'IsaacLab/source/isaaclab/isaaclab'
    if not source.exists():
        pytest.skip('Local Isaac Lab source unavailable')
    module = types.ModuleType('_isolated_isaac_config_schema')
    sys.modules[module.__name__] = module
    try:
        # Execute real configclass and schema definitions, without importing
        # isaaclab.sim (which needs AppLauncher). Dict serialization helpers
        # are irrelevant here; this test invokes only constructors/validate.
        for path in (
            source/'utils/configclass.py',
            source/'sim/converters/asset_converter_base_cfg.py',
            source/'sim/converters/urdf_converter_cfg.py',
        ):
            tree = ast.parse(path.read_text(), filename=str(path))
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.ImportFrom)
                and (node.level or (node.module or '').startswith('isaaclab'))
            )]
            module.__dict__.setdefault('class_to_dict', None)
            module.__dict__.setdefault('update_class_from_dict', None)
            exec(compile(tree, str(path), 'exec'), module.__dict__)
        cls = module.UrdfConverterCfg
        with pytest.raises(TypeError, match='stiffness'):
            cls(asset_path='robot.urdf', fix_base=True).validate()
        config = robot_import_config(cls, ROOT/'assets/wx250s.import.urdf', '/tmp/robot_usd_test')
        config.validate()
        assert config.joint_drive.gains.stiffness == 0.0
        assert config.fix_base and not config.merge_fixed_joints
        assert not config.convert_mimic_joints_to_normal_joints
    finally:
        sys.modules.pop(module.__name__, None)


def test_imported_fingers_keep_original_travel_without_passive_mimic():
    import xml.etree.ElementTree as ET
    original = ET.parse(ROOT/'assets/wx250s.original.urdf')
    imported = ET.parse(ROOT/'assets/wx250s.import.urdf')
    assert original.findall('.//mimic')
    assert not imported.findall('.//mimic')
    for name in ('left_finger', 'right_finger'):
        path = f".//joint[@name='{name}']/limit"
        assert original.find(path).attrib == imported.find(path).attrib

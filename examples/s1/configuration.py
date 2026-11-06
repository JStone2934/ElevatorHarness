"""Explicit deployment calibration; no measured site data ships with the example."""
import hashlib
import json
from pathlib import Path
from elevator_harness import Capabilities, OperationFrame, ToolConfiguration, PressConfig, PerceptionConfig


def configuration_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def capabilities(settings):
    return Capabilities(OperationFrame(**settings['frame']), ('global', 'local'),
        tuple(ToolConfiguration(**row) for row in settings['tools']), ('global',), True,
        manipulator='astribot_arm_right', alignment_guidance=settings.get('alignment_guidance', ''))


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    caps = capabilities(settings)
    if caps.frame.reference != 'chassis':
        raise ValueError('S1 executor expects operation axes expressed in chassis coordinates')
    if len(caps.tools) != 2 or set(settings['gripper_keys']) != {t.name for t in caps.tools} or set(settings['gripper_keys'].values()) != {'o', 'c'}:
        raise ValueError('S1 needs exactly two configurations mapped to o/c')
    if len({t.calibration_version for t in caps.tools}) != 1:
        raise ValueError('S1 tools must share one calibration revision')
    point = settings.get('depth_point')
    if not isinstance(point, list) or len(point) != 2 or any(type(v) not in (float, int) or not 0 <= v <= 1 for v in point):
        raise ValueError('fill in calibrated normalized depth_point before starting hardware')
    if settings['strategy'] not in ('localized', 'end_to_end'):
        raise ValueError('unknown strategy')
    if settings['strategy'] == 'localized':
        perception = PerceptionConfig(**settings.get('perception', {}))
        if (perception.local_camera, perception.global_camera) != ('local', 'global'):
            raise ValueError('S1 maps camera roles to local/global')
        cfg = PressConfig(**settings['press'])
        if perception.press_tool is not None and perception.press_tool != cfg.press_tool:
            raise ValueError('perception and press policy must use the same press tool')
        if (cfg.open_tool not in settings['gripper_keys'] or cfg.press_tool not in settings['gripper_keys'] or
                settings['gripper_keys'][cfg.open_tool] != 'o' or settings['gripper_keys'][cfg.press_tool] != 'c'):
            raise ValueError('press tools must map to open/closed gripper commands')
        if caps.tool(cfg.press_tool).local_contact_uv is None or list(caps.tool(cfg.press_tool).local_contact_uv) != point:
            raise ValueError('closed contact pixel must match the depth measurement point')
    return settings

"""Composable checks over evidence. Never infer or silently clip robot commands."""
from dataclasses import dataclass, asdict
import math

from .types import Decision


class CompositePolicy:
    def __init__(self, *policies):
        self.policies = policies

    def prepare(self, context):
        for policy in self.policies:
            if hasattr(policy, 'prepare'):
                policy.prepare(context)

    def check(self, action, context):
        for policy in self.policies:
            decision = policy.check(action, context)
            if not decision.allowed:
                return decision
        return Decision(True)

    def record(self, action, before, result):
        for policy in self.policies:
            policy.record(action, before, result)

    def describe(self):
        return {type(p).__name__: p.describe() for p in self.policies}


def contained(context, point=None):
    evidence = context.evidence
    if evidence is None or not evidence.matches(context.observation):
        return False
    values = evidence.values
    box = values.get('target_box')
    point = point if point is not None else values.get('contact_uv')
    return bool(values.get('fully_visible') is True and not values.get('views_conflict') and
                box and point and all(box[i] <= point[i] <= box[i+2] for i in (0, 1)))


class OverlapPolicy:
    def check(self, action, context):
        if action.kind == 'move' and action.arguments['delta'][2] > 0 and not contained(context):
            return Decision(False, 'current local contact point must be inside the target face in both axes')
        return Decision(True)

    def record(self, action, before, result):
        pass

    def describe(self):
        return {'rule': '2D overlap only; not proof of 3D contact'}


@dataclass(frozen=True)
class PressConfig:
    open_tool: str
    press_tool: str
    contact_depth_m: float
    max_travel_m: float
    standoff_m: float = .010
    overpress_m: float = 0.
    withdraw_m: float = .010
    max_attempts: int = 3
    range_min_m: float = .001
    range_max_m: float = 1.
    require_path_review: bool = True
    final_normal_only: bool = True

    def __post_init__(self):
        for key in ('contact_depth_m', 'max_travel_m', 'standoff_m', 'overpress_m',
                    'withdraw_m', 'range_min_m', 'range_max_m'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid press distance: ' + key)
        if (self.contact_depth_m <= self.overpress_m or self.max_travel_m <= 0 or self.withdraw_m <= 0 or
                self.range_min_m >= self.range_max_m or self.max_attempts < 1 or self.open_tool == self.press_tool):
            raise ValueError('invalid press configuration')


class CalibratedPressPolicy:
    """Open-view range may cross into ONE explicitly configured closed-tool geometry.

    Frozen limits survive missing live depth, never calibration changes. Measured
    normal displacement is preferred; commanded displacement is the fallback.
    """
    def __init__(self, config: PressConfig):
        self.config = config
        self.frozen = None
        self.pending = None
        self.commanded = 0.
        self.approached = False
        self.pressed = False
        self.withdrawn = 0.
        self.attempts = 0
        self.last_allowance = None

    def _normal(self, obs, capabilities):
        position = obs.state.get('position_m')
        return capabilities.frame.normal_position(position) if position is not None else None

    def _remaining(self, context):
        current = self._normal(context.observation, context.capabilities)
        start = self.frozen['normal_position_m']
        used = current-start if current is not None and start is not None else self.commanded
        target = self.config.contact_depth_m-self.config.overpress_m
        remaining = min(self.config.max_travel_m, self.frozen['depth_m']-target)-used
        obs = context.observation
        live = obs.depth_m if obs.depth_valid else None
        if live is not None and self.config.range_min_m <= live <= self.config.range_max_m:
            remaining = min(remaining, live-target)
        source = 'measured' if current is not None and start is not None else 'commanded'
        self.last_allowance = {'observation_id': obs.id, 'remaining_m': max(0., remaining),
                               'used_m': used, 'travel_source': source,
                               'live_depth_used': live is not None and self.config.range_min_m <= live <= self.config.range_max_m}
        return max(0., remaining), used, source

    def prepare(self, context):
        obs = context.observation
        if self.frozen and obs.calibration_version != self.frozen['calibration_version']:
            self.frozen = None
            self.pending = None
            self.last_allowance = None
        if self.frozen and obs.tool == self.config.press_tool:
            self._remaining(context)

    def check(self, action, context):
        obs, cfg = context.observation, self.config
        if action.kind in ('unable', 'observe', 'adjust_view', 'done'):
            return Decision(True)
        if self.frozen and obs.calibration_version != self.frozen['calibration_version']:
            self.frozen = None
            self.pending = None
        if action.kind == 'configure_tool':
            target = action.arguments['configuration']
            if target == cfg.press_tool and obs.tool == cfg.open_tool:
                if self.attempts >= cfg.max_attempts:
                    return Decision(False, 'press attempt budget exhausted')
                try:
                    point = context.capabilities.tool(cfg.press_tool).local_contact_uv
                except StopIteration:
                    return Decision(False, 'press tool not available')
                if not contained(context, point):
                    return Decision(False, 'align target with calibrated merged contact pixel before closing')
                if not obs.depth_valid or obs.depth_m is None or not cfg.range_min_m <= obs.depth_m <= cfg.range_max_m:
                    return Decision(False, 'trusted in-range open-tool depth required before closing')
                target_depth = cfg.contact_depth_m-cfg.overpress_m
                if obs.depth_m <= target_depth:
                    return Decision(False, 'withdraw before freezing a new approach range')
                self.pending = {'observation_id': obs.id, 'depth_m': obs.depth_m,
                    'normal_position_m': self._normal(obs, context.capabilities),
                    'normal_axis': list(context.capabilities.frame.n),
                    'calibration_version': obs.calibration_version,
                    'from_tool': obs.tool, 'to_tool': cfg.press_tool,
                    'source': 'measured_depth_with_model_alignment'}
            elif target == cfg.open_tool and self.pressed and self.withdrawn + 1e-9 < cfg.withdraw_m:
                return Decision(False, 'withdraw before reopening after a press')
            return Decision(True)
        if action.kind != 'move':
            return Decision(True)
        du, dv, dn = action.arguments['delta']
        if dn <= 0:
            return Decision(True)
        if obs.tool == cfg.open_tool:
            return Decision(contained(context), 'current local overlap required for open-tool approach')
        if obs.tool != cfg.press_tool or self.frozen is None:
            return Decision(False, 'press needs a range frozen from the configured open-tool view')
        if self.pressed:
            return Decision(False, 'press completed: withdraw, inspect indicator, reopen and realign if needed')
        remaining, used, source = self._remaining(context)
        limit = remaining if self.approached else max(0., remaining-cfg.standoff_m-cfg.overpress_m)
        # If already at the pre-contact distance, a fresh path review can authorize final pressing.
        final_phase = self.approached or limit <= 1e-6
        if final_phase and cfg.final_normal_only and (du != 0 or dv != 0):
            return Decision(False, 'final press must be normal-only; align in the plane separately')
        if final_phase and cfg.require_path_review:
            evidence = context.evidence
            if (evidence is None or not evidence.matches(obs) or
                    evidence.values.get('path_clear') is not True or evidence.values.get('views_conflict')):
                return Decision(False, 'fresh global/local path review required before final pressing')
            limit = remaining
        if dn > limit + 1e-9:
            return Decision(False, f'normal step exceeds {limit:.6f} m remaining ({source}); candidate not clipped')
        return Decision(True)

    def record(self, action, before, result):
        if action.kind == 'configure_tool':
            if action.arguments['configuration'] == self.config.press_tool and before.tool == self.config.open_tool:
                if self.pending is None or self.pending['observation_id'] != before.id:
                    raise RuntimeError('missing admitted frozen range')
                self.frozen, self.pending = self.pending, None
                self.commanded, self.withdrawn = 0., 0.
                self.approached, self.pressed = False, False
            elif action.arguments['configuration'] == self.config.open_tool:
                self.frozen, self.pending = None, None
                self.approached, self.pressed = False, False
        elif action.kind == 'move' and self.frozen:
            dn = action.arguments['delta'][2]
            self.commanded += dn
            if dn < 0:
                self.withdrawn += -dn
            if dn > 0 and before.tool == self.config.press_tool:
                obs = result.observation
                measured = (obs.state if obs else result.measured_state).get('position_m')
                start = self.frozen['normal_position_m']
                used = (sum(a*b for a, b in zip(measured, self.frozen['normal_axis'])) - start
                        if measured is not None and start is not None else self.commanded)
                target = self.config.contact_depth_m-self.config.overpress_m
                budget = min(self.config.max_travel_m, self.frozen['depth_m']-target)
                live_reached = (obs is not None and obs.depth_valid and obs.depth_m is not None and
                                self.config.range_min_m <= obs.depth_m <= target + 1e-6)
                if live_reached or used >= budget-1e-6:
                    if not self.pressed:
                        self.attempts += 1
                    self.pressed = True
                self.approached = True

    def describe(self):
        return {'configuration': asdict(self.config), 'frozen_range': self.frozen,
                'commanded_normal_displacement_m': self.commanded,
                'approached': self.approached, 'pressed': self.pressed,
                'withdrawn_m': self.withdrawn, 'completed_attempts': self.attempts,
                'last_allowance': self.last_allowance}

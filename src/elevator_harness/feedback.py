"""Actual-command feedback and bounded press-attempt memory, never contact truth."""
import copy
import math


def geometry_error(evidence):
    values = (evidence or {}).get('values', {})
    box, point = values.get('target_box'), values.get('contact_uv')
    if not values.get('fully_visible') or box is None or point is None:
        return None
    size = [box[2]-box[0], box[3]-box[1]]
    return {'offset': [(box[i]+box[i+2])/2-point[i] for i in (0,1)], 'size': size,
            'identity': ((evidence or {}).get('tool'), (evidence or {}).get('calibration_version'))}


def compare_geometry(before, after):
    a, b = geometry_error(before), geometry_error(after)
    result = {'effect': 'unknown', 'reason': 'missing_or_partial_geometry'}
    if a is None or b is None or a['identity'] != b['identity']:
        return result
    if max(max(x/y,y/x) for x,y in zip(a['size'],b['size'])) > 1.2:
        return dict(result, reason='box_scale_changed')
    old, new = math.hypot(*a['offset']), math.hypot(*b['offset'])
    margin = max(.001, .1*old)
    return {'effect': 'improved' if old-new>=margin else 'worsened' if new-old>=margin else 'unchanged',
            'reason': 'comparable_normalized_offsets_not_metric_error', 'before': a, 'after': b}


class PressMemory:
    def __init__(self, limit=3):
        self.limit = limit
        self.attempts = []
        self.last_adjustment = None
        self.seen = set()

    def record(self, transition, press_tool):
        action = transition['action']
        cid = action['command_id']
        if cid in self.seen:
            return
        self.seen.add(cid)
        kind, args = action['kind'], action['arguments']
        if kind == 'observe':
            return
        self.last_adjustment = None
        if kind == 'configure_tool' and transition['tool_after'] == press_tool and transition['tool_before'] != press_tool:
            self.attempts.append({'close_before_id': transition['before_id'], 'close_after_id': transition['after_id'],
                'close_evidence': copy.deepcopy(transition.get('evidence')), 'cumulative_uv_m': [0.,0.],
                'last_press_id': None, 'deepest_press_id': None, 'net_normal_m': 0., 'deepest_normal_m': 0.,
                'withdrawn': False, 'reopened': False, 'last_alignment_claim': None})
            self.attempts = self.attempts[-self.limit:]
        elif self.attempts and not self.attempts[-1]['reopened']:
            entry = self.attempts[-1]
            if kind == 'configure_tool' and transition['tool_after'] != press_tool:
                entry.update(reopened=True, reopen_id=transition['after_id'], reopen_reason_model=action['reason'])
            elif kind == 'move' and transition['tool_before'] == press_tool:
                u,v,n = args['delta']
                entry['net_normal_m'] += n
                if n < 0:
                    entry['withdrawn'] = True
                if not entry['withdrawn']:
                    entry['cumulative_uv_m'] = [a+b for a,b in zip(entry['cumulative_uv_m'],(u,v))]
                    entry['last_alignment_claim'] = copy.deepcopy(action.get('assessment'))
                if n > 0:
                    entry['last_press_id'] = transition['after_id']
                    if entry['net_normal_m'] >= entry['deepest_normal_m']:
                        entry['deepest_normal_m'] = entry['net_normal_m']
                        entry['deepest_press_id'] = transition['after_id']
        if (kind == 'move' and any(args['delta'][:2]) and
                transition.get('decision_context',{}).get('stage') == 'direct_press'):
            before = transition.get('before_state',{}).get('position_m')
            after = transition.get('after_state',{}).get('position_m')
            self.last_adjustment = {'before_id': transition['before_id'], 'after_id': transition['after_id'],
                'command_id': cid, 'delta_uvn_m': args['delta'], 'previous_reason_model': action['reason'],
                'measured_delta_robot_m': [b-a for a,b in zip(before,after)] if before and after else None,
                'settle': transition.get('after_state',{}).get('settle'),
                'instruction': 'Judge this actual correction against before/current images before another correction; do not reuse an old alignment claim.'}

    def describe(self):
        return {'attempts': copy.deepcopy(self.attempts), 'last_adjustment': copy.deepcopy(self.last_adjustment),
                'source': 'executed_command_history_not_contact_or_failure_truth',
                'instruction': 'Compare prior press positions and accumulated u/v corrections before repeating an unsuccessful hypothesis. A reopened gripper or depth limit is not proof of failure.'}


def motion_review_signal(transitions, observations, current, direct_press, repeat_count):
    moves, seen = [], set()
    for transition in reversed(transitions):
        action = transition['action']
        if action['command_id'] in seen:
            continue
        seen.add(action['command_id'])
        if action['kind'] == 'observe':
            continue
        if action['kind'] != 'move':
            break
        moves.append(transition)
    windows = []
    for axis in ((0,1) if direct_press else (0,1,2)):
        streak, sign = [], None
        for t in moves:
            value = t['action']['arguments']['delta'][axis]
            if value == 0 or sign is not None and value*sign <= 0:
                break
            sign = 1 if value > 0 else -1
            streak.append(t)
        if len(streak) >= repeat_count:
            selected = list(reversed(streak[:repeat_count]))
            baseline = selected[0]['before_id']
            if baseline not in observations:
                continue
            comparison = compare_geometry(selected[0].get('evidence'), current)
            if direct_press or axis == 2 and sign > 0 or comparison['effect'] != 'improved':
                windows.append({'axis': ('u','v','n')[axis], 'sign': sign,
                    'baseline_id': baseline, 'last_command_id': selected[-1]['action']['command_id'],
                    'commands': selected, 'projection_comparison': comparison})
    return windows

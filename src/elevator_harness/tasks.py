"""Task progress and completion evidence are separate from Agent declarations."""
from .agents import inspect
from .types import Verification


class PressButtonTask:
    def __init__(self, goal):
        if not goal:
            raise ValueError('task target is required')
        self.goal = goal
        self.stage = 'observe'
        self.commands = 0
        self.configuration = None

    def observe(self, observation):
        if self.configuration != observation.tool:
            self.stage = 'observe'
        self.configuration = observation.tool

    def update(self, action, result):
        self.commands += 1
        if action.kind == 'move':
            u, v, n = action.arguments['delta']
            self.stage = 'withdraw' if n < 0 else 'approach_or_press' if n > 0 else 'align'
        elif action.kind == 'configure_tool':
            self.stage = 'observe'
        elif action.kind == 'adjust_view':
            self.stage = 'acquire_view'

    def describe(self):
        return {'stage': self.stage, 'sent_commands': self.commands, 'tool': self.configuration,
                'completion': 'initial light off, this-run activation, later light on with hand clear'}


def verify_light_sequence(rows, source):
    if len(rows) < 3 or rows[0][1].get('light_on') is not False:
        return Verification('unknown', source, reason='initial off-state or temporal evidence missing')
    activated = None
    for oid, values in rows[1:]:
        if activated and values.get('light_on') is True and values.get('hand_clear') is True:
            return Verification('confirmed', source, (rows[0][0], activated, oid),
                                'observed activation followed by lit target with hand clear')
        if values.get('light_on') is True and activated is None:
            activated = oid
    if rows[-1][1].get('light_on') is False:
        return Verification('denied', source, (rows[-1][0],), 'target is not lit in final observation')
    return Verification('unknown', source, reason='activation and subsequent withdrawal not established')


class VisualVerifier:
    def __init__(self, model):
        self.model = model

    def verify(self, observations, goal, cancel):
        rows = []
        for obs in observations:
            if cancel.is_set():
                raise InterruptedError('verification cancelled')
            evidence = inspect(self.model, obs, goal, cancel)
            rows.append((obs.id, evidence.values))
        return verify_light_sequence(rows, 'model_visual_judgment')

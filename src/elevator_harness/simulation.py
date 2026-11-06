"""Deterministic contract fixture, NOT a physical simulator or success benchmark."""
import base64
import time
import uuid

from .types import (Capabilities, OperationFrame, ToolConfiguration, Observation, Image,
                    ExecutionResult, ActionProposal)
from .tasks import verify_light_sequence

def synthetic_image(light=False):
    import struct
    import zlib
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data))
    width, height = 320, 180
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend((40,220,40) if light and 128 <= x < 192 and 72 <= y < 108 else (160,160,160))
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+
            chunk(b'IDAT',zlib.compress(bytes(rows)))+chunk(b'IEND',b''))


PIXEL = synthetic_image()
LIT_PIXEL = synthetic_image(True)


class SimulatedRobot:
    def __init__(self, *, rotated=False, movable_view=True):
        frame = OperationFrame('sim_base', (0., 1., 0.), (0., 0., 1.), (1., 0., 0.)) if not rotated else \
                OperationFrame('sim_base', (1., 0., 0.), (0., 0., 1.), (0., -1., 0.))
        names = ('open', 'closed') if not rotated else ('inspect', 'contact')
        self.capabilities = Capabilities(frame, ('global', 'local'), tuple(
            ToolConfiguration(n, 'synthetic contact', 'sim-v1', (.5, .5)) for n in names),
            ('global',) if movable_view else (), True)
        self.configuration = names[0]
        self.position = [0., 0., 0.]
        self.normal = 0.
        self.light = False
        self.stopped = False
        self.commands = {}

    def observe(self, cancel, timeout):
        if cancel.is_set() or self.stopped:
            raise InterruptedError('simulation stopped')
        return Observation(uuid.uuid4().hex, time.time_ns(),
            {role: Image(LIT_PIXEL if self.light else PIXEL, 'image/png') for role in self.capabilities.cameras},
            self.configuration, 'sim-v1', {'position_m': self.position[:],
            'light_on': self.light, 'hand_clear': self.normal < .015,
            'view_angles_rad': {'global': [0., 0.]}}, .18-self.normal, True)

    def execute(self, action, cancel, timeout):
        if cancel.is_set() or self.stopped:
            raise InterruptedError('simulation stopped')
        action.validate(self.capabilities)
        if action.command_id in self.commands:
            previous, result = self.commands[action.command_id]
            if previous != action:
                raise ValueError('command identity conflict')
            return result
        if action.kind == 'move':
            self.normal += action.arguments['delta'][2]
            for i, d in enumerate(self.capabilities.frame.to_robot(action.arguments['delta'])):
                self.position[i] += d
            if self.normal >= .03 and self.configuration == self.capabilities.tools[-1].name:
                self.light = True
        elif action.kind == 'configure_tool':
            self.configuration = action.arguments['configuration']
        obs = self.observe(cancel, timeout)
        result = ExecutionResult(action.command_id, True, True, obs.state, 'complete', obs)
        self.commands[action.command_id] = (action, result)
        return result

    def stop(self):
        self.stopped = True

    def close(self):
        pass


class ScriptedAgent:
    def __init__(self):
        self.step = 0

    def decide(self, context, cancel):
        sequence = [
            ('move', {'delta': [.005, -.005, 0.]}),
            ('configure_tool', {'configuration': context.capabilities.tools[-1].name}),
            ('move', {'delta': [0., 0., .02]}),
            ('move', {'delta': [0., 0., .01]}),
            ('move', {'delta': [0., 0., -.02]}),
            ('done', {}),
        ]
        kind, args = sequence[min(self.step, len(sequence)-1)]
        self.step += 1
        return ActionProposal(context.observation.id, kind, args, 'deterministic fixture')


class SimulatedVerifier:
    def verify(self, observations, goal, cancel):
        return verify_light_sequence([(o.id, o.state) for o in observations], 'simulation_fixture')

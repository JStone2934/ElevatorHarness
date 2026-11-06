"""Core RobotAdapter over the optional robot-local S1 service."""
import base64
import hashlib
import math
import time
import uuid
from urllib.parse import urlsplit
from elevator_harness import Observation, Image, ExecutionResult
from .configuration import capabilities, configuration_hash
from .common import request, SESSION_SCHEMA


class S1Adapter:
    def __init__(self, base, settings, recorder=None):
        self.base, self.settings, self.recorder = base, settings, recorder
        self.capabilities = capabilities(settings)
        self.session = request(base, '/v2/session')
        if self.session.get('schema') != SESSION_SCHEMA:
            raise ValueError('unsupported S1 session protocol')
        if self.session.get('configuration_hash') != configuration_hash(settings):
            raise ValueError('client/service calibration or configuration mismatch')
        self.latest = None
        self.closed = False
        self._results = {}

    def _call(self, route, payload, cancel, deadline):
        # POST retries reuse the exact command ID, preserving service idempotency.
        for attempt in range(2):
            if cancel.is_set():
                raise InterruptedError('S1 request cancelled')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('S1 operation timed out')
            try:
                return request(self.base, route, payload, timeout=min(2., remaining))
            except (OSError, TimeoutError):
                if attempt:
                    raise

    def _wait(self, receipt, cancel, deadline):
        while True:
            task = self._call('/v2/tasks/'+receipt['task_id'], None, cancel, deadline)
            if task['status'] in ('completed', 'failed', 'interrupted', 'rejected'):
                if self.recorder:
                    self.recorder.event('s1_task', task)
                return task
            if cancel.wait(.05):
                raise InterruptedError('S1 wait cancelled')

    def _observation(self, task, cancel, deadline):
        raw = self._call(task['observation_url'], None, cancel, deadline)
        if raw['session_id'] != self.session['session_id']:
            raise ValueError('S1 observation belongs to a different session')
        images = {}
        for role, name in (('global', 'head.jpg'), ('local', 'right.jpg')):
            ref = raw['artifacts'][name]
            if not ref['url'].startswith('/v2/artifacts/'):
                raise ValueError('invalid artifact route')
            if cancel.is_set():
                raise InterruptedError('capture download cancelled')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('artifact download timed out')
            data = request(self.base, ref['url'], timeout=min(2., remaining), binary=True)
            if hashlib.sha256(data).hexdigest() != ref['sha256']:
                raise ValueError('image checksum mismatch')
            images[role] = Image(data)
        state = raw['state']
        position = state['joints']['astribot_gripper_right']['position'][0]
        key = 'o' if position <= 10 else 'c' if position >= 90 else None
        if key is None:
            raise ValueError('measured gripper configuration is intermediate; observation is not usable')
        name = next(k for k, v in self.settings['gripper_keys'].items() if v == key)
        expected_point = self.settings['depth_point']
        selection = raw['selection'].get('point_normalized')
        valid = (selection is not None and len(selection) == 2 and
                 all(abs(a-b) < 1e-8 for a, b in zip(selection, expected_point)) and
                 raw['depth'].get('valid_pixels', 0) >= self.settings.get('min_depth_pixels', 5))
        obs = Observation(raw['observation_id'], raw['utc_ns'], images, name,
                          self.capabilities.tool(name).calibration_version,
                          dict(state, position_m=state['pose_m_xyzw'][:3], settle=task.get('settle'),
                               view_angles_rad={'global': [math.radians(v) for v in state['head_deg']]}
                               if state.get('head_deg') is not None else {}),
                          raw['depth'].get('median_m'), valid)
        self.latest = obs.id
        return obs

    def observe(self, cancel, timeout):
        deadline = time.monotonic()+timeout
        payload = {'session_id': self.session['session_id'], 'command_id': uuid.uuid4().hex}
        task = self._wait(self._call('/v2/observations', payload, cancel, deadline), cancel, deadline)
        if task['status'] != 'completed':
            raise RuntimeError('S1 capture failed: '+str(task.get('error')))
        return self._observation(task, cancel, deadline)

    def execute(self, action, cancel, timeout):
        action.validate(self.capabilities)
        if action.command_id in self._results:
            previous, result = self._results[action.command_id]
            if action != previous:
                raise ValueError('command identity conflict')
            return result
        if action.observation_id != self.latest:
            raise ValueError('action observation is stale')
        if action.kind == 'move':
            cmd = {'type': 'move', 'xyz_mm': [1000*x for x in self.capabilities.frame.to_robot(action.arguments['delta'])]}
        elif action.kind == 'configure_tool':
            cmd = {'type': 'gripper', 'key': self.settings['gripper_keys'][action.arguments['configuration']]}
        elif action.kind == 'adjust_view':
            cmd = {'type': 'head', 'head_deg': [math.degrees(x) for x in action.arguments['angles']]}
        else:
            raise ValueError('not an executable S1 action')
        deadline = time.monotonic()+timeout
        payload = dict(cmd, session_id=self.session['session_id'], command_id=action.command_id,
                       observation_id=action.observation_id)
        task = self._wait(self._call('/v2/actions', payload, cancel, deadline), cancel, deadline)
        obs = self._observation(task, cancel, deadline) if task['status'] == 'completed' else None
        result = ExecutionResult(action.command_id, True, task['motion']['action_completed'],
            obs.state if obs else task.get('failure_state', {}), 'complete' if obs else 'failed', obs, task.get('error'))
        self._results[action.command_id] = (action, result)
        return result

    def stop(self):
        if not self.closed:
            request(self.base, '/v2/stop', {}, timeout=2)
            self.closed = True

    def close(self):
        self.stop()

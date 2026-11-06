"""State sampling and service adapter around the shared keyboard executor."""
import copy
import math
import threading
import time
from collections import deque

from .common import number

# Soft settle before a photo: never a gate, only a short best-effort wait.
SETTLE_TIMEOUT_S = 1.5
SETTLE_WINDOW_S = 0.15
SETTLE_POSITION_M = 0.001
SETTLE_HEAD_DEG = 0.3
# Gripper (state units 0 open … ~97.6 closed; command 0 / 100). A gripper command
# completes at its first send, before the fingers move (2026-11-04 runs: photos at
# 5.5, 23.4, 49.8, 58.5 while closing, 96.1 / 64.6 while opening). Before the photo
# the fingers must have left the start or reached the target, then stay still. A
# full stroke plus start latency needs more than the arm timeout.
GRIPPER_SETTLE_TIMEOUT_S = 3.0
GRIPPER_SETTLE_SPAN = 0.5
GRIPPER_MIN_TRAVEL = 5.0
GRIPPER_TARGET_TOLERANCE = 3.0
GRIPPER_PART = 'astribot_gripper_right'


def _gripper_position(row):
    try:
        value = row['joints'][GRIPPER_PART]['position'][0]
    except (KeyError, IndexError, TypeError):
        return None
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


class SDKBackend:
    def __init__(self, robot, monitor, frequency=250, speed_mm_s=20., head_speed_deg_s=20.):
        from .executor import CommandExecutor
        self.robot, self.monitor = robot, monitor
        self.controller = CommandExecutor(robot, frequency, speed_mm_s, head_speed_deg_s)
        self.names = self.controller.names
        self.lock = threading.RLock()
        self.done = self.controller.stop_event
        self.measured = None
        self.history = deque(maxlen=6000)
        self.sampling_error = None
        self.gripper_expect = None  # set by a gripper command, consumed by the next wait_settled
        try:
            self.measured = self._read()
            self.history.append(copy.deepcopy(self.measured))
            self.reader = threading.Thread(target=self._sample, daemon=True)
            self.reader.start()
        except BaseException:
            self.controller.close()
            raise

    @property
    def error(self):
        return self.controller.error

    @property
    def error_traceback(self):
        return self.controller.error_traceback

    def _read(self):
        rows=self.monitor.snapshot()
        needed=('astribot_arm_right','astribot_torso','astribot_gripper_right')
        poses=self.robot.get_current_cartesian_pose(self.names,frame='chassis')
        pose=[number(x) for x in poses[0]]
        target=self.controller.command_pose()
        return dict(pose_m_xyzw=pose,command_pose_m_xyzw=target,frame_id='chassis',reference_point='sdk_end_effector',
                    receipt_ns=time.monotonic_ns(),joints={k:rows[k] for k in needed},
                    head_deg=[math.degrees(v) for v in self.robot.get_current_joints_position(['astribot_head'])[0]],
                    gripper_command=self.controller.gripper_target)

    def _sample(self):
        """Observation recording only; feedback does not gate keyboard motion."""
        while not self.done.is_set():
            try:
                measured=self._read()
                with self.lock:
                    self.measured=measured;self.history.append(measured)
                    self.sampling_error=None
            except Exception as exc:
                self.sampling_error=str(exc)
            self.done.wait(.02)

    def snapshot(self):
        with self.lock:
            if self.error:raise RuntimeError(self.error)
            if self.measured is None or time.monotonic_ns()-self.measured['receipt_ns']>300_000_000:
                raise RuntimeError('state_unavailable')
            return copy.deepcopy(self.measured)

    def wait_settled(self, cancel, timeout_s=SETTLE_TIMEOUT_S):
        """Best effort: wait until the hand and head stop moving before a photo.

        Start a fresh window after command completion, rather than accepting the
        quiet samples from before the arm responded. Settled means observed
        stillness, NOT measured arrival at the commanded target.
        Every state sample of the last SETTLE_WINDOW_S stays within
        SETTLE_POSITION_M of each other (end-effector position) and SETTLE_HEAD_DEG
        (head joints). This never raises and never blocks execution: after timeout_s,
        a stop, or unreadable state, it returns and the photo is taken anyway; the
        returned record says which.
        """
        expect, self.gripper_expect = getattr(self, 'gripper_expect', None), None
        if expect is not None:
            timeout_s = max(timeout_s, GRIPPER_SETTLE_TIMEOUT_S)
        started = time.monotonic_ns()
        deadline = started + int(timeout_s*1e9)
        reason = 'timeout'
        gripper = None
        position = head = None
        while True:
            now = time.monotonic_ns()
            try:
                with self.lock:
                    rows = [row for row in self.history
                            if row['receipt_ns'] >= max(started, now - int(SETTLE_WINDOW_S*1e9))]
                if len(rows) >= 3 and rows[-1]['receipt_ns'] - started >= SETTLE_WINDOW_S*1e9:
                    position = max(math.dist(a['pose_m_xyzw'][:3], b['pose_m_xyzw'][:3])
                                   for a in rows for b in rows)
                    head = max((abs(x-y) for a in rows for b in rows
                                for x, y in zip(a.get('head_deg') or [], b.get('head_deg') or [])), default=0.0)
                    gripper_ok = True
                    if expect is not None:
                        gripper = self._gripper_check(expect, [r for r in rows if r['receipt_ns'] >= expect['began_ns']])
                        gripper_ok = gripper['done']
                    if position <= SETTLE_POSITION_M and head <= SETTLE_HEAD_DEG and gripper_ok:
                        reason = 'settled'
                        break
            except Exception as exc:
                reason = 'state_unreadable:'+str(exc)
                break
            if now >= deadline:
                break
            if cancel is not None and cancel.wait(.02):
                reason = 'stopped'
                break
            if cancel is None:
                time.sleep(.02)
        result = dict(settled=reason == 'settled', reason=reason,
                      waited_ms=round((time.monotonic_ns()-started)/1e6, 1), timeout_ms=timeout_s*1000,
                      position_span_mm=None if position is None else round(position*1000, 3),
                      head_span_deg=head, semantics='observed_stillness_not_target_arrival; advisory_only')
        if expect is not None:
            result['gripper'] = dict(expect, **(gripper or {}))
        return result

    @staticmethod
    def _gripper_check(expect, rows):
        """Fingers have left the start (or are at the target) and stay still over the window."""
        values = [v for v in (_gripper_position(r) for r in rows) if v is not None]
        if len(values) < 3:
            return dict(done=False, position=values[-1] if values else None, why='too_few_gripper_samples')
        last, span = values[-1], max(values)-min(values)
        start, target = expect.get('start'), expect['target']
        moved = start is not None and abs(last-start) >= GRIPPER_MIN_TRAVEL
        at_target = abs(last-target) <= GRIPPER_TARGET_TOLERANCE
        done = span <= GRIPPER_SETTLE_SPAN and (moved or at_target)
        why = ('settled' if done else 'still_moving' if span > GRIPPER_SETTLE_SPAN else 'not_started')
        return dict(done=done, position=last, span=round(span, 3), why=why)

    def state_for_capture(self, bundle):
        source=bundle['frames']['right']['source_ns']
        def stamp(row):
            h=row['joints']['astribot_arm_right']['header']
            return h['sec']*10**9+h['nanosec']
        with self.lock:
            if not self.history:raise RuntimeError('no_state_for_capture')
            state=copy.deepcopy(min(self.history,key=lambda row:abs(stamp(row)-source)))
        state['rgb_state_skew_ms']=abs(stamp(state)-source)/1e6
        if state['rgb_state_skew_ms']>80:raise RuntimeError('image_state_skew')
        return state

    def move(self, kind, data, cancel, phase):
        field = {'move': 'xyz_mm', 'head': 'head_deg', 'gripper': 'key'}[kind]
        prepared = self.controller.prepare(dict(type=kind, **{field: data[field]}))
        if cancel.is_set():
            raise RuntimeError('stopped')
        phase('MOVING')
        if kind == 'gripper':
            from .executor import GRIPPER_KEYS
            with self.lock:
                start = _gripper_position(self.measured) if self.measured else None
            # The next photo waits for these fingers; the command itself returns at first send.
            self.gripper_expect = dict(key=data[field], target=float(GRIPPER_KEYS[data[field]]),
                                       start=start, began_ns=time.monotonic_ns())
        task = self.controller.start(prepared)
        self.controller.wait(task, cancel)
        began = task['started']['monotonic_ns']
        with self.lock:
            final = copy.deepcopy(self.measured)
            history = [copy.deepcopy(row) for row in self.history if row['receipt_ns'] >= began]
        return dict(start_pose_m_xyzw=task['start'], target_pose_m_xyzw=task['goal'], final_state=final,
            returned_to_start=None, completion_basis='first_successful_send' if kind == 'gripper' else 'command_interpolation',
            started_monotonic_ns=began, completed_monotonic_ns=task['ended']['monotonic_ns'],
            state_trace=history, execution_mode='shared_keyboard_executor')

    def diagnostic(self):
        with self.lock:
            return dict(error=self.error, traceback=self.error_traceback,
                last_state=copy.deepcopy(self.measured), command_pose=self.controller.command_pose(),
                state_trace=copy.deepcopy(list(self.history)), sampling_error=self.sampling_error)

    def stop(self):
        self.controller.stop()

    def close(self):
        try:
            self.controller.close()
        finally:
            self.reader.join(3)
            if self.reader.is_alive():
                raise RuntimeError('SDK state reader did not stop')

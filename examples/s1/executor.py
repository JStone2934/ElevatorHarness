"""Single keyboard/agent command executor; no terminal, network or recording I/O."""
import copy
import math
import threading
import time
import traceback

GRIPPER_KEYS = {'c': 100., 'o': 0.}


def interpolate_pose(start, goal, elapsed, duration):
    fraction = min(1., max(0., elapsed/duration))
    blend = 10*fraction**3-15*fraction**4+6*fraction**5
    return [start[i]+blend*(goal[i]-start[i]) for i in range(3)]+start[3:], fraction >= 1.


def send_targets(robot, names, current, torso, gripper_target, head_target=None):
    if head_target is not None:
        # Serial head-only operation: one SDK set per tick, no competing writer.
        robot.set_joints_position(['astribot_head'],[list(head_target)],
            control_way='filter',use_wbc=False,add_default_torso=False)
    elif gripper_target is None:
        robot.set_cartesian_pose(names, [list(current), torso], control_way='filter',
            use_wbc=False, add_default_torso=False, remote_wbc_option=False)
    else:
        # SDK example 110: mixed Cartesian/joint targets in ONE periodic set.
        robot.set_different_type_command(list(names)+['astribot_gripper_right'],
            ['cartesian', 'cartesian', 'joints'], [list(current), torso, [gripper_target]],
            control_way='filter', use_wbc=False)


def run_cartesian_loop(robot, frequency, stop, lock, targets, on_error, on_sent=None):
    """The keyboard command loop; callers only supply the current targets."""
    deadline = time.monotonic()
    try:
        while not stop.is_set():
            with lock:
                if stop.is_set():break
                if not robot.get_control_rights_status():
                    raise RuntimeError('SDK control rights lost')
                send_targets(robot, *targets())
                if on_sent:on_sent()
            deadline += 1/frequency
            if deadline < time.monotonic():deadline = time.monotonic()+1/frequency
            stop.wait(max(0., deadline-time.monotonic()))
    except BaseException as exc:
        on_error(exc)


def interpolate_head(start, goal, elapsed, duration):
    fraction=min(1.,max(0.,elapsed/duration))
    blend=10*fraction**3-15*fraction**4+6*fraction**5
    return [a+blend*(b-a) for a,b in zip(start,goal)],fraction>=1.


def target_for_xyz(pose, delta_mm):
    return [pose[i]+delta_mm[i]/1000 for i in range(3)]+list(pose[3:])


def validate_command(command):
    """The same finite-number contract for terminal and model commands."""
    if not isinstance(command, dict):
        raise ValueError('invalid_command')
    kind = command.get('type')
    field = {'move': 'xyz_mm', 'head': 'head_deg', 'gripper': 'key'}.get(kind)
    if field is None or set(command) != {'type', field}:
        raise ValueError('invalid_command')
    value = command[field]
    if kind == 'gripper':
        if not isinstance(value, str) or value not in GRIPPER_KEYS:
            raise ValueError('invalid_gripper_key')
    else:
        size = 3 if kind == 'move' else 2
        if (not isinstance(value, (list, tuple)) or len(value) != size or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
            raise ValueError('finite_vector_required')
    return copy.deepcopy(command)


def keyboard_command(value):
    if isinstance(value, str):
        command = dict(type='gripper', key=value)
    elif isinstance(value, dict):
        command = dict(type='head', head_deg=value['head_deg'])
    else:
        command = dict(type='move', xyz_mm=value)
    return validate_command(command)


class CommandExecutor:
    """One SDK writer and one command state machine shared by both inputs.

    prepare() permits recording the before image without arming a movement.
    A command completes only after its final SDK send has returned.
    """

    def __init__(self, robot, frequency=250., speed_mm_s=20., head_speed_deg_s=20., stop_event=None):
        for value in (frequency, speed_mm_s, head_speed_deg_s):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('positive_speed_and_frequency_required')
        self.robot = robot
        self.frequency, self.speed_mm_s, self.head_speed_deg_s = frequency, speed_mm_s, head_speed_deg_s
        self.lock = threading.RLock()
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.first_send = threading.Event()
        self.error = self.error_traceback = None
        self.names = ['astribot_arm_right', 'astribot_torso']
        self.current, self.torso = copy.deepcopy(robot.get_current_cartesian_pose(self.names, frame='chassis'))
        self.gripper_target = None
        self.active = None
        self._complete = False
        self.thread = threading.Thread(target=self._servo, name='keyboard-command-servo', daemon=True)
        self.thread.start()
        if not self.first_send.wait(3) or self.error:
            self.close()
            raise RuntimeError((self.error or 'initial_command_timeout') + '\n' + (self.error_traceback or ''))

    def command_pose(self):
        with self.lock:
            return list(self.current)

    def prepare(self, command):
        command = validate_command(command)
        with self.lock:
            if self.stop_event.is_set():
                raise RuntimeError(self.error or 'stopped')
            if self.active is not None:
                raise RuntimeError('controller_busy')
            start = list(self.current)
        kind = command['type']
        goal = target_for_xyz(start, command['xyz_mm']) if kind == 'move' else list(start)
        prepared = dict(command=command, start=start, goal=goal)
        if kind == 'head':
            head_start = list(self.robot.get_current_joints_position(['astribot_head'])[0])
            if len(head_start) != 2 or any(not math.isfinite(v) for v in head_start):
                raise RuntimeError('SDK head must have two finite joints')
            head_goal = [math.radians(v) for v in command['head_deg']]
            duration = max(.2, 1.875 * max(abs(b-a) for a, b in zip(head_start, head_goal)) /
                           math.radians(self.head_speed_deg_s))
            prepared.update(head_start=head_start, head_goal=head_goal, duration=duration)
        elif kind == 'move':
            prepared['duration'] = max(.2, 1.875 * math.dist(start[:3], goal[:3]) / (self.speed_mm_s/1000))
        if any(not math.isfinite(v) for v in goal) or not math.isfinite(prepared.get('duration', 0)):
            raise ValueError('command_arithmetic_overflow')
        return prepared

    def start(self, prepared):
        with self.lock:
            if self.stop_event.is_set():
                raise RuntimeError(self.error or 'stopped')
            if self.active is not None:
                raise RuntimeError('controller_busy')
            if prepared['start'] != self.current:
                raise ValueError('prepared_command_changed')
            task = copy.deepcopy(prepared)
            task.update(done=threading.Event(), started=dict(utc_ns=time.time_ns(), monotonic_ns=time.monotonic_ns()),
                        began=time.monotonic(), ended=None)
            if task['command']['type'] == 'gripper':
                self.gripper_target = GRIPPER_KEYS[task['command']['key']]
            self.active = task
            return task

    def _targets(self):
        self._complete = False
        task = self.active
        if task is not None:
            kind = task['command']['type']
            if kind == 'head':
                head, self._complete = interpolate_head(task['head_start'], task['head_goal'],
                                                        time.monotonic()-task['began'], task['duration'])
                return self.names, self.current, self.torso, self.gripper_target, head
            if kind == 'move':
                self.current, self._complete = interpolate_pose(task['start'], task['goal'],
                    time.monotonic()-task['began'], task['duration'])
                if task['start'] == task['goal']:
                    self._complete = True
            else:
                self._complete = True
        return self.names, self.current, self.torso, self.gripper_target

    def _sent(self):
        self.first_send.set()
        if self.active is not None and self._complete:
            task = self.active
            task['ended'] = dict(utc_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
            self.active = None
            task['done'].set()

    def _failed(self, exc):
        self.error, self.error_traceback = str(exc), traceback.format_exc()
        self.first_send.set()
        self.stop()

    def _servo(self):
        run_cartesian_loop(self.robot, self.frequency, self.stop_event, self.lock,
                           self._targets, self._failed, self._sent)

    def wait(self, task, cancel):
        while not task['done'].wait(.01):
            if cancel.is_set() or self.stop_event.is_set():
                raise RuntimeError(self.error or 'stopped')
        return task

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.robot.get_control_rights_status():
                self.robot.stop_robot()

    def close(self):
        self.stop()
        if threading.current_thread() is not self.thread:
            self.thread.join(3)
            if self.thread.is_alive():
                raise RuntimeError('SDK call did not return after stop')

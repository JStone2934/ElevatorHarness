import pytest
import copy
import threading
import time
from types import SimpleNamespace
from examples.s1.executor import CommandExecutor, keyboard_command
from examples.s1.sdk_backend import SDKBackend


class Robot:
    def __init__(self):
        self.pose = [.4, -.2, 1., 0., 0., 0., 1.]
        self.torso = [0., 0., .5, 0., 0., 0., 1.]
        self.head = [0., 0.]
        self.calls = []
        self.stopped = False

    def get_control_rights_status(self):
        return True

    def get_current_cartesian_pose(self, *args, **kwargs):
        assert threading.current_thread().name != 'keyboard-command-servo'
        return [list(self.pose), list(self.torso)]

    def get_current_joints_position(self, names):
        assert threading.current_thread().name != 'keyboard-command-servo'
        return [list(self.head)]

    def set_cartesian_pose(self, names, poses, **kwargs):
        assert kwargs == dict(control_way='filter', use_wbc=False, add_default_torso=False, remote_wbc_option=False)
        self.calls.append(('cartesian', copy.deepcopy((names, poses, kwargs))))
        self.pose = list(poses[0])

    def set_different_type_command(self, names, types, values, **kwargs):
        assert names == ['astribot_arm_right', 'astribot_torso', 'astribot_gripper_right']
        assert types == ['cartesian', 'cartesian', 'joints']
        assert kwargs == dict(control_way='filter', use_wbc=False)
        self.calls.append(('mixed', copy.deepcopy((names, values, kwargs))))
        self.pose = list(values[0])

    def set_joints_position(self, names, values, **kwargs):
        assert names == ['astribot_head']
        assert kwargs == dict(control_way='filter', use_wbc=False, add_default_torso=False)
        self.calls.append(('head', copy.deepcopy((names, values, kwargs))))
        self.head = list(values[0])

    def stop_robot(self):
        self.stopped = True


def monitor_rows():
    now = time.time_ns()
    return {name: dict(header=dict(sec=now//10**9, nanosec=now % 10**9))
            for name in ('astribot_arm_right', 'astribot_torso', 'astribot_gripper_right')}


def test_keyboard_and_backend_same_commands_and_sdk_routes():
    robots = [Robot(), Robot()]
    keyboard = CommandExecutor(robots[0], 100, 1000, 1000)
    backend = SDKBackend(robots[1], SimpleNamespace(snapshot=monitor_rows), 100, 1000, 1000)
    inputs = ['c', {'head_deg': [0, 30]}, [0, 0, 300], [45, 20, -50],
              [15, 0, 0], [-100, 0, 0], 'o', [0, 0, 0]]
    try:
        assert all(call[0] == 'cartesian' for r in robots for call in r.calls)
        for value in inputs:
            command = keyboard_command(value)
            offsets = [len(r.calls) for r in robots]
            prepared = keyboard.prepare(command)
            task = keyboard.start(prepared)
            keyboard.wait(task, threading.Event())
            result = backend.move(command['type'], command, threading.Event(), lambda _: None)
            assert task['ended']['monotonic_ns'] >= task['started']['monotonic_ns']
            assert result['completed_monotonic_ns'] >= result['started_monotonic_ns']
            assert result['target_pose_m_xyzw'] == pytest.approx(task['goal'])
            assert robots[0].pose == pytest.approx(robots[1].pose)
            assert robots[0].head == pytest.approx(robots[1].head)
            route = 'head' if command['type'] == 'head' else 'mixed'
            final_calls = [[call for call in r.calls[offset:] if call[0] == route][-1]
                           for r, offset in zip(robots, offsets)]
            assert final_calls[0] == final_calls[1]
            assert result['completion_basis'] == ('first_successful_send' if route == 'mixed' and
                                                  command['type'] == 'gripper' else 'command_interpolation')
        assert robots[0].pose[:3] == pytest.approx([.36, -.18, 1.25])
        assert robots[0].pose[3:] == [0, 0, 0, 1]
    finally:
        keyboard.close()
        backend.close()

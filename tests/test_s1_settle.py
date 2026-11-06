import threading
import time
from collections import deque

from examples.s1 import sdk_backend
from examples.s1.sdk_backend import SDKBackend
from examples.s1.service import Service

from test_s1_service import FakeBackend, FakeCamera, observe, action, wait


class Sampler:
    """SDKBackend.wait_settled on a synthetic state stream; no SDK."""
    def __init__(self, moving_until_s=0.0, speed_m_s=0.02):
        self.lock = threading.RLock()
        self.history = deque(maxlen=6000)
        self.start = time.monotonic_ns()
        self.done = threading.Event()
        def run():
            while not self.done.is_set():
                now = time.monotonic_ns()
                t = (now-self.start)/1e9
                x = 0.4 + speed_m_s*min(t, moving_until_s)
                with self.lock:
                    self.history.append(dict(receipt_ns=now, pose_m_xyzw=[x, -.2, 1., 0, 0, 0, 1], head_deg=[0., 30.]))
                self.done.wait(.01)
        self.thread = threading.Thread(target=run, daemon=True); self.thread.start()
        time.sleep(.05)
    wait_settled = SDKBackend.wait_settled
    def close(self):
        self.done.set(); self.thread.join(1)


def test_waits_for_rest_then_reports_settled():
    s = Sampler(moving_until_s=0.4)
    try:
        result = s.wait_settled(threading.Event(), timeout_s=2.0)
    finally:
        s.close()
    assert result['settled'] and result['reason'] == 'settled'
    assert 300 <= result['waited_ms'] < 1500


def test_still_moving_times_out_without_raising():
    s = Sampler(moving_until_s=10.0)
    try:
        result = s.wait_settled(threading.Event(), timeout_s=0.3)
    finally:
        s.close()
    assert not result['settled'] and result['reason'] == 'timeout' and result['waited_ms'] < 600


def test_stop_ends_the_wait_immediately():
    s = Sampler(moving_until_s=10.0); cancel = threading.Event(); cancel.set()
    try:
        result = s.wait_settled(cancel, timeout_s=5.0)
    finally:
        s.close()
    assert result['reason'] == 'stopped' and result['waited_ms'] < 200


def test_service_photographs_even_if_settling_fails_or_times_out(tmp_path):
    class Unsettled(FakeBackend):
        def __init__(self, mode):
            super().__init__(); self.mode = mode
        def wait_settled(self, cancel):
            if self.mode == 'raise':
                raise RuntimeError('sdk glitch')
            return dict(settled=False, reason='timeout', waited_ms=1500.0, timeout_ms=1500.0)
    for mode in ('timeout', 'raise'):
        service = Service(Unsettled(mode), FakeCamera(), tmp_path/mode)
        try:
            obs = observe(service)
            task = wait(service, service.submit(action(service, obs, 'm', type='move', xyz_mm=[-10, 0, 0])))
            assert task['status'] == 'completed' and task['capture']['status'] == 'complete'
            assert task['settle']['settled'] is False
            assert ('settle_check_failed' in task['settle']['reason']) == (mode == 'raise')
        finally:
            service.close()


def test_defaults_are_soft_and_short():
    assert sdk_backend.SETTLE_TIMEOUT_S <= 2.0


def test_old_quiet_samples_cannot_end_a_new_wait(monkeypatch):
    # The old implementation could settle immediately on pre-response samples.
    from types import SimpleNamespace
    clock = [1_000_000_000]
    monkeypatch.setattr(sdk_backend.time, 'monotonic_ns', lambda: clock[0])
    def row(stamp):
        return dict(receipt_ns=stamp, pose_m_xyzw=[.4, 0, 1, 0, 0, 0, 1],
                    command_pose_m_xyzw=[.4, 0, 1.008, 0, 0, 0, 1], head_deg=[0, 30])
    sampler = SimpleNamespace(lock=threading.RLock(),
        history=deque(row(t) for t in range(800_000_000, 1_000_000_001, 20_000_000)))
    class FreshSamples:
        def wait(self, seconds):
            clock[0] += int(seconds*1e9)
            sampler.history.append(row(clock[0]))
            return False
    result = SDKBackend.wait_settled(sampler, FreshSamples(), timeout_s=.3)
    assert result['waited_ms'] >= sdk_backend.SETTLE_WINDOW_S*1000
    # Tracking error is not a new arrival gate: even an 8 mm residual is allowed.
    assert result['settled'] and result['waited_ms'] < 300


class GripperSampler(Sampler):
    """Arm still; gripper starts after ``latency_s`` and travels to ``end`` in ``stroke_s``."""
    def __init__(self, start=0.0, end=97.6, latency_s=0.2, stroke_s=0.4):
        self.lock = threading.RLock()
        self.history = deque(maxlen=6000)
        self.start = time.monotonic_ns()
        self.done = threading.Event()
        def run():
            while not self.done.is_set():
                now = time.monotonic_ns()
                t = min(max((now-self.start)/1e9-latency_s, 0.0)/stroke_s, 1.0)
                g = start + (end-start)*t
                with self.lock:
                    self.history.append(dict(receipt_ns=now, pose_m_xyzw=[.4, -.2, 1., 0, 0, 0, 1], head_deg=[0., 30.],
                                             joints={sdk_backend.GRIPPER_PART: dict(position=[g])}))
                self.done.wait(.01)
        self.thread = threading.Thread(target=run, daemon=True); self.thread.start()
        self.gripper_expect = dict(key='c', target=100.0, start=start, began_ns=self.start)
    _gripper_check = staticmethod(SDKBackend._gripper_check)


def _settle(sampler, **kw):
    try:
        return sampler.wait_settled(threading.Event(), **kw)
    finally:
        sampler.close()


def test_photo_waits_for_the_gripper_to_finish_closing():
    # Arm is still the whole time: before this fix the photo was taken at once.
    result = _settle(GripperSampler(latency_s=0.2, stroke_s=0.4))
    assert result['settled'] and result['gripper']['why'] == 'settled'
    assert result['gripper']['position'] > 97 and result['waited_ms'] >= 600


def test_photo_waits_for_the_gripper_to_finish_opening():
    s = GripperSampler(start=97.6, end=0.0, latency_s=0.1, stroke_s=0.3)
    s.gripper_expect = dict(key='o', target=0.0, start=97.6, began_ns=s.start)
    result = _settle(s)
    assert result['settled'] and abs(result['gripper']['position']) < 1 and result['waited_ms'] >= 400


def test_gripper_stalled_on_an_object_counts_once_it_moved_and_stopped():
    result = _settle(GripperSampler(end=60.0, latency_s=0.05, stroke_s=0.2))
    assert result['settled'] and 59 < result['gripper']['position'] <= 60


def test_gripper_that_never_starts_times_out_and_is_recorded():
    result = _settle(GripperSampler(latency_s=99), timeout_s=0.1)
    # The gripper timeout replaces the shorter arm default; the record says why.
    assert not result['settled'] and result['reason'] == 'timeout'
    assert result['timeout_ms'] == sdk_backend.GRIPPER_SETTLE_TIMEOUT_S*1000
    assert result['gripper']['why'] == 'not_started'


def test_expectation_is_used_once():
    s = GripperSampler(latency_s=0.0, stroke_s=0.1)
    _settle(s)
    assert s.gripper_expect is None

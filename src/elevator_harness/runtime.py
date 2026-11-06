"""One finite action at a time. A stopped run never resumes or replays commands."""
from dataclasses import dataclass, asdict
from threading import Event
import copy

from .types import Context, Decision, Verification


class AllowActions:
    def check(self, action, context):
        return Decision(True)
    def record(self, action, before, result):
        pass
    def describe(self):
        return {}


@dataclass(frozen=True)
class RunResult:
    status: str
    rounds: int
    model_reported_done: bool = False
    verification: Verification | None = None
    error: str | None = None


class Harness:
    def __init__(self, robot, agent, task, verifier, recorder, policy=None, *,
                 max_rounds=40, max_corrections=3, operation_timeout=120., history_limit=8):
        if (max_rounds < 1 or max_corrections < 0 or history_limit < 1 or operation_timeout <= 0):
            raise ValueError('invalid run budget')
        self.robot, self.agent, self.task = robot, agent, task
        self.verifier, self.recorder = verifier, recorder
        self.policy = policy or AllowActions()
        self.max_rounds, self.max_corrections = max_rounds, max_corrections
        self.operation_timeout, self.history_limit = operation_timeout, history_limit
        self.cancel = Event()
        self.started = False

    def stop(self):
        self.cancel.set()
        self.robot.stop()

    def _check(self):
        if self.cancel.is_set():
            raise InterruptedError('run cancelled')

    def run(self):
        if self.started:
            raise RuntimeError('Harness instances are single-use')
        self.started = True
        count = 0
        result = RunResult('budget_exhausted', 0)
        observations, history, commands = [], [], set()

        def accept(obs):
            self._check()
            if any(old.id == obs.id for old in observations):
                raise ValueError('observation must be fresh')
            if observations and obs.timestamp_ns < observations[-1].timestamp_ns:
                raise ValueError('observation timestamp moved backwards')
            config = self.robot.capabilities.tool(obs.tool)
            if config.calibration_version != obs.calibration_version:
                raise ValueError('observation calibration mismatch')
            observations.append(obs)
            self.task.observe(obs)
            self.recorder.observation(obs)
            return obs

        try:
            self.recorder.event('run', {'goal': self.task.goal, 'capabilities': self.robot.capabilities,
                'package_version': '0.1.0', 'agent': type(self.agent).__name__,
                'policy': self.policy.describe(), 'max_rounds': self.max_rounds,
                'max_corrections': self.max_corrections, 'operation_timeout': self.operation_timeout,
                'history_limit': self.history_limit})
            self._check()
            obs = accept(self.robot.observe(self.cancel, self.operation_timeout))
            for count in range(1, self.max_rounds + 1):
                self._check()
                context = Context(obs, self.robot.capabilities, self.task.goal,
                                  copy.deepcopy(history[-self.history_limit:]), self.task.describe(),
                                  policy_state=self.policy.describe())
                if hasattr(self.policy, 'prepare'):
                    self.policy.prepare(context)
                    context.policy_state = self.policy.describe()
                for correction in range(self.max_corrections + 1):
                    self._check()
                    try:
                        action = self.agent.decide(context, self.cancel)
                        self._check()
                        action.validate(self.robot.capabilities)
                        if action.observation_id != obs.id:
                            raise ValueError('action refers to an old observation')
                        if action.command_id in commands:
                            raise ValueError('duplicate command_id')
                        decision = self.policy.check(action, context)
                        self.recorder.event('proposal', {'action': action, 'decision': decision,
                                                       'evidence': context.evidence, 'decision_context': context.decision_data})
                        if not decision.allowed:
                            raise ValueError(decision.reason)
                        break
                    except ValueError as exc:
                        self.recorder.event('correction', {'round': count, 'attempt': correction,
                                                          'reason': str(exc)})
                        context.feedback = str(exc)
                        context.policy_state = self.policy.describe()
                        if correction == self.max_corrections:
                            raise RuntimeError('correction budget exhausted') from exc
                self._check()
                if action.kind == 'done':
                    verification = self.verifier.verify(observations, self.task.goal, self.cancel)
                    self._check()
                    if any(oid not in {o.id for o in observations} for oid in verification.evidence_ids):
                        raise ValueError('verification references unknown evidence')
                    result = RunResult('completed' if verification.verdict == 'confirmed' else 'unverified',
                                       count, True, verification)
                    break
                if action.kind == 'unable':
                    result = RunResult('unable', count, error=action.reason)
                    break
                commands.add(action.command_id)
                self.recorder.event('admitted', action)
                before = obs
                if action.kind == 'observe':
                    obs = accept(self.robot.observe(self.cancel, self.operation_timeout))
                else:
                    executed = self.robot.execute(action, self.cancel, self.operation_timeout)
                    self.recorder.event('execution', executed)
                    self._check()
                    if executed.command_id != action.command_id:
                        raise RuntimeError('execution command mismatch')
                    if not executed.accepted or not executed.sent or executed.error:
                        raise RuntimeError(executed.error or 'action not sent')
                    if executed.capture_status == 'failed':
                        raise RuntimeError('capture failed after command; action will not be repeated')
                    self.policy.record(action, before, executed)
                    self.task.update(action, executed)
                    obs = accept(executed.observation or self.robot.observe(self.cancel, self.operation_timeout))
                history.append({'action': asdict(action), 'before_id': before.id, 'after_id': obs.id,
                                'tool_before': before.tool, 'tool_after': obs.tool,
                                'measured_state': obs.state, 'depth_m': obs.depth_m,
                                'before_state': before.state, 'after_state': obs.state,
                                'decision_context': copy.deepcopy(context.decision_data),
                                'evidence': asdict(context.evidence) if context.evidence else None})
                self.recorder.event('transition', history[-1])
            else:
                result = RunResult('budget_exhausted', count)
        except InterruptedError:
            result = RunResult('cancelled', count)
        except Exception as exc:
            result = RunResult('error', count, error=f'{type(exc).__name__}: {exc}')
        finally:
            self.cancel.set()
            for method in (self.robot.stop, self.robot.close):
                try:
                    method()
                except Exception as exc:
                    self.recorder.event('cleanup_error', {'type': type(exc).__name__})
                    result = RunResult('error', count, result.model_reported_done,
                                       result.verification, 'robot cleanup was not confirmed')
            self.recorder.event('result', result)
        return result

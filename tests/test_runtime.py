from dataclasses import replace
import json
import threading
import time

import pytest
from elevator_harness import (Harness, Recorder, PressButtonTask, ActionProposal, ExecutionResult,
                               Verification, OperationFrame)
from elevator_harness.simulation import SimulatedRobot, ScriptedAgent, SimulatedVerifier


def run_fixture(tmp_path, robot=None, agent=None, **kwargs):
    robot = robot or SimulatedRobot()
    harness = Harness(robot, agent or ScriptedAgent(), PressButtonTask('button'),
                      SimulatedVerifier(), Recorder(tmp_path/'run'), **kwargs)
    return harness, robot


@pytest.mark.parametrize('rotated', [False, True])
def test_complete_and_independent_run_instances(tmp_path, rotated):
    run, robot = run_fixture(tmp_path, SimulatedRobot(rotated=rotated, movable_view=not rotated))
    result = run.run()
    assert result.status == 'completed'
    assert result.verification.source == 'simulation_fixture'
    assert len(result.verification.evidence_ids) == 3
    assert robot.stopped
    assert len(robot.commands) == 5
    assert (tmp_path/'run/events.jsonl').exists()
    with pytest.raises(RuntimeError, match='single-use'):
        run.run()


def test_rotated_coordinate_transform():
    frame = OperationFrame('base', (1, 0, 0), (0, 0, 1), (0, -1, 0))
    assert frame.to_robot((.01, .02, .03)) == pytest.approx((.01, -.03, .02))
    with pytest.raises(ValueError):
        OperationFrame('base', (1, 0, 0), (1, 0, 0), (0, 0, 1))


@pytest.mark.parametrize('bad_kind', ['stale', 'unsupported', 'zero', 'duplicate'])
def test_invalid_actions_never_execute(tmp_path, bad_kind):
    class Bad:
        def decide(self, c, cancel):
            if bad_kind == 'stale':
                return ActionProposal('old', 'move', {'delta': [0, 0, .01]})
            if bad_kind == 'unsupported':
                return ActionProposal(c.observation.id, 'adjust_view', {'camera': 'local', 'angles': [0, 0]})
            if bad_kind == 'zero':
                return ActionProposal(c.observation.id, 'move', {'delta': [0, 0, 0]})
            return ActionProposal(c.observation.id, 'move', {'delta': [0, 0, .001]}, command_id='fixed')
    run, robot = run_fixture(tmp_path, agent=Bad())
    assert run.run().status == 'error'
    assert len(robot.commands) == (1 if bad_kind == 'duplicate' else 0)


def test_capture_failure_does_not_repeat_sent_action(tmp_path):
    class Broken(SimulatedRobot):
        def execute(self, action, cancel, timeout):
            super().execute(action, cancel, timeout)
            return ExecutionResult(action.command_id, True, True, capture_status='failed')
    run, robot = run_fixture(tmp_path, Broken())
    assert run.run().status == 'error'
    assert len(robot.commands) == 1


def test_execution_failure_and_cleanup(tmp_path):
    class Broken(SimulatedRobot):
        def execute(self, *args):
            raise RuntimeError('SDK failure')
    run, robot = run_fixture(tmp_path, Broken())
    assert 'SDK failure' in run.run().error
    assert robot.stopped


def test_cancel_discards_late_decision(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Slow:
        def decide(self, c, cancel):
            entered.set()
            release.wait(1)
            return ActionProposal(c.observation.id, 'move', {'delta': [0, 0, .01]})
    run, robot = run_fixture(tmp_path, agent=Slow())
    results = []
    thread = threading.Thread(target=lambda: results.append(run.run()))
    thread.start(); assert entered.wait(1)
    run.stop(); release.set(); thread.join(2)
    assert results[0].status == 'cancelled'
    assert not robot.commands


def test_budget_and_unknown_completion(tmp_path):
    run, robot = run_fixture(tmp_path, max_rounds=1)
    assert run.run().status == 'budget_exhausted'
    class Done:
        def decide(self, c, cancel):
            return ActionProposal(c.observation.id, 'done')
    run = Harness(SimulatedRobot(), Done(), PressButtonTask('button'), SimulatedVerifier(), Recorder(tmp_path/'done'))
    result = run.run()
    assert result.status == 'unverified' and result.model_reported_done
    assert result.verification.verdict == 'unknown'


def test_old_observation_rejected_after_execution(tmp_path):
    class Stale(SimulatedRobot):
        def observe(self, cancel, timeout):
            if hasattr(self, 'cached'):
                return self.cached
            self.cached = super().observe(cancel, timeout)
            return self.cached
    run, robot = run_fixture(tmp_path, Stale())
    assert run.run().status == 'error'
    assert len(robot.commands) == 1


def test_saved_redecision_has_no_robot_or_counterfactual_transition(tmp_path):
    from elevator_harness import replay_decision
    run, robot = run_fixture(tmp_path)
    assert run.run().status == 'completed'
    rows = [json.loads(line) for line in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    observations = [r['value']['observation'] for r in rows if r['event']=='observation']
    class Check:
        def decide(self, context, cancel):
            assert context.observation.id == observations[2]['id']
            assert len(context.history) == 2
            return ActionProposal(context.observation.id,'move',{'delta':[0,.001,0]})
    old_count = len(robot.commands)
    recorder = Recorder(tmp_path/'replay')
    replay_decision(tmp_path/'run',observations[2]['id'],Check(),recorder)
    assert len(robot.commands)==old_count
    replay = (tmp_path/'replay/events.jsonl').read_text()
    assert 'replay_only' in replay and '"event": "execution"' not in replay

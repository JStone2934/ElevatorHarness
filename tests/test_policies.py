from dataclasses import replace
from threading import Event
import pytest
from elevator_harness import (Context, Evidence, ActionProposal, ExecutionResult, PressConfig,
                             CalibratedPressPolicy, OverlapPolicy)
from elevator_harness.simulation import SimulatedRobot
from elevator_harness.tasks import verify_light_sequence


def fixture():
    robot = SimulatedRobot()
    obs = robot.observe(Event(), 1)
    context = Context(obs, robot.capabilities, 'button', [], {})
    context.evidence = Evidence(obs.id, obs.tool, obs.calibration_version, 'model',
        {'target_box': [.4, .4, .6, .6], 'contact_uv': [.5, .5], 'fully_visible': True,
         'views_conflict': False, 'path_clear': True})
    return robot, context


def transition(policy, robot, context, action):
    assert policy.check(action, context).allowed
    result = robot.execute(action, Event(), 1)
    policy.record(action, context.observation, result)
    context.observation = result.observation
    context.evidence = replace(context.evidence, observation_id=result.observation.id,
                               tool=result.observation.tool)


def test_tool_switch_invalidates_evidence_and_unknown_is_not_permission():
    robot, ctx = fixture()
    move = ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, .01]})
    assert OverlapPolicy().check(move, ctx).allowed
    ctx.observation = replace(ctx.observation, tool='closed')
    assert not OverlapPolicy().check(move, ctx).allowed
    ctx.evidence = None
    assert not OverlapPolicy().check(move, ctx).allowed


def test_freeze_standoff_review_press_withdraw_retry():
    robot, ctx = fixture()
    policy = CalibratedPressPolicy(PressConfig('open', 'closed', .15, .085))
    source = ctx.observation.id
    transition(policy, robot, ctx, ActionProposal(source, 'configure_tool', {'configuration': 'closed'}))
    assert policy.frozen['observation_id'] == source
    assert policy.frozen['from_tool'] == 'open'
    assert not policy.check(ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, .03]}), ctx).allowed
    transition(policy, robot, ctx, ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, .02]}))
    ctx.evidence = replace(ctx.evidence, values=dict(ctx.evidence.values, path_clear=None))
    final = ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, .01]})
    assert not policy.check(final, ctx).allowed
    ctx.evidence = replace(ctx.evidence, values=dict(ctx.evidence.values, path_clear=True))
    transition(policy, robot, ctx, final)
    assert policy.attempts == 1
    opening = ActionProposal(ctx.observation.id, 'configure_tool', {'configuration': 'open'})
    assert not policy.check(opening, ctx).allowed
    transition(policy, robot, ctx, ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, -.01]}))
    transition(policy, robot, ctx, ActionProposal(ctx.observation.id, 'configure_tool', {'configuration': 'open'}))
    assert policy.frozen is None


def test_missing_live_depth_uses_frozen_bound_but_new_calibration_invalidates_it():
    robot, ctx = fixture()
    policy = CalibratedPressPolicy(PressConfig('open', 'closed', .15, .085))
    transition(policy, robot, ctx, ActionProposal(ctx.observation.id, 'configure_tool', {'configuration': 'closed'}))
    ctx.observation = replace(ctx.observation, depth_m=None, depth_valid=False)
    assert policy.check(ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, .01]}), ctx).allowed
    ctx.observation = replace(ctx.observation, calibration_version='new')
    assert not policy.check(ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, .01]}), ctx).allowed
    assert policy.frozen is None


def test_bad_alignment_depth_and_view_conflict_deny_close():
    robot, ctx = fixture()
    policy = CalibratedPressPolicy(PressConfig('open', 'closed', .15, .085))
    close = ActionProposal(ctx.observation.id, 'configure_tool', {'configuration': 'closed'})
    ctx.evidence = replace(ctx.evidence, values=dict(ctx.evidence.values, views_conflict=True))
    assert not policy.check(close, ctx).allowed
    ctx.evidence = replace(ctx.evidence, values=dict(ctx.evidence.values, views_conflict=False))
    ctx.observation = replace(ctx.observation, depth_valid=False)
    assert not policy.check(close, ctx).allowed


def test_overlap_and_model_done_are_not_completion():
    rows = [('a', {'light_on': False}), ('b', {'light_on': True, 'hand_clear': True})]
    assert verify_light_sequence(rows, 'model').verdict == 'unknown'
    rows.append(('c', {'light_on': True, 'hand_clear': True}))
    assert verify_light_sequence(rows, 'model').verdict == 'confirmed'
    rows[0] = ('a', {'light_on': True})
    assert verify_light_sequence(rows, 'model').verdict == 'unknown'


def test_small_final_steps_do_not_prematurely_count_a_press():
    robot, ctx = fixture()
    policy = CalibratedPressPolicy(PressConfig('open', 'closed', .15, .085))
    transition(policy, robot, ctx, ActionProposal(ctx.observation.id, 'configure_tool', {'configuration': 'closed'}))
    for step in (.02, .003, .003):
        transition(policy, robot, ctx, ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, step]}))
        assert not policy.pressed and policy.attempts == 0
    transition(policy, robot, ctx, ActionProposal(ctx.observation.id, 'move', {'delta': [0, 0, .004]}))
    assert policy.pressed and policy.attempts == 1


def test_measured_travel_wins_and_live_depth_cannot_expand_frozen_budget():
    robot, ctx = fixture()
    policy = CalibratedPressPolicy(PressConfig('open', 'closed', .15, .085))
    transition(policy, robot, ctx, ActionProposal(ctx.observation.id, 'configure_tool', {'configuration': 'closed'}))
    policy.commanded = .029  # A command was sent but the measured arm barely moved.
    policy.approached = True
    ctx.observation = replace(ctx.observation, state={'position_m': [.005,0,0]}, depth_m=.9)
    good = ActionProposal(ctx.observation.id, 'move', {'delta': [0,0,.02]})
    assert policy.check(good,ctx).allowed
    assert policy.last_allowance['travel_source']=='measured'
    assert not policy.check(ActionProposal(ctx.observation.id,'move',{'delta':[0,0,.026]}),ctx).allowed

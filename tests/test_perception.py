import base64
import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from threading import Event

from PIL import Image as PILImage
import pytest

from elevator_harness import (ActionProposal, CalibratedPressPolicy, Context, EndToEndAgent,
    Harness, LocalizedAgent, OverlapPolicy, PerceptionConfig, PerceptionPipeline, PressButtonTask,
    PressConfig, Recorder)
from elevator_harness.feedback import compare_geometry
from elevator_harness.perception import crop_content
from elevator_harness.simulation import SimulatedRobot, SimulatedVerifier, synthetic_image
from elevator_harness.types import Image
from elevator_harness.vision import crop_window, restore_box


def text_objects(messages):
    values=[]
    for message in messages:
        if not isinstance(message['content'],list):continue
        for part in message['content']:
            if part['type']=='text':
                try:values.append(json.loads(part['text']))
                except ValueError:pass
    return values


class VisionModel:
    def __init__(self, steps=None):
        self.calls=[]
        self.steps=steps or [{'kind':'observe'}]
        self.index=0
        self.refinement='full'
        self.coarse_units='normalized_01'
        self.tip=[.5,.5]
        self.cancel_refinement=False
        self.press_assessment=True

    def complete(self,messages,schema,cancel):
        name=schema['function']['name']
        self.calls.append((name,copy.deepcopy(messages)))
        values=text_objects(messages)
        observations=[v for v in values if isinstance(v,dict) and 'state' in v and 'observation_id' in v]
        if name=='inspect_panel':
            obs=observations[0]
            factor=1000 if self.coarse_units=='normalized_1000' else 1
            raw=dict(observation_id=obs['observation_id'],target_box=[.4*factor,.4*factor,.6*factor,.6*factor],
                contact_uv=[v*factor for v in self.tip],fully_visible=True,path_clear=True,views_conflict=False,
                light_on=obs['state'].get('light_on'),hand_clear=obs['state'].get('hand_clear'),note='test')
            if 'coordinate_units' in schema['function']['parameters']['properties']:
                raw.update(coordinate_units=self.coarse_units,global_target_box=[.4*factor,.4*factor,.6*factor,.6*factor],
                    global_contact_uv=[.5*factor,.5*factor],target_visibility='full',contact_visibility='full',global_visibility='full')
            return raw
        if name=='locate_button_in_crop':
            if self.cancel_refinement:
                cancel.set()
            if self.refinement=='error':
                raise RuntimeError('temporary model error')
            obs=next(v for v in values if isinstance(v,dict) and 'observation_id' in v)
            return dict(observation_id=obs['observation_id'],coordinate_units='normalized_1000',
                button_visibility=self.refinement,button_box=None if self.refinement=='unknown' else [400,400,600,600])
        if name=='review_progress':
            return dict(observation_id=observations[1]['observation_id'],before_observation_id=observations[0]['observation_id'],
                        progress=None,note='reassess',alternative_explanation='parallax',next_check='check contact path',prediction_check='not_supplied')
        obs=observations[0]
        action=self.steps[min(self.index,len(self.steps)-1)]
        self.index+=1
        response=dict(observation_id=obs['observation_id'],reason='test',action=action)
        if 'press_assessment' in schema['function']['parameters']['properties']:
            response['press_assessment']=dict(observation_id=obs['observation_id'],path_clear=self.press_assessment,
                views_conflict=False,global_view='direct observation',local_view='occluded or uncertain')
        if 'reassessment' in schema['function']['parameters']['properties']:
            response['reassessment']=dict(observation_id=obs['observation_id'],hypothesis='parallax',expected_visual_change='check alignment',if_no_change='reobserve')
        return response


def context(robot=None):
    robot=robot or SimulatedRobot()
    obs=robot.observe(Event(),1)
    return Context(obs,robot.capabilities,'test button',[],{}),robot


def test_crop_geometry_boundaries_and_mapping():
    window=crop_window([.4,.4,.6,.6],[320,180],3)
    assert window==[70,0,250,180]
    assert restore_box([.4,.4,.6,.6],window,[320,180])==pytest.approx([.44375,.4,.55625,.6])
    for box in ([0,0,.01,.01],[.99,.99,1,1],[.5,.5,.500001,.500001]):
        x0,y0,x1,y1=crop_window(box,[320,180],3)
        assert 0<=x0<x1<=320 and 0<=y0<y1<=180
    with pytest.raises(ValueError):restore_box([0,0,1,1],[-1,0,100,100],[320,180])


@pytest.mark.parametrize('units',['normalized_01','normalized_1000'])
def test_refined_box_drives_same_decision_and_gate_not_coarse_box(tmp_path,units):
    model=VisionModel();model.tip=[.43,.5];model.coarse_units=units
    ctx,robot=context()
    recorder=Recorder(tmp_path/'record')
    agent=LocalizedAgent(model,recorder=recorder)
    action=agent.decide(ctx,Event())
    assert [x[0] for x in model.calls]==['inspect_panel','locate_button_in_crop','choose_action']
    evidence=ctx.evidence.values
    assert evidence['coarse_localization']['target_box']==[.4,.4,.6,.6]
    assert evidence['target_box']==pytest.approx([.44375,.4,.55625,.6])
    decision=text_objects(model.calls[-1][1])[0]
    assert decision['evidence']['values']['target_box']==evidence['target_box']
    assert not OverlapPolicy().check(ActionProposal(ctx.observation.id,'move',{'delta':[0,0,.001]}),ctx).allowed
    refine=model.calls[1][1][-1]['content'][-1]['image_url']['url']
    with PILImage.open(io.BytesIO(base64.b64decode(refine.split(',',1)[1]))) as image:
        assert image.size==(768,768)
    events=[json.loads(x) for x in (tmp_path/'record/events.jsonl').read_text().splitlines()]
    crop=next(e['value'] for e in events if e['event']=='local_refinement_crop')
    assert crop['lineage']['observation_id']==ctx.observation.id
    assert (tmp_path/'record'/crop['path']).is_file()
    agent.decide(ctx,Event())
    assert [x[0] for x in model.calls].count('locate_button_in_crop')==1


@pytest.mark.parametrize('result,status',[('error','failed_kept_coarse'),('unknown','no_box_kept_coarse'),('partial','refined')])
def test_refinement_fallback_and_partial_visibility(result,status):
    model=VisionModel();model.refinement=result
    ctx,_=context()
    pipeline=PerceptionPipeline(model)
    ctx.evidence=pipeline.localize(ctx,Event())
    assert ctx.evidence.values['refinement']['status']==status
    if result=='partial':
        assert not ctx.evidence.values['fully_visible']
        assert not OverlapPolicy().check(ActionProposal(ctx.observation.id,'move',{'delta':[0,0,.001]}),ctx).allowed
    else:
        assert ctx.evidence.values['target_box']==[.4,.4,.6,.6]


def test_cancel_during_refinement_does_not_fall_back_to_action():
    model=VisionModel();model.cancel_refinement=True
    ctx,_=context()
    with pytest.raises(InterruptedError):LocalizedAgent(model).decide(ctx,Event())
    assert 'choose_action' not in [x[0] for x in model.calls]


def test_current_global_image_crop_keeps_old_box_only_as_anchor(tmp_path):
    ctx,robot=context();model=VisionModel();pipeline=PerceptionPipeline(model,recorder=Recorder(tmp_path/'r'))
    pipeline.localize(ctx,Event())
    current=replace(ctx.observation,id='closed-current',tool='closed',
        images={'global':Image(synthetic_image(True),'image/png'),'local':ctx.observation.images['local']})
    crop,record=pipeline.global_zoom(current,robot.capabilities,[])
    assert record['status']=='applied'
    assert crop.lineage['observation_id']=='closed-current'
    assert crop.lineage['box_observation_id']==ctx.observation.id
    assert crop.lineage['source_sha256']==hashlib.sha256(current.images['global'].data).hexdigest()
    assert crop.lineage['source_sha256']!=hashlib.sha256(ctx.observation.images['global'].data).hexdigest()
    assert crop.lineage['scale']==6.


@pytest.mark.parametrize('change,expected',[
    ('angle','skipped_view_moved'),('missing_pose','skipped_view_pose_unknown'),
    ('revision','skipped_calibration_changed'),('geometry','skipped_image_geometry_changed'),
    ('pose_revision','skipped_view_pose_revision'),('adjusted','skipped_view_adjusted')])
def test_stale_crop_anchor_is_not_reused(change,expected):
    ctx,robot=context();pipeline=PerceptionPipeline(VisionModel());pipeline.localize(ctx,Event())
    now=replace(ctx.observation,id='current',tool='closed',state=copy.deepcopy(ctx.observation.state))
    history=[]
    if change=='angle':now.state['view_angles_rad']['global']=[.1,0]
    if change=='missing_pose':now.state.pop('view_angles_rad')
    if change=='revision':now=replace(now,calibration_version='new')
    if change=='pose_revision':now.state['view_pose_revision']={'global':2}
    if change=='geometry':
        stream=io.BytesIO();PILImage.new('RGB',(100,100)).save(stream,'PNG')
        now=replace(now,images=dict(now.images,global_placeholder=None))
        now.images.pop('global_placeholder');now.images['global']=Image(stream.getvalue(),'image/png')
    if change=='adjusted':history=[{'before_id':ctx.observation.id,'action':{'kind':'adjust_view','arguments':{'camera':'global'}}}]
    crop,record=pipeline.global_zoom(now,robot.capabilities,history)
    assert crop is None and record['status']==expected


def test_static_camera_needs_no_head_angles_and_missing_anchor_skips():
    ctx,robot=context(SimulatedRobot(movable_view=False));ctx.observation.state.pop('view_angles_rad')
    pipeline=PerceptionPipeline(VisionModel())
    assert pipeline.global_zoom(ctx.observation,robot.capabilities,[])[1]['status']=='skipped_no_prior_global_box'
    pipeline.localize(ctx,Event())
    assert pipeline.global_zoom(replace(ctx.observation,id='new'),robot.capabilities,[])[1]['status']=='applied'


def test_direct_press_then_reopen_restores_refinement_and_review_memory(tmp_path):
    steps=[{'kind':'configure_tool','configuration':'closed'}, {'kind':'move','delta':[0,0,.02]},
           *[{'kind':'move','delta':[.001,0,0]} for _ in range(3)],
           {'kind':'observe'}, {'kind':'move','delta':[0,0,.01]}, {'kind':'move','delta':[0,0,-.01]},
           {'kind':'configure_tool','configuration':'open'}, {'kind':'move','delta':[0,.002,0]},
           {'kind':'configure_tool','configuration':'closed'}, {'kind':'unable'}]
    model=VisionModel(steps);robot=SimulatedRobot();recorder=Recorder(tmp_path/'run')
    agent=LocalizedAgent(model,recorder=recorder)
    run=Harness(robot,agent,PressButtonTask('button'),SimulatedVerifier(),recorder,
                CalibratedPressPolicy(PressConfig('open','closed',.15,.085)))
    result=run.run()
    assert result.status=='unable',result
    names=[x[0] for x in model.calls]
    assert names.count('inspect_panel')==3  # initial, reopened, realigned
    assert names.count('locate_button_in_crop')==3
    assert names.count('review_progress')==1  # observation-only round cannot re-review old motion
    review_messages=next(m for n,m in model.calls if n=='review_progress')
    review_text=json.dumps(review_messages)
    assert 'BEFORE global crop' in review_text and 'CURRENT global crop' in review_text
    actions=[m for n,m in model.calls if n=='choose_action']
    direct=text_objects(actions[1])[0]
    assert direct['decision_context']['stage']=='direct_press' and direct['evidence'] is None
    assert 'press_global_crop' in (tmp_path/'run/events.jsonl').read_text()
    attempts=agent.memory.attempts
    assert len(attempts)==2 and attempts[0]['reopened']
    assert attempts[0]['cumulative_uv_m']==pytest.approx([.003,0])
    assert attempts[0]['deepest_press_id']
    assert 'HISTORICAL EXECUTED PRESS' in json.dumps(actions[-1])
    assert any(text_objects(m)[0]['decision_context']['press_memory']['last_adjustment'] for m in actions)


def test_direct_press_assessment_is_bound_and_gates_final_motion(tmp_path):
    model=VisionModel([{'kind':'configure_tool','configuration':'closed'},
                       {'kind':'move','delta':[0,0,.02]}, {'kind':'move','delta':[0,0,.01]}])
    model.press_assessment=False
    robot=SimulatedRobot();recorder=Recorder(tmp_path/'r')
    run=Harness(robot,LocalizedAgent(model),PressButtonTask('button'),SimulatedVerifier(),recorder,
                CalibratedPressPolicy(PressConfig('open','closed',.15,.085)))
    assert run.run().status=='error'
    assert len(robot.commands)==2  # close and standoff approach only
    assert 'fresh global/local path review required' in (tmp_path/'r/events.jsonl').read_text()


def test_scale_changes_do_not_count_as_improvement():
    def row(box):
        return {'tool':'open','calibration_version':'v1',
                'values':{'target_box':box,'contact_uv':[.8,.5],'fully_visible':True}}
    assert compare_geometry(row([.4,.4,.6,.6]),row([.2,.2,.8,.8]))['reason']=='box_scale_changed'


def test_pipeline_can_disable_crops_and_switching():
    model=VisionModel();ctx,_=context()
    agent=LocalizedAgent(model,perception=PerceptionConfig(refine_local=False,press_zoom=False,direct_press=False))
    agent.decide(ctx,Event())
    assert [n for n,_ in model.calls]==['inspect_panel','choose_action']


def test_review_requires_current_prediction_in_action_response():
    class BadPrediction(VisionModel):
        def complete(self,messages,schema,cancel):
            result=super().complete(messages,schema,cancel)
            if 'reassessment' in result:
                result['reassessment']['observation_id']='old'
            return result
    ctx,_=context()
    ctx.decision_data={'independent_review': {'note':'check new evidence'}}
    with pytest.raises(ValueError,match='current hypothesis'):
        EndToEndAgent(BadPrediction()).decide(ctx,Event())


def test_corrupt_refinement_coordinates_cannot_replace_coarse_result():
    class BadUnits(VisionModel):
        def complete(self,messages,schema,cancel):
            result=super().complete(messages,schema,cancel)
            if schema['function']['name']=='locate_button_in_crop':
                result['coordinate_units']='pixels'
            return result
    ctx,_=context()
    result=PerceptionPipeline(BadUnits()).localize(ctx,Event())
    assert result.values['target_box']==[.4,.4,.6,.6]
    assert result.values['refinement']['status']=='failed_kept_coarse'


def test_allowance_preparation_uses_current_observation_before_model():
    robot=SimulatedRobot()
    ctx,_=context(robot)
    policy=CalibratedPressPolicy(PressConfig('open','closed',.15,.085))
    ctx.evidence=PerceptionPipeline(VisionModel()).localize(ctx,Event())
    close=ActionProposal(ctx.observation.id,'configure_tool',{'configuration':'closed'})
    assert policy.check(close,ctx).allowed
    result=robot.execute(close,Event(),1);policy.record(close,ctx.observation,result)
    ctx.observation=replace(result.observation,depth_m=.17)
    policy.prepare(ctx)
    assert policy.describe()['last_allowance']['observation_id']==ctx.observation.id
    assert policy.describe()['last_allowance']['remaining_m']==pytest.approx(.02)


def test_null_view_metadata_skips_zoom_without_failing_decision():
    ctx,robot=context();pipeline=PerceptionPipeline(VisionModel());pipeline.localize(ctx,Event())
    current=replace(ctx.observation,id='unknown-view',state=dict(ctx.observation.state,view_angles_rad=None))
    assert pipeline.global_zoom(current,robot.capabilities,[])[1]['status']=='skipped_view_pose_unknown'

import base64


import copy


import io


import json


import threading


import time


from pathlib import Path


import numpy as np


import pytest


from PIL import Image


from examples.s1.common import pose_error, request


from examples.s1.service import Service, make_server


class FakeBackend:
    error=None
    def __init__(self):
        self.pose=[.4,-.2,1.,0.,0.,0.,1.];self.calls=[];self.stopped=False;self.block=None
    def snapshot(self):
        return dict(pose_m_xyzw=list(self.pose),receipt_ns=time.monotonic_ns(),joints={})
    def move(self,kind,data,cancel,phase):
        self.calls.append((kind,data));phase('MOVING' if kind=='move' else 'EXTENDING')
        if self.block:
            while not self.block.is_set():
                if cancel.wait(.01):raise RuntimeError('stopped')
        if kind=='move':self.pose[:3]=[v+d/1000 for v,d in zip(self.pose[:3],data['xyz_mm'])]
        return dict(returned_to_start=True if kind=='press' else None)
    def stop(self):self.stopped=True
    def close(self):self.stop()


class FakeCamera:
    def __init__(self):self.revision=0;self.fail=False
    def selection(self):return dict(revision=self.revision,point_normalized=None)
    def capture(self,after,cancel):
        if self.fail:raise ValueError('missing_depth')
        image=io.BytesIO();Image.new('RGB',(16,16),'red').save(image,format='JPEG')
        now=time.monotonic_ns()
        return dict(frames={k:dict(data_base64=base64.b64encode(v).decode(),source_ns=time.time_ns(),receipt_ns=now)
                    for k,v in dict(head=image.getvalue(),right=image.getvalue(),depth=np.full((2,2),1800,dtype='<u2').tobytes()).items()},
            depth=dict(median_m=.18,center_m=.18,valid_pixels=25),
            metadata=dict(depth=dict(height=2,width=2),scale_m=.0001),selection=self.selection())


@pytest.fixture
def setup(tmp_path):
    backend=FakeBackend();camera=FakeCamera();service=Service(backend,camera,tmp_path/'robot')
    yield service,backend,camera
    service.close()


def wait(service,receipt):
    service.worker.join(3)
    assert not service.worker.is_alive()
    return service.task(receipt['task_id'])


def observe(service):
    task=wait(service,service.submit(dict(session_id=service.session_id,command_id=str(time.time_ns())),True))
    assert task['status']=='completed',task
    return service.observations[task['capture']['observation_id']]


def action(service,obs,cid='cmd',**kwargs):
    return dict(session_id=service.session_id,observation_id=obs['observation_id'],command_id=cid,**kwargs)


def test_task_polling_never_sees_a_task_while_the_worker_mutates_it(tmp_path):
    """2026-11-04 run ended on HTTP 503 'dictionary changed size during iteration':
    the client polls /v2/tasks while the worker updated the task outside the lock."""
    class BigMotionBackend(FakeBackend):
        def move(self,kind,data,cancel,phase):
            super().move(kind,data,cancel,phase)
            return {f'field_{n}':n for n in range(3000)}
    service=Service(BigMotionBackend(),FakeCamera(),tmp_path/'robot')
    errors,done=[],threading.Event()
    try:
        obs=observe(service)
        tid=[None]
        def poll():
            while not done.is_set():
                if tid[0]:
                    try:service.task(tid[0])
                    except Exception as exc:errors.append(repr(exc));return
        pollers=[threading.Thread(target=poll) for _ in range(4)]
        for p in pollers:p.start()
        for n in range(15):
            receipt=service.submit(action(service,obs,f'c{n}',type='move',xyz_mm=[0,0,0]))
            tid[0]=receipt['task_id']
            service.worker.join(30)        # pollers contend for the lock; allow time under load
            task=wait(service,receipt)
            assert task['status']=='completed',task
            obs=service.observations[task['capture']['observation_id']]
        done.set()
        for p in pollers:p.join(3)
    finally:
        done.set();service.close()
    assert errors==[]


def tool(name,args):
    return dict(choices=[dict(finish_reason='tool_calls',message=dict(content=None,tool_calls=[
        dict(function=dict(name=name,arguments=json.dumps(args)))]))])


def test_move_deduplication_and_fresh_observation(setup):
    service,backend,camera=setup;obs=observe(service)
    body=action(service,obs,type='move',xyz_mm=[0,-10,-5])
    receipt=service.submit(body);result=wait(service,receipt)
    assert result['status']=='completed'
    assert backend.pose[:3]==pytest.approx([.4,-.21,.995])
    assert service.submit(body)==receipt and len(backend.calls)==1
    with pytest.raises(ValueError,match='conflict'):service.submit(dict(body,xyz_mm=[0,-2,-5]))
    with pytest.raises(ValueError,match='old_observation'):service.submit(dict(body,command_id='next'))


def test_xyz_persists_and_legacy_press_is_rejected(setup):
    service,backend,_=setup;obs=observe(service)
    with pytest.raises(ValueError,match='invalid_command'):
        service.submit(action(service,obs,type='press',forward_mm=15))
    result=wait(service,service.submit(action(service,obs,type='move',xyz_mm=[45,20,0])))
    assert result['motion']['action_completed']
    assert backend.pose[:3]==pytest.approx([.445,-.18,1.])
    assert result['motion']['button_trigger_evidence'] is None
    obs=service.observations[result['capture']['observation_id']]
    wait(service,service.submit(action(service,obs,cid='retract',type='move',xyz_mm=[-100,0,0])))
    assert backend.pose[0]==pytest.approx(.345)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,'3',[]])
def test_invalid_move(setup,value):
    service,_,_=setup;obs=observe(service)
    with pytest.raises(ValueError):service.submit(action(service,obs,type='move',xyz_mm=[value,0,0]))


def test_stop_priority_and_busy(setup):
    service,backend,_=setup;obs=observe(service);backend.block=threading.Event()
    receipt=service.submit(action(service,obs,type='move',xyz_mm=[0,-1,0]))
    with pytest.raises(ValueError,match='not_ready'):service.submit(action(service,obs,cid='other',type='move',xyz_mm=[0,-1,0]))
    service.stop();result=wait(service,receipt)
    assert backend.stopped and result['status']=='interrupted'
    with pytest.raises(ValueError):service.submit(action(service,obs,cid='after-stop',type='move',xyz_mm=[0,-1,0]))


def test_capture_failure_does_not_replay_motion(setup):
    service,backend,camera=setup;obs=observe(service);camera.fail=True
    body=action(service,obs,type='move',xyz_mm=[0,-1,0])
    receipt=service.submit(body);result=wait(service,receipt)
    assert result['status']=='failed' and result['motion']['action_completed']
    assert service.phase=='NEEDS_OBSERVATION'
    assert service.submit(body)==receipt and len(backend.calls)==1
    camera.fail=False;observe(service)
    assert service.phase=='READY' and len(backend.calls)==1


def test_session_observation_tokens_and_depth_is_not_motion_gate(setup):
    service,backend,camera=setup;obs=observe(service)
    body=action(service,obs,type='move',xyz_mm=[1,0,0])
    with pytest.raises(ValueError,match='session'):service.submit(dict(body,session_id='old'))
    with pytest.raises(ValueError,match='old_observation'):service.submit(dict(body,observation_id='old'))
    backend.pose[1]+=.01
    camera.revision+=1
    obs['depth']['median_m']=None
    assert wait(service,service.submit(body))['status']=='completed'


class SkewBackend(FakeBackend):
    """state_for_capture raises image_state_skew for the next ``skews`` captures."""
    def __init__(self, skews):
        super().__init__();self.skews=skews
    def state_for_capture(self, bundle):
        if self.skews:
            self.skews-=1;raise RuntimeError('image_state_skew')
        return self.snapshot()


def _serve(service):
    server=make_server(service,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server,thread


def test_service_retakes_a_skewed_photo_after_motion(tmp_path):
    """2026-11-04 run ended on task_failed:image_state_skew after a completed -20 mm move."""
    backend=SkewBackend(0);service=Service(backend,FakeCamera(),tmp_path/'robot')
    try:
        obs=observe(service)
        backend.skews=2
        task=wait(service,service.submit(action(service,obs,'skew',type='move',xyz_mm=[-20,0,0])))
        assert task['status']=='completed' and task['capture_retries']==['image_state_skew']*2
    finally:service.close()


def test_restart_journal_not_replayed(tmp_path):
    root=tmp_path/'sessions';old=root/'old'/'tasks';old.mkdir(parents=True)
    (old/'task.json').write_text(json.dumps(dict(status='running')))
    service=Service(FakeBackend(),FakeCamera(),root)
    try:assert json.loads((old/'task.json').read_text())['status']=='interrupted'
    finally:service.close()


def test_restart_does_not_read_archived_tasks_after_migration(tmp_path):
    root=tmp_path/'sessions';old=root/'old'/'tasks';old.mkdir(parents=True)
    archive=old/'task.json'
    archive.write_text(json.dumps(dict(status='completed')))
    service=Service(FakeBackend(),FakeCamera(),root)
    service.close()
    archive.write_text('must not be parsed again')
    service=Service(FakeBackend(),FakeCamera(),root)
    try:assert json.loads(service.pending_path.read_text()) is None
    finally:service.close()


def test_restart_recovers_only_pending_task(tmp_path):
    root=tmp_path/'sessions';old=root/'old'/'tasks';old.mkdir(parents=True)
    (old/'task.json').write_text(json.dumps(dict(status='running')))
    (old/'archive.json').write_text('must not be parsed')
    (root/'.pending_task.json').write_text(json.dumps('old/tasks/task.json'))
    service=Service(FakeBackend(),FakeCamera(),root)
    try:
        assert json.loads((old/'task.json').read_text())['status']=='interrupted'
        assert json.loads(service.pending_path.read_text()) is None
    finally:service.close()


@pytest.mark.parametrize('capture_fails',[False,True])
def test_pending_task_tracks_admission_and_completion(setup,capture_fails):
    service,backend,camera=setup;obs=observe(service)
    backend.block=threading.Event();camera.fail=capture_fails
    receipt=service.submit(action(service,obs,type='move',xyz_mm=[0,1,0]))
    assert json.loads(service.pending_path.read_text())==f'{service.session_id}/tasks/{receipt["task_id"]}.json'
    backend.block.set()
    result=wait(service,receipt)
    assert result['status']==('failed' if capture_fails else 'completed')
    assert json.loads(service.pending_path.read_text()) is None


def test_pose_quaternion_sign():
    assert pose_error([0,0,0,0,0,0,1],[0,0,0,0,0,0,-1])==(0,0)

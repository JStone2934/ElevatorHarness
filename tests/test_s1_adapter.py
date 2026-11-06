import copy
import json
from pathlib import Path
from threading import Event, Thread
import time

import pytest
from elevator_harness import ActionProposal
from examples.s1.configuration import load_settings, configuration_hash
from examples.s1.adapter import S1Adapter
from examples.s1.service import Service, make_server
from test_s1_service import FakeBackend, FakeCamera


def settings_file(tmp_path):
    settings=json.loads(Path('examples/s1/config.template.json').read_text())
    settings['depth_point']=[.5,.5]
    settings['tools'][1]['local_contact_uv']=[.5,.5]
    settings['press']['contact_depth_m']=.15
    p=tmp_path/'settings.json';p.write_text(json.dumps(settings))
    return p,load_settings(p)


def test_template_rejected_before_sdk_import():
    with pytest.raises(ValueError,match='depth_point'):
        load_settings('examples/s1/config.template.json')


def test_s1_http_adapter_observations_units_config_and_idempotency(tmp_path):
    path,settings=settings_file(tmp_path)
    class Backend(FakeBackend):
        gripper=0
        def snapshot(self):
            return dict(super().snapshot(),joints={'astribot_gripper_right':{'position':[self.gripper]}})
        def move(self,kind,data,cancel,phase):
            value=super().move(kind,data,cancel,phase)
            if kind=='gripper':self.gripper=97 if data['key']=='c' else 0
            return value
    class Camera(FakeCamera):
        def selection(self):return dict(revision=0,point_normalized=[.5,.5])
        def capture(self,after,cancel):return dict(super().capture(after,cancel),selection=self.selection())
    backend=Backend();service=Service(backend,Camera(),tmp_path/'robot')
    service.session['configuration_hash']=configuration_hash(settings)
    server=make_server(service,0);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    adapter=None
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        bad=copy.deepcopy(settings);bad['depth_point']=[.1,.1]
        with pytest.raises(ValueError,match='mismatch'):S1Adapter(base,bad)
        adapter=S1Adapter(base,settings)
        obs=adapter.observe(Event(),2)
        assert obs.tool=='open' and obs.depth_valid
        assert set(obs.images)=={'global','local'}
        action=ActionProposal(obs.id,'move',{'delta':[.01,.02,.03]})
        result=adapter.execute(action,Event(),2)
        assert result.sent and result.observation.id != obs.id
        assert backend.calls[0][1]['xyz_mm']==[30.,10.,20.]
        assert adapter.execute(action,Event(),2)==result
        assert len(backend.calls)==1
        close=ActionProposal(result.observation.id,'configure_tool',{'configuration':'closed'})
        result=adapter.execute(close,Event(),2)
        assert result.observation.tool=='closed'
        assert backend.calls[-1][1]['key']=='c'
    finally:
        if adapter:adapter.stop()
        server.shutdown();server.server_close();thread.join(2);service.close()


def test_robot_local_startup_and_cleanup_with_fake_sdk(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from examples.s1 import hardware, connection, sdk_backend, robot_state, service as service_module
    config, settings = settings_file(tmp_path)
    events=[]
    class Robot:
        def __init__(self, **kw):
            events.append('acquire')
            self.astribot_interface=SimpleNamespace(
                shutdown_control_rights_srv=lambda:events.append('release'),
                executor=SimpleNamespace(shutdown=lambda **kw:events.append('executor_shutdown')),
                spin_thread=SimpleNamespace(join=lambda **kw:events.append('join')),
                shutdown=lambda:events.append('interface_shutdown'))
        def get_control_rights_status(self):return True
        def set_head_follow_effector(self, value):events.append('head_independent')
        def stop_robot(self):events.append('stop_robot')
    class Monitor:
        def __init__(self,*a,**kw):pass
        def snapshot(self):return {}
        def close(self):events.append('monitor_close')
    class Backend:
        error=None
        def __init__(self,*a):pass
        def close(self):events.append('backend_close')
    class Service:
        phase='READY'
        def __init__(self, backend, camera, root):
            self.session={};self.backend=backend;self.shutdown_requested=Event();self.shutdown_requested.set()
        def close(self):self.backend.close();events.append('service_close')
        def stop(self):self.shutdown_requested.set()
    api=ModuleType('astribot_sdk.core.astribot_api.astribot_client');api.Astribot=Robot
    for name in ['astribot_sdk','astribot_sdk.core','astribot_sdk.core.astribot_api']:
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    monkeypatch.setitem(sys.modules,api.__name__,api)
    ros=ModuleType('rclpy');ros.ok=lambda:True;ros.shutdown=lambda:events.append('ros_shutdown')
    monkeypatch.setitem(sys.modules,'rclpy',ros)
    monkeypatch.setattr(robot_state,'RobotState',Monitor)
    monkeypatch.setattr(sdk_backend,'SDKBackend',Backend)
    monkeypatch.setattr(service_module,'Service',Service)
    monkeypatch.setattr(service_module,'make_server',lambda *a:SimpleNamespace(server_close=lambda:events.append('server_close')))
    monkeypatch.setattr(connection,'connect_robot',lambda factory,*a,**kw:factory())
    monkeypatch.setattr(hardware,'wait_for_viewer',lambda *a:events.append('camera_ready'))
    process=SimpleNamespace(terminate=lambda:events.append('camera_stop'),wait=lambda *a:None)
    monkeypatch.setattr(hardware.subprocess,'Popen',lambda *a,**kw:process)
    monkeypatch.setattr(hardware.signal,'signal',lambda *a:None)
    args=SimpleNamespace(config=str(config),output=str(tmp_path/'service'),view_port=8765,port=6210,
                         control_hz=250,speed_mm_s=20,head_speed_deg_s=20,auto_confirm=False)
    hardware.run(args)
    assert events.index('camera_ready') < events.index('acquire')
    assert events.index('executor_shutdown') < events.index('ros_shutdown') < events.index('interface_shutdown')
    assert {'server_close','service_close','backend_close','monitor_close','release','camera_stop'} <= set(events)
    point=json.loads((tmp_path/'service/depth_point.json').read_text())
    from examples.s1.camera import Frames
    assert Frames(tmp_path/'service/depth_point.json').selection()['point_normalized']==[.5,.5]

"""Explicit robot-local launch. Hardware imports occur inside run()."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen
from .common import atomic_json
from .configuration import load_settings, configuration_hash
_SDK_KEEPALIVE = []


def wait_for_viewer(process, port, timeout=30.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('camera process exited')
        try:
            with urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                if response.status == 200 and json.load(response).get('pid') == process.pid:
                    return
        except OSError:
            time.sleep(.1)
    raise RuntimeError('camera service startup 超时')


def run(args):
    from .connection import connect_robot
    from .hardware import wait_for_viewer, _SDK_KEEPALIVE
    from .sdk_backend import SDKBackend
    from .service import Camera,Service,make_server
    settings=load_settings(args.config)
    viewer=robot=monitor=backend=service=server=None
    done=False
    def stop(*unused):
        nonlocal done
        done=True
        if service:service.stop()
        elif backend:backend.stop()
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    log=None
    try:
        out=Path(args.output).expanduser();out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'depth_point.json', {'point_normalized': settings['depth_point'], 'schema': 's1b-view-point-v1', 'camera': 'right_wrist_rgbd'})
        log=(out/'camera_view.log').open('ab')
        viewer=subprocess.Popen([sys.executable, '-m', 'examples.s1.camera',
                                 '--port',str(args.view_port), '--settings', args.config,
                                 '--point-config', str(out/'depth_point.json')],stdout=log,stderr=log)
        wait_for_viewer(viewer,args.view_port)
        print(f'相机网页已就绪：http://127.0.0.1:{args.view_port}',flush=True)
        from astribot_sdk.core.astribot_api.astribot_client import Astribot
        from .robot_state import RobotState
        robot=connect_robot(Astribot,args.control_hz,
                            auto_confirm=getattr(args, 'auto_confirm', False));_SDK_KEEPALIVE.append(robot)
        if not robot.get_control_rights_status():raise RuntimeError('未取得 SDK 控制权')
        print('运动模式：共用键盘执行器，XYZ / h / c / o；q 回车或 Ctrl-C 停止。',flush=True)
        robot.set_head_follow_effector(False)
        monitor=RobotState(lambda row:None,shared_context=True,control_only=True)
        deadline=time.monotonic()+5
        while True:
            try:monitor.snapshot();break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(.05)
        if done:
            raise InterruptedError('startup_stopped')
        backend=SDKBackend(robot,monitor,args.control_hz,args.speed_mm_s,args.head_speed_deg_s)
        service=Service(backend,Camera(f'http://127.0.0.1:{args.view_port}'),out)
        service.session['configuration_hash']=configuration_hash(settings)
        server=make_server(service,args.port);server.timeout=.1
        print(f'Agent API 已就绪：http://127.0.0.1:{args.port}/v2/session\n默认 20 mm/s；在另一终端启动 examples.s1.run。',flush=True)
        fault_saved=False
        while not done and not service.shutdown_requested.is_set():
            if viewer.poll() is not None:raise RuntimeError('相机服务退出，详情见 '+str(out/'camera_view.log'))
            if backend.error:service.backend_failed()
            if service.phase=='FAULT' and not fault_saved:
                diagnostic=backend.diagnostic()
                diagnostic['error']=backend.error or service.fault
                path=service.root/'fault.json'
                atomic_json(path,diagnostic);fault_saved=True
                print('运动已停止：'+str(diagnostic['error'])+'\n完整故障记录：'+str(path)+
                      '\nAPI 保留故障查询；客户端停止或 Ctrl-C 后退出。',file=sys.stderr,flush=True)
            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                line=sys.stdin.readline()
                if not line or line.strip().lower()=='q':
                    stop();break
            server.handle_request()
    except Exception as exc:
        diagnostic=dict(error=str(exc),traceback=traceback.format_exc(),
            servo_traceback=getattr(backend,'error_traceback',None),
            torso=getattr(backend,'torso_diagnostics',{}))
        path=(service.root if service else Path(args.output).expanduser())/'fault.json'
        try:atomic_json(path,diagnostic)
        except OSError:pass
        print('完整故障记录：'+str(path),file=sys.stderr,flush=True)
        raise
    finally:
        failures=[]
        if backend is not None and backend.error:
            path=(service.root if service else Path(args.output).expanduser())/'fault.json'
            try:atomic_json(path,backend.diagnostic())
            except OSError as exc:failures.append('fault log: '+str(exc))
        for close in (server.server_close if server else None,service.close if service else backend.close if backend else None,
                      monitor.close if monitor else None):
            if close:
                try:close()
                except Exception as exc:failures.append(str(exc))
        if robot:
            try:
                if robot.get_control_rights_status():
                    robot.stop_robot();robot.astribot_interface.shutdown_control_rights_srv()
                interface=robot.astribot_interface
                interface.executor.shutdown(timeout_sec=2)
                import rclpy
                if rclpy.ok():rclpy.shutdown()
                interface.spin_thread.join(timeout=2);interface.shutdown()
            except Exception as exc:failures.append(str(exc))
        if viewer:
            viewer.terminate()
            try:viewer.wait(4)
            except subprocess.TimeoutExpired:viewer.kill();viewer.wait()
        if log:log.close()
        if failures:print('退出清理：'+'; '.join(failures),file=sys.stderr)

#!/usr/bin/env python3
"""Local camera webpage with a persistent measurement point; no robot commands."""
import os
# The viewer shares the robot CPU with the 250 Hz SDK control loop. Unbounded BLAS/OpenMP
# threads in the depth reprojection starved it (SDK frequency warnings, stale captures).
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_name,'1')

import argparse
import base64
from collections import deque
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from pathlib import Path
import signal
import subprocess
import threading
import time
import tempfile

import cv2
import numpy as np

cv2.setNumThreads(1)

TOPICS = {'head':'/astribot_camera/head_rgbd/color_compress/compressed',
          'right':'/astribot_camera/right_wrist_rgbd/color_compress/compressed',
          'depth':'/astribot_camera/right_wrist_rgbd/depth_compress'}


def validate_point(point):
    if point is None:return None
    if (not isinstance(point,list) or len(point)!=2
            or any(type(v) not in (float,int) or not np.isfinite(v) or not 0<=v<=1 for v in point)):
        raise ValueError('测距点必须是 0–1 范围内的二维坐标')
    return [float(v) for v in point]


def point_pixel(point,width,height):
    if point is None:return width//2,height//2
    return tuple(int(np.floor(v*(size-1)+.5)) for v,size in zip(validate_point(point),(width,height)))


def matrix(i):
    return np.array([[i['fx'],0,i['ppx']],[0,i['fy'],i['ppy']],[0,0,1.]],float)


def rays(i):
    yy,xx=np.indices((i['height'],i['width']))
    pixels=np.stack([xx,yy],axis=-1).astype(float).reshape(-1,1,2)
    k=matrix(i);coeff=np.asarray(i['coeffs'],float)
    if i['model'] in (0,4):
        xy=cv2.undistortPoints(pixels,k,coeff if i['model']==4 else np.zeros(5)).reshape(-1,2)
    elif i['model']==2:  # inverse Brown: observed normalized coordinates -> ideal ray
        xy=(pixels.reshape(-1,2)-[i['ppx'],i['ppy']])/[i['fx'],i['fy']]
        xyz=np.column_stack([xy,np.ones(len(xy))])
        xy=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),np.eye(3),coeff)[0].reshape(-1,2)
    else:
        raise ValueError('unsupported_factory_distortion_model')
    return np.column_stack([xy,np.ones(len(xy))])


def project(points,i):
    if i['model'] in (0,4):
        return cv2.projectPoints(points,np.zeros(3),np.zeros(3),matrix(i),
                                 np.asarray(i['coeffs']) if i['model']==4 else np.zeros(5))[0].reshape(-1,2)
    if i['model']==2:
        ideal=(points[:,:2]/points[:,2:3]).reshape(-1,1,2)
        observed=cv2.undistortPoints(ideal,np.eye(3),np.asarray(i['coeffs'])).reshape(-1,2)
        return observed*[i['fx'],i['fy']]+[i['ppx'],i['ppy']]
    raise ValueError('unsupported_factory_distortion_model')


class CenterDepth:
    def __init__(self, metadata):
        self.m=metadata
        self.rays=rays(metadata['depth'])
        self.rotation=np.asarray(metadata['rotation_column_major']).reshape(3,3,order='F')
        self.translation=np.asarray(metadata['translation_m'])
        self.scale=float(metadata['scale_m'])
        if not np.isfinite(self.scale) or self.scale<=0:
            raise ValueError('invalid_device_depth_units')

    def measure(self,data,color_shape,point=None):
        d,c=self.m['depth'],self.m['color']
        if tuple(color_shape[:2]) != (c['height'],c['width']):
            raise ValueError('RGB_resolution_does_not_match_factory_profile')
        if len(data)!=d['width']*d['height']*2:
            raise ValueError('raw_depth_size_mismatch')
        z=np.frombuffer(data,dtype='<u2').astype(float)*self.scale
        valid=(z>0)&np.isfinite(z)
        points=(self.rays[valid]*z[valid,None])@self.rotation.T+self.translation
        points=points[np.isfinite(points).all(1)&(points[:,2]>0)]
        image=np.full((c['height'],c['width']),np.inf)
        if len(points):
            uv=project(points,c)
            inside=np.isfinite(uv).all(1)&(uv[:,0]>-.5)&(uv[:,0]<c['width']-.5)&(uv[:,1]>-.5)&(uv[:,1]<c['height']-.5)
            pixels=np.rint(uv[inside]).astype(int)
            np.minimum.at(image,(pixels[:,1],pixels[:,0]),points[inside,2])
        x,y=point_pixel(point,c['width'],c['height'])
        patch=image[max(0,y-2):y+3,max(0,x-2):x+3]
        samples=patch[np.isfinite(patch)]
        exact=image[y,x]
        return dict(center_px=[x,y],center_m=float(exact) if np.isfinite(exact) else None,
                    median_m=float(np.median(samples)) if len(samples) else None,
                    valid_pixels=int(len(samples)),window=5,window_pixels=int(patch.size),scale_m=self.scale,
                    status='ok' if len(samples) else '测距区域无有效深度',
                    serial=self.m['serial'])


def factory_metadata(source):
    source=Path(source)
    target=source.with_suffix('.bin')
    headers=os.environ.get('S1_CAMERA_HEADERS', '/opt/astribot_ros/software/astribot_camera_sdk/include/realsense/include')
    if not target.exists() or target.stat().st_mtime<source.stat().st_mtime:
        subprocess.run(['g++','-std=c++11',str(source),'-I'+headers,
                        '-L/usr/lib/aarch64-linux-gnu','-l:librealsense2.so.2.56','-o',str(target)],
                       check=True,capture_output=True,timeout=30)
    serial=json.loads(Path(os.environ.get('ASTRIBOT_CAMERA_INFO_PATH', '/etc/config/camera_info.json')).read_text())['right_wrist_rgbd']
    result=subprocess.run([str(target),serial],check=True,capture_output=True,text=True,timeout=10)
    return json.loads(result.stdout)


GRIPPER_STATES=('closed','open')


class FingertipPoints:
    """Per-gripper-state fingertip pixel in the right wrist image; persisted atomically."""
    def __init__(self,path=None):
        self.path=Path(path).expanduser() if path else None
        self.points={s:None for s in GRIPPER_STATES}
        self.saved={s:None for s in GRIPPER_STATES}
        self.revision=0
        self.error=None
        if self.path and self.path.exists():
            try:
                data=json.loads(self.path.read_text())
                if not isinstance(data,dict) or data.get('schema')!='s1b-fingertip-point-v1' or data.get('camera')!='right_wrist_rgbd':
                    raise ValueError('配置版本或相机不匹配')
                for s in GRIPPER_STATES:
                    entry=data['points'].get(s)
                    if entry is not None:self.points[s]=self.saved[s]=validate_point(entry['point_normalized'])
            except (ValueError,KeyError,OSError,TypeError,AttributeError) as exc:
                self.error='未加载指端点配置：'+str(exc)

    def info(self):
        return dict(points=dict(self.points),revision=self.revision,error=self.error,
                    saved={s:self.points[s] is not None and self.points[s]==self.saved[s] for s in GRIPPER_STATES},
                    config_path=str(self.path) if self.path else None)

    def select(self,state,point):
        if state not in GRIPPER_STATES:raise ValueError('夹爪状态无效')
        point=validate_point(point)
        self.points[state]=point;self.revision+=1
        return self.info()

    def save(self,state,revision,image_jpeg=None):
        if state not in GRIPPER_STATES:raise ValueError('夹爪状态无效')
        if type(revision) is not int or revision!=self.revision:raise ValueError('指端点已变更，请刷新后重新保存')
        if self.path is None:raise ValueError('未配置保存路径')
        if self.points[state] is None:raise ValueError('该状态尚未标记指端点')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        entries={}
        for s in GRIPPER_STATES:
            point=self.points[s] if s==state else self.saved[s]
            if point is None:continue
            entries[s]=dict(point_normalized=point)
        size=None
        if image_jpeg:
            image=cv2.imdecode(np.frombuffer(image_jpeg,np.uint8),cv2.IMREAD_COLOR)
            if image is not None:
                size=[image.shape[1],image.shape[0]]
                for s,e in entries.items():
                    if s==state:
                        e['pixel']=list(point_pixel(e['point_normalized'],*size));e['image_size']=size
                        e['saved_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
                        reference=self.path.with_name(self.path.stem+'_'+s+'.jpg')
                        reference.write_bytes(image_jpeg)
        if state in entries and 'pixel' not in entries[state]:
            raise ValueError('无有效右腕画面，无法保存指端点')
        old={}
        if self.path.exists():
            try:old=json.loads(self.path.read_text()).get('points',{})
            except (ValueError,OSError,AttributeError):old={}
        for s,e in entries.items():
            if s!=state and s in old:e.update({k:v for k,v in old[s].items() if k!='point_normalized'})
        data=dict(schema='s1b-fingertip-point-v1',camera='right_wrist_rgbd',points=entries)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=self.path.parent,delete=False) as out:
                temporary=out.name
                json.dump(data,out,allow_nan=False,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
            os.replace(temporary,self.path)
        finally:
            if temporary and os.path.exists(temporary):os.unlink(temporary)
        self.saved[state]=self.points[state];self.error=None
        return self.info()


class Frames:
    def __init__(self,config_path=None,fingertip_path=None):
        self.fingertip=FingertipPoints(fingertip_path)
        self.lock=threading.Lock()
        self.latest={}
        self.right=deque(maxlen=8)
        self.head=deque(maxlen=8)
        self.depth=deque(maxlen=8)
        self.result=None
        self.error=None
        self.measurement_error='正在读取相机出厂标定'
        self.measurement=None
        self.activation='等待相机'
        self.config_path=Path(config_path).expanduser() if config_path is not None else None
        self.point=None
        self.saved_point=None
        self.has_saved=False
        self.revision=0
        self.config_error=None
        if self.config_path and self.config_path.exists():
            try:
                data=json.loads(self.config_path.read_text())
                if not isinstance(data,dict) or data.get('schema')!='s1b-view-point-v1' or data.get('camera')!='right_wrist_rgbd':
                    raise ValueError('配置版本或相机不匹配')
                self.point=validate_point(data['point_normalized'])
                self.saved_point=self.point
                self.has_saved=True
            except (ValueError,KeyError,OSError,TypeError) as exc:
                self.config_error='未加载测距点配置：'+str(exc)

    def selection(self):
        return dict(point_normalized=self.point,revision=self.revision,
                    saved=self.has_saved and self.point==self.saved_point,
                    config_path=str(self.config_path) if self.config_path else None,error=self.config_error)

    def select_point(self,point):
        point=validate_point(point)
        with self.lock:
            self.point=point
            self.revision+=1
            self.result=None  # Never display the previous point's distance under a new marker.
            return self.selection()

    def save_point(self,revision):
        with self.lock:
            if type(revision) is not int or revision!=self.revision:
                raise ValueError('测距点已变更，请刷新后重新保存')
            if self.config_path is None:raise ValueError('未配置保存路径')
            self.config_path.parent.mkdir(parents=True,exist_ok=True)
            data=dict(schema='s1b-view-point-v1',camera='right_wrist_rgbd',point_normalized=self.point)
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(mode='w',dir=self.config_path.parent,delete=False) as out:
                    temporary=out.name
                    json.dump(data,out,allow_nan=False,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
                os.replace(temporary,self.config_path)
            finally:
                if temporary and os.path.exists(temporary):os.unlink(temporary)
            self.saved_point=self.point
            self.has_saved=True
            self.config_error=None
            return self.selection()

    def select_fingertip(self,state,point):
        with self.lock:return self.fingertip.select(state,point)

    def save_fingertip(self,state,revision):
        with self.lock:
            latest=self.latest.get('right')
            return self.fingertip.save(state,revision,latest['data'] if latest else None)

    def receive(self,key,message):
        try:
            fmt=message.format.lower()
            if key=='depth' and fmt!='raw':raise ValueError('unsupported_depth_format:'+fmt)
            if key!='depth' and 'jpeg' not in fmt:raise ValueError('unsupported_color_format:'+fmt)
            stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec
            if stamp<=0:raise ValueError('missing_camera_timestamp')
            frame=dict(data=bytes(message.data),source_ns=stamp,receipt_ns=time.monotonic_ns())
            with self.lock:
                old=self.latest.get(key)
                if old and stamp<=old['source_ns']:return
                self.latest[key]=frame
                self.error=None
                if key=='right':self.right.append(frame)
                if key=='head':self.head.append(frame)
                if key=='depth':self.depth.append(frame)
        except Exception as exc:
            self.error=str(exc)

    def update(self):
        with self.lock:
            if not self.right or not self.depth:return
            d=self.depth[-1]
            rgb=min(self.right,key=lambda f:abs(f['source_ns']-d['source_ns']))
            point,revision=self.point,self.revision
        skew=abs(rgb['source_ns']-d['source_ns'])
        if skew>80_000_000:
            self.measurement_error='右腕彩色与深度时间差过大'
            return
        if self.measurement is None:return
        if self.result and self.result['depth_source_ns']==d['source_ns']:return
        try:
            image=cv2.imdecode(np.frombuffer(rgb['data'],np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('invalid_right_jpeg')
            value=self.measurement.measure(d['data'],image.shape,point)
            value.update(depth_source_ns=d['source_ns'],rgb_source_ns=rgb['source_ns'],skew_ms=skew/1e6)
            with self.lock:
                if revision!=self.revision:return
                self.result=value
                self.paired=rgb
                self.paired_depth=d
                self.depth_receipt_ns=d['receipt_ns']
                self.measurement_error=None
        except Exception as exc:
            self.measurement_error=str(exc)

    def snapshot(self):
        now=time.monotonic_ns()
        with self.lock:
            result=dict(self.result) if self.result else dict(median_m=None,center_m=None,valid_pixels=0)
            selected={k:v for k,v in self.latest.items() if k in ('head','right')}
            if (self.result and not self.measurement_error
                    and now-self.depth_receipt_ns<=1_000_000_000):selected['right']=self.paired
            frames={k:dict(jpeg=base64.b64encode(v['data']).decode(),age_ms=(now-v['receipt_ns'])/1e6,
                           stale=now-v['receipt_ns']>1_000_000_000,source_ns=v['source_ns']) for k,v in selected.items()}
            stale=(not self.result or now-self.depth_receipt_ns>1_000_000_000
                   or 'right' not in frames or frames['right']['stale'])
            if stale or self.measurement_error:
                result.update(median_m=None,center_m=None,status=self.measurement_error or '深度已过期')
            return dict(frames=frames,depth=result,error=self.error,activation=self.activation,selection=self.selection(),fingertip=self.fingertip.info())

    def capture(self):
        """Atomic, timestamped export for persisted observations; never substitutes old depth."""
        with self.lock:
            now=time.monotonic_ns()
            if not self.result or self.measurement_error or self.measurement is None:
                raise ValueError('depth_not_ready')
            selected={'right':self.paired, 'depth':self.paired_depth}
            if not self.head:raise ValueError('head_not_ready')
            # Match the retained head history to the paired wrist exposure,
            # rather than combining delayed wrist processing with latest head.
            selected['head']=min(self.head,key=lambda f:abs(f['source_ns']-self.paired_depth['source_ns']))
            ages={k:round((now-f['receipt_ns'])/1e6) for k,f in selected.items()}
            if any(not 0 <= now-f['receipt_ns'] <= 1_000_000_000 for f in selected.values()):
                raise ValueError('stale_capture:age_ms='+json.dumps(ages)+',latest_age_ms='+json.dumps(
                    {k:round((now-v['receipt_ns'])/1e6) for k,v in self.latest.items()}))
            if any(abs(time.time_ns()-f['source_ns'])>1_000_000_000 for f in selected.values()):
                raise ValueError('stale_capture_source')
            if max(f['source_ns'] for f in selected.values())-min(f['source_ns'] for f in selected.values())>80_000_000:
                raise ValueError('capture_skew')
            return dict(frames={k:dict(data_base64=base64.b64encode(f['data']).decode(),
                source_ns=f['source_ns'],receipt_ns=f['receipt_ns']) for k,f in selected.items()},
                depth=dict(self.result),metadata=self.measurement.m,selection=self.selection())


HTML = "<p>ElevatorHarness S1 camera service. Use /api/snapshot or /api/capture.</p>"


def health(message):
    """Append viewer health to a local file so stale-capture causes can be read afterwards."""
    try:
        with open(os.path.expanduser('~/.config/elevator_harness/viewer_health.log'),'a') as out:
            out.write(time.strftime('%H:%M:%S ')+message+'\n')
    except OSError:pass


def serve(frames,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def setup(self):
            super().setup();self.connection.settimeout(2)
        def reply(self,status,payload):
            body=json.dumps(payload,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(body)
        def do_POST(self):
            try:
                host=self.headers.get('Host','')
                origin=self.headers.get('Origin')
                if urlsplit('http://'+host).hostname not in ('127.0.0.1','localhost','::1') or (origin and urlsplit(origin).netloc!=host):
                    self.reply(403,dict(error='仅允许本地网页修改测距点'));return
                if self.headers.get('Content-Type','').split(';')[0]!='application/json' or self.headers.get('Transfer-Encoding'):
                    raise ValueError('需要 JSON 请求')
                length=int(self.headers.get('Content-Length','-1'))
                if not 0<length<=1024:raise ValueError('请求长度无效')
                data=json.loads(self.rfile.read(length))
                if self.path=='/api/selection' and isinstance(data,dict) and set(data)=={'point_normalized'}:
                    self.reply(200,frames.select_point(data['point_normalized']))
                elif self.path=='/api/selection/save' and isinstance(data,dict) and set(data)=={'revision'}:
                    self.reply(200,frames.save_point(data['revision']))
                elif self.path=='/api/fingertip' and isinstance(data,dict) and set(data)=={'state','point_normalized'}:
                    self.reply(200,frames.select_fingertip(data['state'],data['point_normalized']))
                elif self.path=='/api/fingertip/save' and isinstance(data,dict) and set(data)=={'state','revision'}:
                    self.reply(200,frames.save_fingertip(data['state'],data['revision']))
                else:raise ValueError('请求路径或参数无效')
            except (ValueError,TypeError) as exc:self.reply(400,dict(error=str(exc)))
            except OSError as exc:self.reply(500,dict(error='配置保存失败：'+str(exc)))
        def do_GET(self):
            if self.path=='/api/capture':
                try:self.reply(200,frames.capture())
                except ValueError as exc:self.reply(503,dict(error=str(exc)))
                return
            if self.path=='/':body=HTML.encode();kind='text/html; charset=utf-8'
            elif self.path=='/health':body=json.dumps(dict(service='s1b-camera-view-v1',pid=os.getpid())).encode();kind='application/json'
            elif self.path=='/api/snapshot':body=json.dumps(frames.snapshot(),allow_nan=False).encode();kind='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server,thread


CAMERA_PROCESSES=('astribot_camera_main','astribot_camera_sub')
CAMERA_STOP_TIMEOUT_S=40
CAMERA_START_GRACE_S=30


def camera_processes_running():
    """True while any vendor camera main/sub process exists on this host."""
    # '[a]stribot...' keeps pgrep from matching its own command line.
    pattern='|'.join('[%s]%s'%(name[0],name[1:]) for name in CAMERA_PROCESSES)
    return subprocess.run(['pgrep','-f',pattern],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


class CameraRestart:
    """Restart the vendor camera as stop → wait for exit → start.

    The vendor start path (SetBool true) only checks camera_main: after main crashes it
    starts a new main AND a new sub while the old sub keeps its shared memory, so
    orphan subs pile up until /dev/shm is full. SetBool false runs the vendor stop
    script for the old main and sub first; true is sent only once both have exited.
    """
    def __init__(self,call,running=camera_processes_running,timeout_s=CAMERA_STOP_TIMEOUT_S,clock=time.monotonic):
        self.call,self.running,self.timeout_s,self.clock=call,running,timeout_s,clock
        self.phase='idle';self.future=None;self.since=None;self.message=None

    def busy(self):
        return self.phase!='idle'

    def begin(self):
        self.phase,self.future,self.since='stopping',self.call(False),self.clock()
        return '正在停止旧相机进程'

    def step(self):
        """Advance; returns a status message when it changes, else None."""
        if self.phase in ('stopping','waiting') and self.clock()-self.since>self.timeout_s:
            self.phase,self.future='failed',None
            return '相机停止超时（%ds），取消本次重启'%self.timeout_s
        if self.phase=='stopping' and self.future.done():
            self.phase,self.future='waiting',None
            return '等待旧相机进程退出'
        if self.phase=='waiting':
            gone=not self.running()
            if gone:
                self.phase,self.future='starting',self.call(True)
                return '正在启动相机'
        if self.phase=='starting' and self.future.done():
            response=self.future.result();self.phase,self.future='idle',None
            return response.message if response else '相机启动失败'
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--point-config',default='~/.config/elevator_harness/right_wrist_depth_point.json')
    parser.add_argument('--fingertip-config',default='~/.config/elevator_harness/right_wrist_fingertip_point.json')
    args=parser.parse_args()
    settings=json.loads(Path(args.settings).read_text())
    TOPICS.update(settings.get('camera_topics', {}))
    os.environ['S1_CAMERA_HEADERS']=settings.get('camera_headers', '/opt/astribot_ros/software/astribot_camera_sdk/include/realsense/include')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from std_srvs.srv import SetBool
    rclpy.init();node=rclpy.create_node('elevator_harness_keyboard_camera_view')
    frames=Frames(args.point_config,args.fingertip_config);done=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *a:done.set())
    server,thread=serve(frames,args.port)
    for key,topic in TOPICS.items():
        node.create_subscription(CompressedImage,topic,lambda m,k=key:frames.receive(k,m),qos_profile_sensor_data)
    def metadata():
        try:
            frames.measurement=CenterDepth(factory_metadata(Path(__file__).with_name('depth_metadata.cpp')))
            frames.measurement_error=None
        except Exception as exc:frames.measurement_error='深度标定读取失败：'+str(exc)
    meta=threading.Thread(target=metadata,daemon=True);meta.start()
    client=node.create_client(SetBool,'/astribot_camera_control')
    def camera_control(on):
        request=SetBool.Request();request.data=on;return client.call_async(request)
    restart=CameraRestart(camera_control)
    started=time.monotonic();activation_tried=False;next_update=0;next_health=0;next_activation=0
    try:
        while not done.is_set():
            rclpy.spin_once(node,timeout_sec=.02)
            with frames.lock:
                newest=max((v['receipt_ns'] for v in frames.latest.values()),default=0)
            silent=(time.monotonic_ns()-newest)/1e9 if newest else time.monotonic()-started
            # Retry activation (not just once at startup) whenever every stream has been silent
            # for 5 s, e.g. service not ready at launch or the driver stopped mid-run.
            # Each restart stops the old main/sub first (see CameraRestart).
            if time.monotonic()-started>2 and not restart.busy() and silent>(2 if not activation_tried else 5) \
                    and time.monotonic()>=next_activation:
                activation_tried=True;next_activation=time.monotonic()+10
                if client.service_is_ready():
                    frames.activation=restart.begin()
                    health('no camera frames for %.1fs; stopping old camera before restart'%silent)
                else:
                    frames.activation='等待相机画面（复用现有驱动）';health('camera control service not ready')
            if restart.busy():
                message=restart.step()
                if message is not None:
                    frames.activation=message;health('camera restart: '+message)
                    if not restart.busy():
                        next_activation=time.monotonic()+CAMERA_START_GRACE_S
            if time.monotonic()>=next_update:
                began=time.monotonic();frames.update();took=time.monotonic()-began
                next_update=time.monotonic()+.2
                if took>.5:health('update took %.2fs'%took)
            if time.monotonic()>=next_health:
                next_health=time.monotonic()+5
                with frames.lock:
                    now=time.monotonic_ns()
                    health('latest_age_ms='+json.dumps({k:round((now-v['receipt_ns'])/1e6) for k,v in frames.latest.items()})
                           +' error='+str(frames.measurement_error)+' load='+str(os.getloadavg()[0]))
    finally:
        server.shutdown();server.server_close();thread.join(2)
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()

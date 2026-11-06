"""Read-only live state/command monitor for P4; no SDK imports."""
import threading
import time
import numpy as np
PARTS = ['astribot_arm_right', 'astribot_gripper_right', 'astribot_head']

class ExecutionError(RuntimeError):
    pass

ALL_PARTS=['astribot_chassis','astribot_torso','astribot_arm_left','astribot_gripper_left',*PARTS]

class RobotState:
    def __init__(self, emit, *, shared_context=False, on_fault=None, control_only=False):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy,HistoryPolicy
        from astribot_msgs.msg import RobotJointState,RobotJointController
        from std_msgs.msg import Int32MultiArray
        if not rclpy.ok():rclpy.init()
        self.shared_context=shared_context
        from rclpy.executors import SingleThreadedExecutor
        self.ros_executor=SingleThreadedExecutor()
        self.rclpy=rclpy;self.node=rclpy.create_node('elevator_harness_p4_state_monitor')
        self.driver_fault=None;self.on_fault=on_fault
        self.lock=threading.Lock();self.states={};self.commands={};self.accepted={};self.emit=emit;self.error=None
        # Control consumes current feedback; queued old samples only delay the
        # SDK loop and stop observer. Diagnostic mode keeps the full topic set.
        qos=(QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
             if control_only else qos_profile_sensor_data)
        for part in ALL_PARTS:
            self.node.create_subscription(RobotJointState,'/'+part+'/joint_space_states',lambda m,p=part:self.callback(p,m,False),qos)
            if not control_only:
                self.node.create_subscription(RobotJointController,'/'+part+'/joint_space_command',lambda m,p=part:self.callback(p,m,True),qos)
            if not control_only or part in PARTS or part=='astribot_torso':
                self.node.create_subscription(RobotJointState,'/'+part+'/joint_space_command_recv',lambda m,p=part:self.accepted_callback(p,m),qos)
        self.node.create_subscription(Int32MultiArray,'/astribot_error_code/control_driver',self.driver_callback,qos_profile_sensor_data)
        self.ros_executor.add_node(self.node)
        self.thread=threading.Thread(target=self.spin,daemon=True);self.thread.start()

    def driver_callback(self, message):
        codes=[int(x) for x in message.data]
        # SDK show_error_code uses the third decimal digit as severity: 2=error.
        errors=[c for c in codes if len(str(c))==8 and str(c)[2]=='2']
        reason='driver_error:'+','.join(map(str,errors)) if errors else None
        with self.lock:
            if reason and self.driver_fault is None:self.driver_fault=reason
        # Stop before logging: a slow or broken log must not defer a driver fault.
        if reason and self.on_fault:self.on_fault(reason)
        try:self.emit(dict(event='driver_status',receipt_ns=time.monotonic_ns(),codes=codes,error_codes=errors))
        except Exception as exc:self.error=repr(exc)

    def callback(self, part, message, command):
        row=dict(event='command_topic' if command else 'state',part=part,receipt_ns=time.monotonic_ns(),header=dict(sec=message.header.stamp.sec,nanosec=message.header.stamp.nanosec,frame_id=message.header.frame_id),position=list(message.command if command else message.position),mode=int(message.mode),names=list(message.name))
        if not command:row['velocity']=list(message.velocity);row['torque']=list(message.torque)
        with self.lock:(self.commands if command else self.states)[part]=row
        try:self.emit(row)
        except BaseException as exc:self.error=repr(exc)

    def accepted_callback(self, part, message):
        row=dict(event='accepted_command',part=part,receipt_ns=time.monotonic_ns(),position=list(message.position),mode=int(message.mode),header=dict(sec=message.header.stamp.sec,nanosec=message.header.stamp.nanosec))
        with self.lock:self.accepted[part]=row
        try:self.emit(row)
        except BaseException as exc:self.error=repr(exc)

    def torso_command(self):
        with self.lock:
            row=self.accepted.get('astribot_torso')
            if (row is None or row['mode']!=1 or len(row['position'])!=4 or
                    not 0<=time.monotonic_ns()-row['receipt_ns']<=80_000_000):
                raise ExecutionError('torso_accepted_command_unavailable')
            return np.asarray(row['position'],dtype=np.float64).copy()

    def command10(self):
        with self.lock:
            if any(p not in self.accepted for p in PARTS):raise ExecutionError('missing_accepted_command')
            rows=[self.accepted[p] for p in PARTS]
            if any(len(r['position'])!=n for r,n in zip(rows,[7,1,2])):raise ExecutionError('accepted_command_length')
            if time.monotonic_ns()-min(r['receipt_ns'] for r in rows)>80_000_000:raise ExecutionError('accepted_command_stale')
            if any(r['mode']!=1 for r in rows):raise ExecutionError('accepted_command_mode_unverified')
            return np.concatenate([r['position'] for r in rows]).astype(np.float64)

    def spin(self):
        try:self.ros_executor.spin()
        except Exception as exc:
            if self.rclpy.ok():self.error=repr(exc)

    def snapshot(self):
        with self.lock:
            if any(part not in self.states for part in ALL_PARTS):raise ExecutionError('missing_component_state')
            return {k:dict(v) for k,v in self.states.items()}

    def joint10(self):
        states=self.snapshot()
        if any(len(states[p]['position'])!=n for p,n in zip(PARTS,[7,1,2])):raise ExecutionError('component_state_length')
        return np.concatenate([states[p]['position'] for p in PARTS]),min(states[p]['receipt_ns'] for p in PARTS)

    def owners(self):
        found=[]
        for name,namespace in self.node.get_node_names_and_namespaces():
            try:
                if any(s=='/astribot/control_rights' for s,t in self.node.get_service_names_and_types_by_node(name,namespace)):found.append(name)
            except Exception:pass
        return found

    def control_rights_available(self, timeout=0.2):
        """Discover ownership even when per-node graph introspection is incomplete.

        Waiting for a service does not invoke it or transfer control.
        """
        from astribot_msgs.srv import RawRequest
        client=self.node.create_client(RawRequest,'/astribot/control_rights')
        try:
            return client.wait_for_service(timeout_sec=timeout)
        finally:
            self.node.destroy_client(client)

    def request_handover(self, timeout=3.):
        from astribot_msgs.srv import RawRequest
        client=self.node.create_client(RawRequest,'/astribot/control_rights')
        try:
            if not client.wait_for_service(timeout_sec=timeout):raise ExecutionError('control_handover_service_unavailable')
            future=client.call_async(RawRequest.Request());until=time.monotonic()+timeout
            while not future.done() and time.monotonic()<until:time.sleep(.005)
            if not future.done() or future.result() is None:raise ExecutionError('control_handover_timeout')
            result=future.result().response
            if result!='success':raise ExecutionError('control_handover_rejected:'+str(result))
            self.emit(dict(event='control_handover_response',monotonic_ns=time.monotonic_ns(),response=result))
            return result
        finally:self.node.destroy_client(client)

    def close(self):
        self.ros_executor.shutdown(timeout_sec=1)
        if not self.shared_context and self.rclpy.ok():self.rclpy.shutdown()
        self.thread.join(1)
        self.node.destroy_node()

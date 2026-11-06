import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
import time

import pytest
from elevator_harness import (ChatCompletionsClient, EndToEndAgent, LocalizedAgent, Harness,
    PressButtonTask, VisualVerifier, Recorder, Context, CalibratedPressPolicy, PressConfig)
from elevator_harness.agents import action_tool
from elevator_harness.simulation import SimulatedRobot


class FakeEndpoint:
    def __init__(self):
        self.requests = []
        self.action_index = 0
        self.bad = 0
        self.delay = 0
        self.echo = ''
        self.started = Event()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append((body, self.headers.get('Authorization')))
                owner.started.set()
                time.sleep(owner.delay)
                name = body['tools'][0]['function']['name']
                values = []
                for m in body['messages']:
                    if isinstance(m['content'], list):
                        for item in m['content']:
                            if item['type'] == 'text':
                                try:
                                    values.append(json.loads(item['text']))
                                except ValueError:
                                    pass
                obs = next(v for v in (values if name == 'choose_action' else reversed(values)) if isinstance(v, dict) and 'observation_id' in v)
                if name == 'inspect_panel':
                    args = dict(observation_id=obs['observation_id'], target_box=[.4,.4,.6,.6],
                        contact_uv=[.5,.5], fully_visible=True, path_clear=True, views_conflict=False,
                        light_on=obs['state']['light_on'], hand_clear=obs['state']['hand_clear'], note='fixture')
                    if 'coordinate_units' in body['tools'][0]['function']['parameters']['properties']:
                        args.update(coordinate_units='normalized_01', global_target_box=[.4,.4,.6,.6],
                                    global_contact_uv=[.5,.5], global_visibility='full',
                                    target_visibility='full', contact_visibility='full')
                elif name == 'locate_button_in_crop':
                    args = dict(observation_id=obs['observation_id'], coordinate_units='normalized_1000',
                                button_visibility='full', button_box=[400,400,600,600])
                elif name == 'review_progress':
                    args = dict(observation_id=obs['observation_id'], before_observation_id=next(v['observation_id'] for v in values if isinstance(v,dict) and 'state' in v),
                                progress=True, note='fixture review', alternative_explanation='unknown', next_check='inspect next image',prediction_check='not_supplied')
                else:
                    meta = next((v for v in values if isinstance(v, dict) and 'capabilities' in v), {'capabilities': {'tools': [{'name': 'closed'}]}})
                    closed = meta['capabilities']['tools'][-1]['name']
                    steps = [dict(kind='move',delta=[.005,-.005,0]),
                        dict(kind='configure_tool',configuration=closed), dict(kind='move',delta=[0,0,.02]),
                        dict(kind='move',delta=[0,0,.01]), dict(kind='move',delta=[0,0,-.02]), dict(kind='done')]
                    args = dict(observation_id=obs['observation_id'], reason='fixture', action=steps[min(owner.action_index,5)])
                    if 'press_assessment' in body['tools'][0]['function']['parameters']['properties']:
                        args['press_assessment'] = dict(observation_id=obs['observation_id'],path_clear=True,views_conflict=False,global_view='clear path',local_view='uncertain occlusion')
                    if 'reassessment' in body['tools'][0]['function']['parameters']['properties']:
                        args['reassessment'] = dict(observation_id=obs['observation_id'],hypothesis='parallax',expected_visual_change='closer',if_no_change='reobserve')
                    if owner.bad:
                        owner.bad -= 1
                        args = {'invalid': True}
                    else:
                        owner.action_index += 1
                response = dict(choices=[dict(finish_reason='tool_calls',message=dict(tool_calls=[
                    dict(function=dict(name=name,arguments=json.dumps(args)))]))], extra=owner.echo)
                data = json.dumps(response).encode()
                try:
                    self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread = Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}/v1'

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(2)


@pytest.fixture
def endpoint():
    server = FakeEndpoint()
    yield server
    server.close()


@pytest.mark.parametrize('agent_type', [EndToEndAgent, LocalizedAgent])
@pytest.mark.parametrize('rotated', [False, True])
def test_two_agents_two_robot_configurations_over_real_http(tmp_path, endpoint, agent_type, rotated):
    robot = SimulatedRobot(rotated=rotated, movable_view=not rotated)
    recorder = Recorder(tmp_path/'run')
    model = ChatCompletionsClient(endpoint.base, 'test-model', recorder=recorder)
    names = [t.name for t in robot.capabilities.tools]
    policy = CalibratedPressPolicy(PressConfig(*names, .15, .085)) if agent_type is LocalizedAgent else None
    run = Harness(robot, agent_type(model), PressButtonTask('test button'), VisualVerifier(model), recorder, policy)
    assert run.run().status == 'completed'
    names = [r[0]['tools'][0]['function']['name'] for r in endpoint.requests]
    assert names.count('choose_action') == 6
    assert names.count('inspect_panel') == (8 if agent_type is LocalizedAgent else 6)
    assert names.count('locate_button_in_crop') == (2 if agent_type is LocalizedAgent else 0)
    if not rotated:
        assert 'adjust_view' in json.dumps(endpoint.requests[0][0]) if agent_type is EndToEndAgent else True
    assert all(header is None for _, header in endpoint.requests)


def test_repair_and_credential_redaction(tmp_path, endpoint, monkeypatch):
    secret = 'never-write-this-secret'
    monkeypatch.setenv('HARNESS_TEST_KEY', secret)
    endpoint.bad, endpoint.echo = 1, secret
    recorder = Recorder(tmp_path/'run')
    model = ChatCompletionsClient(endpoint.base, 'fixture', key_env='HARNESS_TEST_KEY', recorder=recorder)
    run = Harness(SimulatedRobot(), EndToEndAgent(model), PressButtonTask('button'), VisualVerifier(model), recorder)
    assert run.run().status == 'completed'
    assert endpoint.requests[0][1] == 'Bearer ' + secret
    assert secret not in (tmp_path/'run/events.jsonl').read_text()
    assert 'correction' in (tmp_path/'run/events.jsonl').read_text()


def test_model_timeout_and_cancellation(endpoint):
    endpoint.delay = .3
    robot = SimulatedRobot()
    schema = action_tool(robot.capabilities)
    # Transport timeout happens before response parsing, so no real action context is needed.
    messages = [{'role':'user','content':[{'type':'text','text':json.dumps({'observation_id':'a'})}]}]
    client = ChatCompletionsClient(endpoint.base,'fixture',timeout=.05)
    with pytest.raises((TimeoutError, RuntimeError)):
        client.complete(messages,schema,Event())
    endpoint.started.clear()
    client = ChatCompletionsClient(endpoint.base,'fixture',timeout=2)
    cancel = Event();results=[]
    def call():
        try:client.complete(messages,schema,cancel)
        except Exception as exc:results.append(exc)
    thread=Thread(target=call);thread.start();assert endpoint.started.wait(1)
    cancel.set();thread.join(.2)
    assert not thread.is_alive() and isinstance(results[0],InterruptedError)


def test_missing_capabilities_are_not_advertised():
    robot = SimulatedRobot(rotated=True,movable_view=False)
    assert 'adjust_view' not in json.dumps(action_tool(robot.capabilities))

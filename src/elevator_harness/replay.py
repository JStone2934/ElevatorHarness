"""Read-only re-decision on recorded evidence. No robot, transition or success score."""
import json
from pathlib import Path
from threading import Event

from .types import Capabilities, OperationFrame, ToolConfiguration, Observation, Image, Context


def replay_decision(source, observation_id, agent, recorder, cancel=None):
    """Propose once using the actual history preceding a saved observation.

    Subsequent recorded frames belong to the ORIGINAL commands and are deliberately
    not advanced after the new proposal. Use a new Agent instance for each replay.
    """
    source = Path(source).resolve()
    history = []
    metadata = None
    observation = None
    for line in (source/'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] == 'run':
            metadata = event['value']
        elif event['event'] == 'transition':
            history.append(event['value'])
        elif event['event'] == 'observation':
            value = event['value']
            raw = value['observation']
            if raw['id'] == observation_id:
                images = {}
                for role, image in value['images'].items():
                    path = (source/image['path']).resolve()
                    if not path.is_relative_to(source):
                        raise ValueError('recorded image is outside run directory')
                    images[role] = Image(path.read_bytes(), image['mime_type'])
                raw = dict(raw, images=images)
                observation = Observation(**raw)
                # The transition for this observation is written just after its capture.
                continue
            if observation is not None:
                break
        if observation is not None and event['event'] == 'transition':
            break
        if observation is not None and event['event'] in ('proposal', 'model_request', 'result'):
            break
    if observation is None or metadata is None:
        raise ValueError('recorded observation or run metadata missing')
    cap = dict(metadata['capabilities'])
    cap['frame'] = OperationFrame(**cap['frame'])
    cap['tools'] = tuple(ToolConfiguration(**row) for row in cap['tools'])
    capabilities = Capabilities(**cap)
    context = Context(observation, capabilities, metadata['goal'],
                      history[-metadata.get('history_limit', 8):], {'stage': 'recorded_observation'},
                      feedback='Offline re-decision only. No new action is executed or followed by a simulated outcome.')
    recorder.observation(observation)
    proposal = agent.decide(context, cancel or Event())
    proposal.validate(capabilities)
    if proposal.observation_id != observation.id:
        raise ValueError('replay proposal references different observation')
    recorder.event('replay_only', {'proposal': proposal, 'evidence': context.evidence,
                                 'semantics': 'no execution; no closed-loop success score'})
    return proposal

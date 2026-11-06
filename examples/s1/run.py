"""Run against an explicitly started S1 service, directly or through an SSH tunnel."""
import argparse
from dataclasses import asdict
import json
import signal
from elevator_harness import (Harness, PressButtonTask, Recorder, ChatCompletionsClient,
    EndToEndAgent, LocalizedAgent, VisualVerifier, CalibratedPressPolicy, PressConfig, PerceptionConfig)
from .adapter import S1Adapter
from .configuration import load_settings


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--robot-base', default='http://127.0.0.1:6210')
    p.add_argument('--model-base', required=True, help='Base including API prefix, e.g. /v1')
    p.add_argument('--model', required=True)
    p.add_argument('--key-env')
    p.add_argument('--goal', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-rounds', type=int, default=40)
    p.add_argument('--max-corrections', type=int, default=3)
    p.add_argument('--model-timeout', type=float, default=120.)
    args = p.parse_args(argv)
    settings = load_settings(args.config)
    recorder = Recorder(args.output)
    model = ChatCompletionsClient(args.model_base, args.model, key_env=args.key_env,
                                  timeout=args.model_timeout, recorder=recorder)
    localized = settings['strategy'] == 'localized'
    agent = LocalizedAgent(model, perception=PerceptionConfig(**settings.get('perception', {}))) if localized else EndToEndAgent(model)
    policy = CalibratedPressPolicy(PressConfig(**settings['press'])) if localized else None
    robot = S1Adapter(args.robot_base, settings, recorder)
    harness = Harness(robot, agent, PressButtonTask(args.goal), VisualVerifier(model), recorder,
                      policy, max_rounds=args.max_rounds, max_corrections=args.max_corrections)
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: harness.stop())
        result = harness.run()
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.status == 'completed' else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())

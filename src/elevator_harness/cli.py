import argparse
from dataclasses import asdict
import json
import signal

from .recording import Recorder
from .runtime import Harness
from .simulation import SimulatedRobot, ScriptedAgent, SimulatedVerifier
from .tasks import PressButtonTask


def main(argv=None):
    parser = argparse.ArgumentParser(description='ElevatorHarness model-free contract demonstration')
    parser.add_argument('command', choices=['simulate'])
    parser.add_argument('--output', required=True, help='New run directory')
    parser.add_argument('--rotated', action='store_true')
    args = parser.parse_args(argv)
    run = Harness(SimulatedRobot(rotated=args.rotated, movable_view=not args.rotated), ScriptedAgent(),
                  PressButtonTask('activate the synthetic button and withdraw'), SimulatedVerifier(), Recorder(args.output))
    old = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.signal(sig, lambda *_: run.stop())
        result = run.run()
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.status == 'completed' else 1
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())

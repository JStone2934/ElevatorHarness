"""Validate deployment configuration before importing or acquiring the SDK."""
import argparse
from .configuration import load_settings


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, default=6210)
    p.add_argument('--view-port', type=int, default=8765)
    p.add_argument('--control-hz', type=float, default=250.)
    p.add_argument('--speed-mm-s', type=float, default=20.)
    p.add_argument('--head-speed-deg-s', type=float, default=20.)
    p.add_argument('--auto-confirm', action='store_true')
    args = p.parse_args(argv)
    load_settings(args.config)
    from .hardware import run
    return run(args)


if __name__ == '__main__':
    main()

import builtins
import sys
import termios
import time
from urllib.request import urlopen

def connect_robot(factory, frequency, *, auto_confirm=False):
    """Answer SDK initialization prompts; restore normal input after connecting."""
    original_input = builtins.input

    def interactive_answer(*args, **kwargs):
        if auto_confirm:
            prompt = args[0] if args else kwargs.get('prompt', '')
            print(f'{prompt}yes [自动确认 SDK 控制权接管]', flush=True)
            return 'yes'
        try:
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            while True:
                answer = original_input(*args, **kwargs).strip().lower()
                if answer in ('yes', '', 'no', 'n'):
                    return 'yes' if answer == 'yes' else ''
                print('请输入完整的 yes 后回车以接管，或直接回车取消；尚未进入移动模式。',flush=True)
        except (EOFError, KeyboardInterrupt, OSError, termios.error):
            # The SDK's broad exception fallback can acquire rights on EOF.
            # A disconnected terminal must instead decline ownership.
            return ''

    builtins.input = interactive_answer
    try:
        return factory(freq=frequency, high_control_rights=False, node_name='elevator_s1_service')
    finally:
        builtins.input = original_input

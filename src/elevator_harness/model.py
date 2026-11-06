"""Configurable Chat Completions transport; no model-specific assumptions."""
import json
import os
import queue
import threading
import time
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlsplit


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward Authorization to a redirect destination.


class ChatCompletionsClient:
    def __init__(self, base_url, model, *, key_env=None, timeout=120., recorder=None):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('model base must be an HTTP(S) URL without credentials/query/fragment')
        if not model or timeout <= 0:
            raise ValueError('model name and positive timeout required')
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model, self.timeout, self.recorder = model, timeout, recorder
        self._key = os.environ.get(key_env) if key_env else None
        if key_env and not self._key:
            raise ValueError('configured API key environment variable is empty')

    def _redact(self, value):
        encoded = json.dumps(value, ensure_ascii=False)
        if self._key:
            encoded = encoded.replace(json.dumps(self._key, ensure_ascii=False)[1:-1], '[REDACTED]')
        return json.loads(encoded)

    def complete(self, messages, tool, cancel):
        if cancel.is_set():
            raise InterruptedError('model call cancelled')
        body = {'model': self.model, 'messages': messages, 'tools': [tool],
                'tool_choice': {'type': 'function', 'function': {'name': tool['function']['name']}}}
        if self.recorder:
            self.recorder.event('model_request', self._redact(body))
        result = queue.Queue(maxsize=1)

        def request():
            try:
                headers = {'Content-Type': 'application/json'}
                if self._key:
                    headers['Authorization'] = 'Bearer ' + self._key
                req = Request(self.url, json.dumps(body, allow_nan=False).encode(), headers)
                with build_opener(NoRedirect).open(req, timeout=self.timeout) as response:
                    raw = json.loads(response.read())
                result.put((True, raw))
            except Exception as exc:
                # Raw HTTP errors and server bodies can contain credentials.
                result.put((False, type(exc).__name__))

        threading.Thread(target=request, daemon=True).start()
        deadline = time.monotonic() + self.timeout
        while True:
            if cancel.is_set():
                raise InterruptedError('model call cancelled; late response discarded')
            if time.monotonic() >= deadline:
                raise TimeoutError('model request timed out')
            try:
                ok, raw = result.get(timeout=min(.05, max(.001, deadline-time.monotonic())))
                break
            except queue.Empty:
                continue
        if cancel.is_set():
            raise InterruptedError('model call cancelled')
        if not ok:
            raise RuntimeError('model transport failed: ' + raw)
        raw = self._redact(raw)
        if self.recorder:
            self.recorder.event('model_response', raw)
        try:
            choices = raw['choices']
            calls = choices[0]['message']['tool_calls']
            if len(choices) != 1 or len(calls) != 1 or choices[0]['finish_reason'] != 'tool_calls':
                raise ValueError('one complete tool call required')
            fn = calls[0]['function']
            if fn['name'] != tool['function']['name']:
                raise ValueError('unexpected model tool')
            args = json.loads(fn['arguments'])
            if not isinstance(args, dict):
                raise ValueError('tool arguments must be an object')
            return args
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('invalid model response') from exc

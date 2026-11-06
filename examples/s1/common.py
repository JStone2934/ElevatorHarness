import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError



def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def request(base, route, payload=None, timeout=10, binary=False):
    url = base.rstrip('/')+route
    # Robot/camera transport never attaches model credentials.
    req = Request(url, data=None if payload is None else json_bytes(payload),
                  headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=timeout) as response:
            data = response.read()
    except HTTPError as exc:
        raise ValueError(f'HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc
    return data if binary else json.loads(data)


def number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('finite_number_required')
    return float(value)


def pose_error(a, b):
    distance = math.dist(a[:3], b[:3])*1000
    qa, qb = a[3:7], b[3:7]
    dot = abs(sum(x*y for x,y in zip(qa,qb))) / (math.sqrt(sum(x*x for x in qa))*math.sqrt(sum(x*x for x in qb)))
    return distance, math.degrees(2*math.acos(min(1., dot)))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


SESSION_SCHEMA = "elevator-s1-session-v1"

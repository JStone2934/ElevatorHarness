"""Append-only run evidence, with no configuration credentials in the journal."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import time


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, bytes):
        return {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self._ids = set()

    def event(self, kind, value):
        with (self.root/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'time_ns': time.time_ns(), 'event': kind,
                                'value': serializable(value)}, allow_nan=False) + '\n')

    def observation(self, obs):
        if obs.id in self._ids:
            return
        self._ids.add(obs.id)
        # Hash identifiers rather than accepting external values as filesystem paths.
        directory = self.root/'observations'/hashlib.sha256(obs.id.encode()).hexdigest()
        directory.mkdir(parents=True)
        images = {}
        for index, (role, image) in enumerate(obs.images.items()):
            name = f'{index}.image'
            (directory/name).write_bytes(image.data)
            images[role] = {'path': str((directory/name).relative_to(self.root)),
                            'mime_type': image.mime_type}
        self.event('observation', {'observation': obs, 'images': images})

    def derived_image(self, kind, image, lineage):
        directory = self.root/'derived_images'
        directory.mkdir(exist_ok=True)
        name = hashlib.sha256(image.data).hexdigest()+'.jpg'
        (directory/name).write_bytes(image.data)
        self.event(kind, {'lineage': lineage, 'path': str((directory/name).relative_to(self.root)),
                          'sha256': hashlib.sha256(image.data).hexdigest(), 'mime_type': image.mime_type})

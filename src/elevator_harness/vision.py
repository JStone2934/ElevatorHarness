"""Image transforms with explicit source/crop coordinates and immutable lineage."""
from dataclasses import dataclass
import hashlib
import io
import math

from .types import Image, vector


def require_pillow():
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise RuntimeError('Install elevator-harness[vision] for crop refinement and press zoom') from exc
    return PILImage


def box01(box):
    values = vector(box, 4)
    if any(not 0 <= x <= 1 for x in values) or values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError('nonempty normalized image box required')
    return list(values)


def crop_window(box, size, scale=3.):
    """Square window, scale times the larger box dimension, shifted inside image."""
    box = box01(box)
    width, height = size
    if min(width, height) < 1 or not math.isfinite(scale) or scale < 1:
        raise ValueError('invalid crop dimensions or scale')
    left, top, right, bottom = [v * size[i % 2] for i, v in enumerate(box)]
    side = max(1, min(round(scale * max(right-left, bottom-top)), width, height))
    x = max(0, min(width-side, round((left+right-side)/2)))
    y = max(0, min(height-side, round((top+bottom-side)/2)))
    return [x, y, x+side, y+side]


def restore_box(box, window, size):
    """Crop-relative [0,1] box -> source full-image [0,1] box (not resized pixels)."""
    box = box01(box)
    x0, y0, x1, y1 = window
    width, height = size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('crop window outside image')
    return box01([(x0+box[0]*(x1-x0))/width, (y0+box[1]*(y1-y0))/height,
                  (x0+box[2]*(x1-x0))/width, (y0+box[3]*(y1-y0))/height])


@dataclass(frozen=True)
class Crop:
    image: Image
    lineage: dict


def make_crop(image, box, *, observation_id, camera, box_observation_id,
              scale=3., side_px=768):
    PILImage = require_pillow()
    with PILImage.open(io.BytesIO(image.data)) as source:
        source = source.convert('RGB')
        size = list(source.size)
        window = crop_window(box, size, scale)
        crop = source.crop(tuple(window))
        ratio = side_px/max(crop.size)
        crop = crop.resize((max(1, round(crop.width*ratio)), max(1, round(crop.height*ratio))),
                           PILImage.Resampling.LANCZOS)
        stream = io.BytesIO()
        crop.save(stream, format='JPEG', quality=92)
    return Crop(Image(stream.getvalue()), {
        'observation_id': observation_id, 'camera': camera,
        'box_observation_id': box_observation_id, 'source_box': list(box),
        'source_sha256': hashlib.sha256(image.data).hexdigest(), 'source_size': size,
        'window_px': window, 'output_size': list(crop.size), 'scale': scale,
        'semantics': 'crop of current image; historical box is an ROI hint, not current localization',
    })

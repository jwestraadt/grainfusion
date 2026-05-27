from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
DISPLAY_MAX_SIDE = 2048


def read_image(path: str | Path) -> np.ndarray:
    """Read a TIFF, PNG, or JPEG image as a NumPy array."""
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {suffix}")

    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile

            array = tifffile.imread(image_path)
            return _coerce_image_array(array)
        except Exception:
            pass

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"1", "P", "CMYK", "YCbCr"}:
            image = image.convert("RGB")
        return _coerce_image_array(np.asarray(image))


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image array to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(image)
    if output_path.suffix.lower() in {".jpg", ".jpeg"} and array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]

    if array.dtype.kind == "f":
        array = np.clip(array, 0, 255).astype(np.uint8)

    Image.fromarray(array).save(output_path)


def as_display_rgb(image: np.ndarray) -> np.ndarray:
    """Return a uint8 RGB display image with contrast scaled for viewing."""
    array = np.asarray(image)
    if array.ndim == 2:
        gray = normalize_to_uint8(array)
        return np.dstack([gray, gray, gray])

    if array.ndim == 3 and array.shape[2] == 1:
        gray = normalize_to_uint8(array[:, :, 0])
        return np.dstack([gray, gray, gray])

    if array.ndim == 3 and array.shape[2] >= 3:
        return normalize_to_uint8(array[:, :, :3])

    raise ValueError(f"Unsupported image shape for display: {array.shape}")


def display_preview_rgb(image: np.ndarray, max_side: int = DISPLAY_MAX_SIDE) -> tuple[np.ndarray, int]:
    """Return a downsampled RGB preview and its integer pixel step."""
    array = np.asarray(image)
    height, width = array.shape[:2]
    step = max(1, int(np.ceil(max(height, width) / max(1, max_side))))
    preview = array[::step, ::step] if step > 1 else array
    return as_display_rgb(preview), step


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale an array to uint8 using its finite min/max range."""
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.copy()

    if array.dtype == np.bool_:
        return (array.astype(np.uint8) * 255)

    if array.dtype.kind in {"i", "u"}:
        min_value = array.min()
        max_value = array.max()
    else:
        finite = np.isfinite(array)
        if not finite.any():
            return np.zeros(array.shape, dtype=np.uint8)
        values = array[finite].astype(np.float64)
        min_value = values.min()
        max_value = values.max()

    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array.astype(np.float64) - min_value) / (max_value - min_value)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def _coerce_image_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)

    if array.ndim == 0:
        raise ValueError("Image has no pixel dimensions")

    if array.ndim == 3 and array.shape[-1] not in {1, 3, 4}:
        array = array[0]

    if array.ndim > 3:
        while array.ndim > 3:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] not in {1, 3, 4}:
            array = array[0]

    if array.ndim not in {2, 3}:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    return np.ascontiguousarray(array)


def load_ang_ipfz(path: str | Path) -> np.ndarray:
    """Load an EDAX .ang file and return an IPF-Z colour map as (h, w, 3) uint8."""
    from orix.io import load as _orix_load
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d

    xmap = _orix_load(str(path))
    h, w = xmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for phase_id, phase in xmap.phases:
        if phase_id == -1:
            continue
        mask = xmap.phase_id == phase_id
        ckey = IPFColorKeyTSL(phase.point_group, direction=Vector3d.zvector())
        colors = ckey.orientation2color(xmap[mask].orientations)
        rgb.reshape(-1, 3)[mask] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    return rgb

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import as_display_rgb
from .transforms import resample_fixed_to_output, warp_moving_to_output


def make_overlay(
    fixed_image: np.ndarray,
    moving_or_modality_image: np.ndarray,
    settings: dict,
    alpha: float,
    scale_bar_length: float | None = None,
    scale_bar_units: str = "um",
    crop_to_common: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))

    fixed_output, fixed_mask = resample_fixed_to_output(fixed_image, settings)
    moving_output, moving_mask = warp_moving_to_output(moving_or_modality_image, settings)

    if crop_to_common:
        fixed_output, moving_output, moving_mask = crop_to_common_area(
            fixed_output,
            moving_output,
            fixed_mask,
            moving_mask,
        )

    fixed_rgb = as_display_rgb(fixed_output)
    moving_rgb = as_display_rgb(moving_output)
    mask_alpha = moving_mask[:, :, None].astype(np.float32) * alpha

    overlay = fixed_rgb.astype(np.float32) * (1.0 - mask_alpha)
    overlay += moving_rgb.astype(np.float32) * mask_alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if scale_bar_length is not None and scale_bar_length > 0:
        overlay = add_scale_bar(
            overlay,
            pixel_size=float(settings["registration"]["output_pixel_size"]),
            length=float(scale_bar_length),
            units=scale_bar_units,
        )

    return overlay, moving_output, fixed_output


def crop_to_common_area(
    fixed_output: np.ndarray,
    moving_output: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_mask = np.asarray(fixed_mask, dtype=bool) & np.asarray(moving_mask, dtype=bool)
    y_indices, x_indices = np.nonzero(common_mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("The fixed and moving images do not overlap on the output grid.")

    y_slice = slice(int(y_indices.min()), int(y_indices.max()) + 1)
    x_slice = slice(int(x_indices.min()), int(x_indices.max()) + 1)
    return (
        fixed_output[y_slice, x_slice],
        moving_output[y_slice, x_slice],
        moving_mask[y_slice, x_slice],
    )


def add_scale_bar(
    image_rgb: np.ndarray,
    pixel_size: float,
    length: float,
    units: str = "um",
    color: tuple[int, int, int] = (255, 255, 255),
    margin: int = 24,
    thickness: int = 6,
) -> np.ndarray:
    if pixel_size <= 0:
        raise ValueError("Pixel size must be greater than zero.")

    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    max_bar_width = max(1, width - 2 * margin)
    requested_bar_width = max(1, int(round(length / pixel_size)))
    bar_width = min(requested_bar_width, max_bar_width)
    displayed_length = bar_width * pixel_size

    x0 = margin
    y0 = max(margin, height - margin - thickness)
    x1 = x0 + bar_width
    y1 = y0 + thickness

    draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill=(0, 0, 0))
    draw.rectangle((x0, y0, x1, y1), fill=color)

    label = f"{displayed_length:g} {units}".strip()
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = x0
    text_y = max(0, y0 - text_height - 6)
    draw.rectangle(
        (text_x - 3, text_y - 2, text_x + text_width + 3, text_y + text_height + 2),
        fill=(0, 0, 0),
    )
    draw.text((text_x, text_y), label, fill=color, font=font)
    return np.asarray(image)

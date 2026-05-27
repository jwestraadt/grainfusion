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
    scale_bar_font_size: int | None = None,
    scale_bar_thickness: int | None = None,
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
            font_size=scale_bar_font_size,
            thickness=scale_bar_thickness,
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
    margin: int | None = None,
    thickness: int | None = None,
    font_size: int | None = None,
) -> np.ndarray:
    if pixel_size <= 0:
        raise ValueError("Pixel size must be greater than zero.")

    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    margin, thickness, font_size, label_padding = scale_bar_style(
        width=width,
        height=height,
        margin=margin,
        thickness=thickness,
        font_size=font_size,
    )

    max_bar_width = max(1, width - 2 * margin)
    requested_bar_width = max(1, int(round(length / pixel_size)))
    bar_width = min(requested_bar_width, max_bar_width)
    displayed_length = round(bar_width * pixel_size)

    x0 = margin
    y0 = max(margin, height - margin - thickness)
    x1 = x0 + bar_width
    y1 = y0 + thickness

    draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill=(0, 0, 0))
    draw.rectangle((x0, y0, x1, y1), fill=color)

    label = f"{displayed_length:d} {units}".strip()
    font = fitted_font(label, font_size, max_width=max_bar_width, draw=draw)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_padding_x = max(3, int(round(font_size * 0.18)))
    text_padding_y = max(2, int(round(font_size * 0.12)))
    text_box_x0 = min(x0, max(0, width - margin - text_width - 2 * text_padding_x))
    text_box_y0 = max(0, y0 - text_height - label_padding - 2 * text_padding_y)
    text_box_x1 = min(width, text_box_x0 + text_width + 2 * text_padding_x)
    text_box_y1 = min(height, text_box_y0 + text_height + 2 * text_padding_y)
    text_x = text_box_x0 + text_padding_x - bbox[0]
    text_y = text_box_y0 + text_padding_y - bbox[1]
    draw.rectangle(
        (
            text_box_x0,
            text_box_y0,
            text_box_x1,
            text_box_y1,
        ),
        fill=(0, 0, 0),
    )
    draw.text((text_x, text_y), label, fill=color, font=font)
    return np.asarray(image)


def scale_bar_style(
    width: int,
    height: int,
    margin: int | None = None,
    thickness: int | None = None,
    font_size: int | None = None,
) -> tuple[int, int, int, int]:
    short_side = max(1, min(width, height))
    resolved_margin = margin if margin is not None else round(short_side * 0.035)
    resolved_thickness = thickness if thickness is not None else round(short_side * 0.0075)
    resolved_font_size = font_size if font_size is not None else round(short_side * 0.03)

    resolved_margin = int(np.clip(resolved_margin, 10, 96))
    resolved_thickness = int(np.clip(resolved_thickness, 3, 28))
    resolved_font_size = int(np.clip(resolved_font_size, 11, 72))
    label_padding = max(4, int(round(resolved_font_size * 0.35)))
    return resolved_margin, resolved_thickness, resolved_font_size, label_padding


def fitted_font(label: str, font_size: int, max_width: int, draw: ImageDraw.ImageDraw) -> ImageFont.ImageFont:
    size = max(8, int(font_size))
    while size >= 8:
        font = load_label_font(size)
        bbox = draw.textbbox((0, 0), label, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
    return load_label_font(8)


def load_label_font(font_size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=font_size)
        except OSError:
            continue
    return ImageFont.load_default()

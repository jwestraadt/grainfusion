from __future__ import annotations

from math import ceil
from typing import Iterable

import cv2
import numpy as np


class RegistrationError(ValueError):
    """Raised when a registration cannot be estimated or applied."""


def build_registration_settings(
    fixed_points: Iterable[tuple[float, float]],
    moving_points: Iterable[tuple[float, float]],
    transform_type: str,
    fixed_pixel_size: float,
    moving_pixel_size: float,
    output_basis: str,
    fixed_shape: tuple[int, int],
    moving_shape: tuple[int, int],
    fixed_path: str | None = None,
    moving_path: str | None = None,
) -> dict:
    fixed = np.asarray(list(fixed_points), dtype=np.float64)
    moving = np.asarray(list(moving_points), dtype=np.float64)

    if fixed.shape != moving.shape or fixed.ndim != 2 or fixed.shape[1] != 2:
        raise RegistrationError("Fixed and moving point lists must contain matching x/y pairs.")

    transform_type = transform_type.lower()
    if transform_type not in {"affine", "homography"}:
        raise RegistrationError("Transform type must be 'affine' or 'homography'.")

    output_basis = output_basis.lower()
    if output_basis not in {"fixed", "moving"}:
        raise RegistrationError("Output basis must be 'fixed' or 'moving'.")

    min_points = 3 if transform_type == "affine" else 4
    if len(fixed) < min_points:
        raise RegistrationError(f"{transform_type} registration needs at least {min_points} point pairs.")

    if fixed_pixel_size <= 0 or moving_pixel_size <= 0:
        raise RegistrationError("Pixel sizes must be greater than zero.")

    fixed_phys = fixed * fixed_pixel_size
    moving_phys = moving * moving_pixel_size
    matrix_physical = estimate_physical_matrix(moving_phys, fixed_phys, transform_type)
    residuals = registration_residuals(moving_phys, fixed_phys, matrix_physical)

    output_shape, output_pixel_size = output_grid(
        fixed_shape=fixed_shape,
        fixed_pixel_size=fixed_pixel_size,
        moving_pixel_size=moving_pixel_size,
        output_basis=output_basis,
    )

    return {
        "registration": {
            "transform_type": transform_type,
            "matrix_physical": matrix_physical,
            "fixed_points": fixed,
            "moving_points": moving,
            "fixed_pixel_size": float(fixed_pixel_size),
            "moving_pixel_size": float(moving_pixel_size),
            "output_basis": output_basis,
            "output_pixel_size": float(output_pixel_size),
            "fixed_shape": [int(fixed_shape[0]), int(fixed_shape[1])],
            "moving_shape": [int(moving_shape[0]), int(moving_shape[1])],
            "output_shape": [int(output_shape[0]), int(output_shape[1])],
            "fixed_path": fixed_path,
            "moving_path": moving_path,
            "mean_residual": float(np.mean(residuals)),
            "max_residual": float(np.max(residuals)),
        }
    }


def estimate_physical_matrix(
    moving_phys: np.ndarray,
    fixed_phys: np.ndarray,
    transform_type: str,
) -> np.ndarray:
    if transform_type == "affine":
        if len(moving_phys) == 3:
            affine = cv2.getAffineTransform(
                moving_phys.astype(np.float32),
                fixed_phys.astype(np.float32),
            )
        else:
            affine, _ = cv2.estimateAffine2D(
                moving_phys,
                fixed_phys,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=5000,
                confidence=0.995,
            )
        if affine is None:
            raise RegistrationError("Could not estimate affine transform from the selected points.")
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :] = affine
        return matrix

    if len(moving_phys) == 4:
        matrix = cv2.getPerspectiveTransform(
            moving_phys.astype(np.float32),
            fixed_phys.astype(np.float32),
        )
    else:
        matrix, _ = cv2.findHomography(
            moving_phys,
            fixed_phys,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.995,
        )
    if matrix is None:
        raise RegistrationError("Could not estimate homography from the selected points.")
    return matrix.astype(np.float64)


def registration_residuals(
    moving_phys: np.ndarray,
    fixed_phys: np.ndarray,
    matrix_physical: np.ndarray,
) -> np.ndarray:
    projected = apply_homogeneous(matrix_physical, moving_phys)
    return np.linalg.norm(projected - fixed_phys, axis=1)


def apply_homogeneous(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    denom = projected[:, 2:3]
    if np.any(np.isclose(denom, 0)):
        raise RegistrationError("Transform projected at least one point to infinity.")
    return projected[:, :2] / denom


def output_grid(
    fixed_shape: tuple[int, int] | list[int],
    fixed_pixel_size: float,
    moving_pixel_size: float,
    output_basis: str,
) -> tuple[tuple[int, int], float]:
    fixed_height, fixed_width = int(fixed_shape[0]), int(fixed_shape[1])
    if output_basis == "fixed":
        return (fixed_height, fixed_width), float(fixed_pixel_size)

    output_pixel_size = float(moving_pixel_size)
    output_width = ceil(fixed_width * fixed_pixel_size / output_pixel_size)
    output_height = ceil(fixed_height * fixed_pixel_size / output_pixel_size)
    return (int(output_height), int(output_width)), output_pixel_size


def moving_pixel_to_output_pixel_matrix(settings: dict) -> np.ndarray:
    registration = settings["registration"]
    matrix_physical = np.asarray(registration["matrix_physical"], dtype=np.float64)
    moving_pixel_size = float(registration["moving_pixel_size"])
    output_pixel_size = float(registration["output_pixel_size"])

    source_to_physical = np.diag([moving_pixel_size, moving_pixel_size, 1.0])
    physical_to_output = np.diag([1.0 / output_pixel_size, 1.0 / output_pixel_size, 1.0])
    return physical_to_output @ matrix_physical @ source_to_physical


def fixed_pixel_to_output_pixel_matrix(settings: dict) -> np.ndarray:
    registration = settings["registration"]
    fixed_pixel_size = float(registration["fixed_pixel_size"])
    output_pixel_size = float(registration["output_pixel_size"])
    scale = fixed_pixel_size / output_pixel_size
    return np.diag([scale, scale, 1.0])


def warp_moving_to_output(
    image: np.ndarray,
    settings: dict,
    interpolation: str = "linear",
    border_value: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = moving_pixel_to_output_pixel_matrix(settings)
    output_shape = tuple(settings["registration"]["output_shape"])
    return warp_image(image, matrix, output_shape, interpolation, border_value)


def resample_fixed_to_output(
    image: np.ndarray,
    settings: dict,
    interpolation: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    matrix = fixed_pixel_to_output_pixel_matrix(settings)
    output_shape = tuple(settings["registration"]["output_shape"])
    return warp_image(image, matrix, output_shape, interpolation, border_value=0)


def warp_image(
    image: np.ndarray,
    matrix_source_to_output: np.ndarray,
    output_shape: tuple[int, int] | list[int],
    interpolation: str = "linear",
    border_value: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(output_shape[0]), int(output_shape[1])
    flags = _interpolation_flag(interpolation)
    source = _opencv_safe_array(np.asarray(image))

    warped = cv2.warpPerspective(
        source,
        np.asarray(matrix_source_to_output, dtype=np.float64),
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    source_mask = np.full(source.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        source_mask,
        np.asarray(matrix_source_to_output, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, mask > 0


def _interpolation_flag(interpolation: str) -> int:
    options = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }
    try:
        return options[interpolation.lower()]
    except KeyError as exc:
        raise RegistrationError(f"Unsupported interpolation mode: {interpolation}") from exc


def _opencv_safe_array(image: np.ndarray) -> np.ndarray:
    if image.dtype in {np.dtype("uint8"), np.dtype("uint16"), np.dtype("int16"), np.dtype("float32")}:
        return np.ascontiguousarray(image)
    if image.dtype == np.float64:
        return np.ascontiguousarray(image.astype(np.float32))
    if image.dtype.kind in {"i", "u", "f"}:
        return np.ascontiguousarray(image.astype(np.float32))
    raise RegistrationError(f"Unsupported image dtype for warping: {image.dtype}")


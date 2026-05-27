from __future__ import annotations

import threading
import tkinter as tk
from math import ceil
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .io import (
    SUPPORTED_EXTENSIONS,
    ang_available_visualizations,
    ang_to_image,
    as_display_rgb,
    load_ang,
    read_image,
    write_image,
)
from .overlay import make_overlay
from .settings import load_settings, save_settings
from .transforms import RegistrationError, build_registration_settings


IMAGE_FILETYPES = [
    ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg"),
    ("TIFF", "*.tif *.tiff"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("All files", "*.*"),
]


def magnifier_crop_rgb(
    image: np.ndarray,
    center_xy: tuple[float, float],
    radius: int,
) -> np.ndarray:
    source = np.asarray(image)
    height, width = source.shape[:2]
    radius = max(1, int(radius))
    crop_size = radius * 2 + 1
    center_x = int(np.clip(round(center_xy[0]), 0, width - 1))
    center_y = int(np.clip(round(center_xy[1]), 0, height - 1))

    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    paste_x0 = x0 - (center_x - radius)
    paste_y0 = y0 - (center_y - radius)

    if source.ndim == 2:
        padded = np.zeros((crop_size, crop_size), dtype=source.dtype)
        padded[paste_y0 : paste_y0 + (y1 - y0), paste_x0 : paste_x0 + (x1 - x0)] = source[y0:y1, x0:x1]
    else:
        padded = np.zeros((crop_size, crop_size, source.shape[2]), dtype=source.dtype)
        padded[paste_y0 : paste_y0 + (y1 - y0), paste_x0 : paste_x0 + (x1 - x0), :] = source[y0:y1, x0:x1, :]

    return as_display_rgb(padded)


class ImagePanel(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        click_callback=None,
        magnifier_enabled=None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.click_callback = click_callback
        self.magnifier_enabled = magnifier_enabled or (lambda: False)
        self.image: np.ndarray | None = None
        self.display_rgb: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.photo_key: tuple | None = None
        self.magnifier_photo: ImageTk.PhotoImage | None = None
        self.magnifier_point: tuple[float, float] | None = None
        self.magnifier_canvas_xy: tuple[int, int] | None = None
        self.magnifier_source_radius = 32
        self.points: list[tuple[float, float]] = []
        self.pending_point: tuple[float, float] | None = None
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_last_xy: tuple[int, int] | None = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_width = 1
        self.display_height = 1

        ttk.Label(self, text=title).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=430, height=340, bg="#1f1f1f", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._hide_magnifier)
        self.canvas.bind("<ButtonPress-2>", self._on_middle_press)
        self.canvas.bind("<B2-Motion>", self._on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def set_image(self, image: np.ndarray | None) -> None:
        self.image = image
        self.display_rgb = None
        self.photo = None
        self.photo_key = None
        self.magnifier_photo = None
        self.magnifier_point = None
        self.magnifier_canvas_xy = None
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        if image is not None:
            self.display_rgb = as_display_rgb(image)
        self.redraw()

    def set_points(
        self,
        points: list[tuple[float, float]] | None = None,
        pending_point: tuple[float, float] | None = None,
        redraw: bool = True,
    ) -> None:
        self.points = list(points or [])
        self.pending_point = pending_point
        if redraw:
            self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.display_rgb is None:
            self.canvas.create_text(
                max(10, self.canvas.winfo_width() // 2),
                max(10, self.canvas.winfo_height() // 2),
                text="No image loaded",
                fill="#c8c8c8",
            )
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_height, image_width = self.display_rgb.shape[:2]
        fit_scale = min(canvas_width / image_width, canvas_height / image_height)
        self.scale = fit_scale * self.zoom_factor
        self.display_width = max(1, int(round(image_width * self.scale)))
        self.display_height = max(1, int(round(image_height * self.scale)))
        self.offset_x, self.pan_x = self._offset_and_clamped_pan(
            canvas_size=canvas_width,
            display_size=self.display_width,
            pan=self.pan_x,
        )
        self.offset_y, self.pan_y = self._offset_and_clamped_pan(
            canvas_size=canvas_height,
            display_size=self.display_height,
            pan=self.pan_y,
        )

        # Crop only the visible portion of display_rgb to the canvas.
        img_x0 = max(0, int(-self.offset_x / self.scale))
        img_y0 = max(0, int(-self.offset_y / self.scale))
        img_x1 = min(image_width,  ceil((canvas_width  - self.offset_x) / self.scale))
        img_y1 = min(image_height, ceil((canvas_height - self.offset_y) / self.scale))
        crop = self.display_rgb[img_y0:img_y1, img_x0:img_x1]
        render_w = max(1, int(round((img_x1 - img_x0) * self.scale)))
        render_h = max(1, int(round((img_y1 - img_y0) * self.scale)))
        resample = Image.Resampling.NEAREST if self.scale >= 1.0 else Image.Resampling.BILINEAR

        photo_key = (id(self.display_rgb), img_x0, img_y0, img_x1, img_y1, render_w, render_h)
        if self.photo is None or self.photo_key != photo_key:
            pil_image = Image.fromarray(crop, mode="RGB")
            pil_image = pil_image.resize((render_w, render_h), resample)
            self.photo = ImageTk.PhotoImage(pil_image)
            self.photo_key = photo_key
        canvas_img_x = max(0, int(round(self.offset_x + img_x0 * self.scale)))
        canvas_img_y = max(0, int(round(self.offset_y + img_y0 * self.scale)))
        self.canvas.create_image(canvas_img_x, canvas_img_y, image=self.photo, anchor="nw")
        self._draw_points()
        self._draw_magnifier()

    def _draw_points(self) -> None:
        for index, point in enumerate(self.points, start=1):
            self._draw_marker(point, index, fill="#42d36b")
        if self.pending_point is not None:
            self._draw_marker(self.pending_point, len(self.points) + 1, fill="#ffd166")

    def _draw_marker(self, point: tuple[float, float], index: int, fill: str) -> None:
        x = self.offset_x + point[0] * self.scale
        y = self.offset_y + point[1] * self.scale
        radius = 5
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="black", width=2)
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=fill, width=2)
        self.canvas.create_text(x + 10, y - 10, text=str(index), fill=fill, anchor="w")

    def _on_click(self, event: tk.Event) -> None:
        if self.display_rgb is None or self.click_callback is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is not None:
            self.click_callback(point)

    def _on_motion(self, event: tk.Event) -> None:
        if self.display_rgb is None or self.click_callback is None or not self.magnifier_enabled():
            self._hide_magnifier()
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            self._hide_magnifier()
            return
        self.magnifier_point = point
        self.magnifier_canvas_xy = (int(event.x), int(event.y))
        self.redraw()

    def _on_middle_press(self, event: tk.Event) -> None:
        self._hide_magnifier()
        self._pan_last_xy = (int(event.x_root), int(event.y_root))

    def _on_middle_drag(self, event: tk.Event) -> None:
        current_xy = (int(event.x_root), int(event.y_root))
        if self._pan_last_xy is None:
            self._pan_last_xy = current_xy
            return
        dx = current_xy[0] - self._pan_last_xy[0]
        dy = current_xy[1] - self._pan_last_xy[1]
        self.pan_x += dx
        self.pan_y += dy
        self._pan_last_xy = current_xy
        self.redraw()

    def _on_middle_release(self, event: tk.Event) -> None:
        self._pan_last_xy = None

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        direction = self._wheel_direction(event)
        if direction == 0:
            return
        new_zoom = float(np.clip(self.zoom_factor * (1.2 if direction > 0 else 1.0 / 1.2), 1.0, 12.0))
        if np.isclose(new_zoom, self.zoom_factor):
            return
        self.pan_x, self.pan_y = self.pan_for_zoom_at(new_zoom, int(event.x), int(event.y))
        self.zoom_factor = new_zoom
        self.redraw()

    def _wheel_direction(self, event: tk.Event) -> int:
        if hasattr(event, "delta") and event.delta:
            return 1 if event.delta > 0 else -1
        button = getattr(event, "num", None)
        if button == 4:
            return 1
        if button == 5:
            return -1
        return 0

    def _hide_magnifier(self, _event: tk.Event | None = None) -> None:
        if self.magnifier_point is None and self.magnifier_canvas_xy is None:
            return
        self.magnifier_point = None
        self.magnifier_canvas_xy = None
        self.magnifier_photo = None
        self.redraw()

    def canvas_to_image(self, x: int, y: int) -> tuple[float, float] | None:
        if self.image is None:
            return None
        if not (self.offset_x <= x <= self.offset_x + self.display_width):
            return None
        if not (self.offset_y <= y <= self.offset_y + self.display_height):
            return None

        image_x = (x - self.offset_x) / self.scale
        image_y = (y - self.offset_y) / self.scale
        height, width = self.image.shape[:2]
        if not (0 <= image_x < width and 0 <= image_y < height):
            return None
        return float(image_x), float(image_y)

    def _draw_magnifier(self) -> None:
        if self.image is None or self.magnifier_point is None or self.magnifier_canvas_xy is None:
            return

        panel_size = self._magnifier_panel_size()
        crop_rgb = magnifier_crop_rgb(self.image, self.magnifier_point, self.magnifier_source_radius)
        pil_image = Image.fromarray(crop_rgb, mode="RGB")
        pil_image = pil_image.resize((panel_size, panel_size), Image.Resampling.NEAREST)
        self.magnifier_photo = ImageTk.PhotoImage(pil_image)

        x0, y0 = self._magnifier_position(panel_size)
        x1 = x0 + panel_size
        y1 = y0 + panel_size
        center_x = x0 + panel_size // 2
        center_y = y0 + panel_size // 2

        self.canvas.create_rectangle(x0 - 3, y0 - 3, x1 + 3, y1 + 3, fill="#101010", outline="#ffffff", width=1)
        self.canvas.create_image(x0, y0, image=self.magnifier_photo, anchor="nw")
        self.canvas.create_line(center_x, y0, center_x, y1, fill="#ffeb3b", width=1)
        self.canvas.create_line(x0, center_y, x1, center_y, fill="#ffeb3b", width=1)
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ffffff", width=1)

    def _magnifier_panel_size(self) -> int:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        return int(np.clip(min(canvas_width, canvas_height) * 0.42, 130, 200))

    def _magnifier_position(self, panel_size: int) -> tuple[int, int]:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        cursor_x, cursor_y = self.magnifier_canvas_xy or (0, 0)
        offset = 18
        x0 = cursor_x + offset
        y0 = cursor_y + offset
        if x0 + panel_size + 4 > canvas_width:
            x0 = cursor_x - panel_size - offset
        if y0 + panel_size + 4 > canvas_height:
            y0 = cursor_y - panel_size - offset
        x0 = int(np.clip(x0, 4, max(4, canvas_width - panel_size - 4)))
        y0 = int(np.clip(y0, 4, max(4, canvas_height - panel_size - 4)))
        return x0, y0

    def pan_for_zoom_at(self, zoom_factor: float, canvas_x: int, canvas_y: int) -> tuple[float, float]:
        if self.display_rgb is None:
            return self.pan_x, self.pan_y

        image_height, image_width = self.display_rgb.shape[:2]
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        preview_x = (canvas_x - self.offset_x) / max(self.scale, 1e-9)
        preview_y = (canvas_y - self.offset_y) / max(self.scale, 1e-9)
        preview_x = float(np.clip(preview_x, 0, image_width))
        preview_y = float(np.clip(preview_y, 0, image_height))

        fit_scale = min(canvas_width / image_width, canvas_height / image_height)
        new_scale = fit_scale * float(np.clip(zoom_factor, 1.0, 12.0))
        new_display_width = max(1, int(round(image_width * new_scale)))
        new_display_height = max(1, int(round(image_height * new_scale)))
        center_x = (canvas_width - new_display_width) / 2.0
        center_y = (canvas_height - new_display_height) / 2.0
        new_pan_x = canvas_x - preview_x * new_scale - center_x
        new_pan_y = canvas_y - preview_y * new_scale - center_y
        return new_pan_x, new_pan_y

    def _offset_and_clamped_pan(self, canvas_size: int, display_size: int, pan: float) -> tuple[int, float]:
        centered_offset = (canvas_size - display_size) / 2.0
        if display_size <= canvas_size:
            return int(round(centered_offset)), 0.0

        offset = centered_offset + pan
        min_offset = canvas_size - display_size
        max_offset = 0
        offset = float(np.clip(offset, min_offset, max_offset))
        return int(round(offset)), offset - centered_offset


class RegistrationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DataFusion Image Register")
        self.geometry("1320x820")

        self.fixed_image: np.ndarray | None = None
        self.moving_image: np.ndarray | None = None
        self.fixed_path: str | None = None
        self.moving_path: str | None = None
        self.fixed_points: list[tuple[float, float]] = []
        self.moving_points: list[tuple[float, float]] = []
        self.pending_fixed_point: tuple[float, float] | None = None
        self.current_settings: dict | None = None
        self.preview_overlay: np.ndarray | None = None
        self.preview_registered: np.ndarray | None = None
        self.preview_fixed: np.ndarray | None = None
        self.xmap = None
        self.ang_viz_var = tk.StringVar(value="IPF-Z")
        self.ang_miso_var = tk.StringVar(value="5.0")
        self.ang_transparent_bg_var = tk.BooleanVar(value=True)

        self.transform_type_var = tk.StringVar(value="affine")
        self.output_basis_var = tk.StringVar(value="fixed")
        self.fixed_pixel_size_var = tk.StringVar(value="1.0")
        self.moving_pixel_size_var = tk.StringVar(value="1.0")
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.scale_bar_length_var = tk.StringVar(value="10")
        self.scale_bar_units_var = tk.StringVar(value="um")
        self.scale_bar_font_size_var = tk.StringVar(value="")
        self.scale_bar_thickness_var = tk.StringVar(value="")
        self.crop_to_common_var = tk.BooleanVar(value=False)
        self.magnifier_enabled_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Load fixed and moving images, then click fixed/moving point pairs.")

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self)
        sidebar.grid(row=0, column=0, sticky="ns")
        self._ctrl_canvas = tk.Canvas(sidebar, width=175, highlightthickness=0, bd=0)
        ctrl_scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=self._ctrl_canvas.yview)
        self._ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)
        ctrl_scrollbar.pack(side="right", fill="y")
        self._ctrl_canvas.pack(side="left", fill="both", expand=True)
        controls = ttk.Frame(self._ctrl_canvas, padding=10)
        self._ctrl_window_id = self._ctrl_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", self._on_ctrl_frame_configure)
        self._ctrl_canvas.bind("<Configure>", self._on_ctrl_canvas_configure)
        self._ctrl_canvas.bind("<MouseWheel>", self._on_ctrl_scroll)
        controls.bind("<MouseWheel>", self._on_ctrl_scroll)

        viewer = ttk.Frame(self, padding=(0, 10, 10, 10))
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.columnconfigure(1, weight=1)
        viewer.rowconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)

        self.fixed_panel = ImagePanel(
            viewer,
            "Fixed / Reference",
            self._add_fixed_point,
            magnifier_enabled=self._magnifier_enabled,
        )
        self.fixed_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.moving_panel = ImagePanel(
            viewer,
            "Moving",
            self._add_moving_point,
            magnifier_enabled=self._magnifier_enabled,
        )
        self.moving_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 8))

        self.overlay_panel = ImagePanel(viewer, "Overlay Preview")
        self.overlay_panel.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self._build_controls(controls)

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _on_ctrl_frame_configure(self, _event: tk.Event) -> None:
        self._ctrl_canvas.configure(scrollregion=self._ctrl_canvas.bbox("all"))

    def _on_ctrl_canvas_configure(self, event: tk.Event) -> None:
        self._ctrl_canvas.itemconfigure(self._ctrl_window_id, width=event.width)

    def _on_ctrl_scroll(self, event: tk.Event) -> None:
        self._ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_controls(self, parent: ttk.Frame) -> None:
        ttk.Button(parent, text="Load Fixed", command=self._load_fixed).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Load Moving", command=self._load_moving).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Load .ang", command=self._load_ang).pack(fill="x", pady=(0, 4))

        ttk.Label(parent, text=".ang visualization").pack(anchor="w", pady=(4, 0))
        self._ang_viz_combo = ttk.Combobox(
            parent, textvariable=self.ang_viz_var, values=["IPF-Z"],
            state="readonly", width=16,
        )
        self._ang_viz_combo.pack(fill="x", pady=(0, 2))
        self._entry(parent, "Min misorientation (°)", self.ang_miso_var)
        ttk.Checkbutton(
            parent, text="Transparent background",
            variable=self.ang_transparent_bg_var,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Button(parent, text="Apply Visualization",
                   command=self._apply_ang_viz).pack(fill="x", pady=(4, 12))

        ttk.Label(parent, text="Transform").pack(anchor="w")
        ttk.Combobox(
            parent,
            textvariable=self.transform_type_var,
            values=("affine", "homography", "morphops_tps"),
            state="readonly",
        ).pack(fill="x", pady=(0, 8))

        ttk.Label(parent, text="Output pixels").pack(anchor="w")
        ttk.Combobox(
            parent,
            textvariable=self.output_basis_var,
            values=("fixed", "moving"),
            state="readonly",
        ).pack(fill="x", pady=(0, 8))

        self._entry(parent, "Fixed pixel size", self.fixed_pixel_size_var)
        self._entry(parent, "Moving pixel size", self.moving_pixel_size_var)

        ttk.Label(parent, text="Blend alpha").pack(anchor="w", pady=(8, 0))
        ttk.Scale(
            parent,
            from_=0.0,
            to=1.0,
            variable=self.alpha_var,
            command=lambda _value: self._refresh_preview_if_ready(),
        ).pack(fill="x")

        self._entry(parent, "Scale bar length", self.scale_bar_length_var)
        self._entry(parent, "Scale bar units", self.scale_bar_units_var)
        self._entry(parent, "Scale bar font px", self.scale_bar_font_size_var)
        self._entry(parent, "Scale bar height px", self.scale_bar_thickness_var)
        ttk.Checkbutton(
            parent,
            text="Magnifier",
            variable=self.magnifier_enabled_var,
            command=self._toggle_magnifier,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(
            parent,
            text="Crop to common area",
            variable=self.crop_to_common_var,
            command=self._refresh_preview_if_ready,
        ).pack(anchor="w", pady=(8, 0))

        ttk.Separator(parent).pack(fill="x", pady=12)
        ttk.Button(parent, text="Preview / Estimate", command=self._estimate_and_preview).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Undo", command=self._undo).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Reset Points", command=self._reset_points).pack(fill="x", pady=(0, 12))

        ttk.Button(parent, text="Save Settings", command=self._save_settings).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Load Settings", command=self._load_settings).pack(fill="x", pady=(0, 12))

        ttk.Button(parent, text="Export Overlay", command=self._export_overlay).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Export Registered", command=self._export_registered).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Export Fixed", command=self._export_fixed).pack(fill="x")

    def _entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=variable, width=18).pack(fill="x", pady=(0, 2))

    def _load_fixed(self) -> None:
        path = self._ask_image_path()
        if path is None:
            return
        self._set_status(f"Loading fixed image: {Path(path).name}")
        self.update_idletasks()
        self.fixed_image = self._read_image_or_alert(path)
        if self.fixed_image is None:
            return
        self.fixed_path = str(path)
        self.fixed_panel.set_image(self.fixed_image)
        self.current_settings = None
        self.preview_overlay = None
        self.overlay_panel.set_image(None)
        self._set_status(f"Loaded fixed image: {Path(path).name}")

    def _load_moving(self) -> None:
        path = self._ask_image_path()
        if path is None:
            return
        self._set_status(f"Loading moving image: {Path(path).name} …")

        def _load() -> None:
            try:
                image = read_image(path)
                self.after(0, lambda: self._on_moving_loaded(image, path))
            except Exception as exc:
                self.after(0, lambda: self._on_moving_load_error(exc))

        threading.Thread(target=_load, daemon=True).start()

    def _on_moving_loaded(self, image: np.ndarray, path: Path) -> None:
        self.moving_image = image
        self.moving_path = str(path)
        self._reset_points(redraw=False)
        self.fixed_panel.set_points(self.fixed_points, self.pending_fixed_point)
        self.moving_panel.set_points(self.moving_points, redraw=False)
        self.moving_panel.set_image(self.moving_image)
        self.current_settings = None
        self.preview_overlay = None
        self.overlay_panel.set_image(None)
        self._set_status(f"Loaded moving image: {Path(path).name}")

    def _on_moving_load_error(self, exc: Exception) -> None:
        messagebox.showerror("Image Load Error", str(exc))
        self._set_status("Failed to load moving image.")

    def _load_ang(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("EDAX ANG", "*.ang"), ("All files", "*.*")])
        if not path:
            return
        path = Path(path)
        self._set_status(f"Loading .ang: {path.name} …")

        def _load() -> None:
            try:
                xmap = load_ang(path)
                image = ang_to_image(xmap, "IPF-Z")
                self.after(0, lambda: self._on_ang_loaded(xmap, image, path))
            except Exception as exc:
                self.after(0, lambda: self._on_moving_load_error(exc))

        threading.Thread(target=_load, daemon=True).start()

    def _on_ang_loaded(self, xmap, image: np.ndarray, path: Path) -> None:
        self.xmap = xmap
        self.ang_viz_var.set("IPF-Z")
        self._ang_viz_combo.configure(values=ang_available_visualizations(xmap))
        self._on_moving_loaded(image, path)

    def _apply_ang_viz(self) -> None:
        if self.xmap is None:
            self._set_status("Load a .ang file first.")
            return
        viz = self.ang_viz_var.get()
        try:
            threshold = float(self.ang_miso_var.get())
        except ValueError:
            self._set_status("Min misorientation must be a number.")
            return
        transparent = bool(self.ang_transparent_bg_var.get())
        self._set_status(f"Generating {viz} …")

        def _compute() -> None:
            try:
                image = ang_to_image(self.xmap, viz, threshold, transparent)
                self.after(0, lambda: self._update_moving_display(image))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Visualization Error", str(exc)))

        threading.Thread(target=_compute, daemon=True).start()

    def _update_moving_display(self, image: np.ndarray) -> None:
        self.moving_image = image
        self.moving_panel.set_image(image)
        self.current_settings = None
        self.preview_overlay = None
        self.overlay_panel.set_image(None)
        self._set_status("Visualization updated.")

    def _read_image_or_alert(self, path: Path) -> np.ndarray | None:
        try:
            return read_image(path)
        except Exception as exc:
            messagebox.showerror("Image Load Error", str(exc))
            return None

    def _ask_image_path(self) -> Path | None:
        path = filedialog.askopenfilename(filetypes=IMAGE_FILETYPES)
        return Path(path) if path else None

    def _magnifier_enabled(self) -> bool:
        return bool(self.magnifier_enabled_var.get())

    def _toggle_magnifier(self) -> None:
        if self._magnifier_enabled():
            return
        self.fixed_panel._hide_magnifier()
        self.moving_panel._hide_magnifier()

    def _add_fixed_point(self, point: tuple[float, float]) -> None:
        if self.fixed_image is None or self.moving_image is None:
            self._set_status("Load fixed and moving images before adding points.")
            return
        if self.pending_fixed_point is not None:
            self._set_status("Click the matching moving point before choosing another fixed point.")
            return
        self.pending_fixed_point = point
        self._update_point_panels()
        self._set_status(f"Fixed point {len(self.fixed_points) + 1} selected. Click its matching moving point.")

    def _add_moving_point(self, point: tuple[float, float]) -> None:
        if self.pending_fixed_point is None:
            self._set_status("Click a fixed/reference point first.")
            return
        self.fixed_points.append(self.pending_fixed_point)
        self.moving_points.append(point)
        self.pending_fixed_point = None
        self.current_settings = None
        self._update_point_panels()
        self._set_status(f"Added point pair {len(self.fixed_points)}.")

    def _update_point_panels(self, redraw: bool = True) -> None:
        self.fixed_panel.set_points(self.fixed_points, self.pending_fixed_point, redraw=redraw)
        self.moving_panel.set_points(self.moving_points, redraw=redraw)

    def _undo(self) -> None:
        if self.pending_fixed_point is not None:
            self.pending_fixed_point = None
        elif self.fixed_points and self.moving_points:
            self.fixed_points.pop()
            self.moving_points.pop()
        self.current_settings = None
        self._update_point_panels()
        self._set_status(f"Point pairs: {len(self.fixed_points)}")

    def _reset_points(self, redraw: bool = True) -> None:
        self.fixed_points.clear()
        self.moving_points.clear()
        self.pending_fixed_point = None
        self.current_settings = None
        self._update_point_panels(redraw=redraw)
        self._set_status("Points reset.")

    def _estimate_and_preview(self) -> None:
        if self.fixed_image is None or self.moving_image is None:
            self._set_status("Load fixed and moving images first.")
            return
        try:
            self.current_settings = self._build_settings()
            self._render_preview(self.moving_image)
        except Exception as exc:
            messagebox.showerror("Registration Error", str(exc))

    def _build_settings(self) -> dict:
        if self.fixed_image is None or self.moving_image is None:
            raise RegistrationError("Load fixed and moving images first.")
        return build_registration_settings(
            fixed_points=self.fixed_points,
            moving_points=self.moving_points,
            transform_type=self.transform_type_var.get(),
            fixed_pixel_size=self._positive_float(self.fixed_pixel_size_var.get(), "Fixed pixel size"),
            moving_pixel_size=self._positive_float(self.moving_pixel_size_var.get(), "Moving pixel size"),
            output_basis=self.output_basis_var.get(),
            fixed_shape=self.fixed_image.shape[:2],
            moving_shape=self.moving_image.shape[:2],
            fixed_path=self.fixed_path,
            moving_path=self.moving_path,
        )

    def _render_preview(self, source_image: np.ndarray) -> None:
        if self.fixed_image is None or self.current_settings is None:
            return
        alpha = float(self.alpha_var.get())
        scale_length = self._optional_positive_float(self.scale_bar_length_var.get(), "Scale bar length")
        units = self.scale_bar_units_var.get().strip() or "units"
        scale_bar_font_size = self._optional_positive_int(
            self.scale_bar_font_size_var.get(),
            "Scale bar font px",
        )
        scale_bar_thickness = self._optional_positive_int(
            self.scale_bar_thickness_var.get(),
            "Scale bar height px",
        )
        overlay, registered, fixed_output = make_overlay(
            fixed_image=self.fixed_image,
            moving_or_modality_image=source_image,
            settings=self.current_settings,
            alpha=alpha,
            scale_bar_length=scale_length,
            scale_bar_units=units,
            crop_to_common=bool(self.crop_to_common_var.get()),
            scale_bar_font_size=scale_bar_font_size,
            scale_bar_thickness=scale_bar_thickness,
        )
        self.preview_overlay = overlay
        self.preview_registered = registered
        self.preview_fixed = fixed_output
        self.overlay_panel.set_image(overlay)
        registration = self.current_settings["registration"]
        self._set_status(
            "Preview ready. "
            f"Pairs: {len(self.fixed_points)}, "
            f"mean residual: {registration['mean_residual']:.4g}, "
            f"max residual: {registration['max_residual']:.4g}"
        )

    def _refresh_preview_if_ready(self) -> None:
        if self.current_settings is None or self.moving_image is None:
            return
        self._render_preview(self.moving_image)

    def _save_settings(self) -> None:
        try:
            if self.current_settings is None:
                self.current_settings = self._build_settings()
            settings = dict(self.current_settings)
            settings["display"] = {
                "alpha": float(self.alpha_var.get()),
                "scale_bar_length": self._optional_positive_float(
                    self.scale_bar_length_var.get(), "Scale bar length"
                ),
                "scale_bar_units": self.scale_bar_units_var.get().strip() or "units",
                "crop_to_common": bool(self.crop_to_common_var.get()),
                "scale_bar_font_size": self._optional_positive_int(
                    self.scale_bar_font_size_var.get(),
                    "Scale bar font px",
                ),
                "scale_bar_thickness": self._optional_positive_int(
                    self.scale_bar_thickness_var.get(),
                    "Scale bar height px",
                ),
            }
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            save_settings(path, settings)
            self._set_status(f"Saved settings: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Save Settings Error", str(exc))

    def _load_settings(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            settings = load_settings(path)
            registration = settings["registration"]
            self.current_settings = settings
            self.transform_type_var.set(registration["transform_type"])
            self.output_basis_var.set(registration["output_basis"])
            self.fixed_pixel_size_var.set(str(registration["fixed_pixel_size"]))
            self.moving_pixel_size_var.set(str(registration["moving_pixel_size"]))
            self.fixed_points = [tuple(point) for point in registration["fixed_points"]]
            self.moving_points = [tuple(point) for point in registration["moving_points"]]
            self.pending_fixed_point = None
            display = settings.get("display", {})
            if "alpha" in display:
                self.alpha_var.set(float(display["alpha"]))
            if display.get("scale_bar_length") is not None:
                self.scale_bar_length_var.set(str(display["scale_bar_length"]))
            if display.get("scale_bar_units"):
                self.scale_bar_units_var.set(str(display["scale_bar_units"]))
            self.scale_bar_font_size_var.set(self._display_optional_int(display.get("scale_bar_font_size")))
            self.scale_bar_thickness_var.set(self._display_optional_int(display.get("scale_bar_thickness")))
            self.crop_to_common_var.set(bool(display.get("crop_to_common", False)))
            self._update_point_panels()
            self._set_status(f"Loaded settings: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Load Settings Error", str(exc))

    def _export_overlay(self) -> None:
        if self.preview_overlay is None:
            self._set_status("Create a preview before exporting an overlay.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif *.tiff"), ("JPEG", "*.jpg *.jpeg")],
        )
        if not path:
            return
        try:
            write_image(path, self.preview_overlay)
            self._set_status(f"Exported overlay: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Export Overlay Error", str(exc))

    def _export_registered(self) -> None:
        if self.preview_registered is None:
            self._set_status("Create a preview before exporting a registered image.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif *.tiff"), ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")],
        )
        if not path:
            return
        try:
            write_image(path, self.preview_registered)
            self._set_status(f"Exported registered image: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Export Registered Error", str(exc))

    def _export_fixed(self) -> None:
        if self.preview_fixed is None:
            self._set_status("Create a preview before exporting the fixed image.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif *.tiff"), ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")],
        )
        if not path:
            return
        try:
            write_image(path, self.preview_fixed)
            self._set_status(f"Exported fixed image: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Export Fixed Error", str(exc))

    def _positive_float(self, value: str, label: str) -> float:
        parsed = float(value)
        if parsed <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        return parsed

    def _optional_positive_float(self, value: str, label: str) -> float | None:
        if not value.strip():
            return None
        return self._positive_float(value, label)

    def _optional_positive_int(self, value: str, label: str) -> int | None:
        if not value.strip():
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a whole number.") from exc
        if parsed <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        return parsed

    def _display_optional_int(self, value: object) -> str:
        if value is None or value == "":
            return ""
        return str(int(value))

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)


def main() -> None:
    app = RegistrationApp()
    app.mainloop()


if __name__ == "__main__":
    main()

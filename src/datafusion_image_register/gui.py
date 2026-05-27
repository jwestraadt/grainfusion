from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .io import SUPPORTED_EXTENSIONS, as_display_rgb, read_image, write_image
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


class ImagePanel(ttk.Frame):
    def __init__(self, parent: tk.Widget, title: str, click_callback=None) -> None:
        super().__init__(parent)
        self.title = title
        self.click_callback = click_callback
        self.image: np.ndarray | None = None
        self.display_rgb: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.points: list[tuple[float, float]] = []
        self.pending_point: tuple[float, float] | None = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_width = 1
        self.display_height = 1

        ttk.Label(self, text=title).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=430, height=340, bg="#1f1f1f", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def set_image(self, image: np.ndarray | None) -> None:
        self.image = image
        self.display_rgb = as_display_rgb(image) if image is not None else None
        self.redraw()

    def set_points(
        self,
        points: list[tuple[float, float]] | None = None,
        pending_point: tuple[float, float] | None = None,
    ) -> None:
        self.points = list(points or [])
        self.pending_point = pending_point
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
        self.scale = min(canvas_width / image_width, canvas_height / image_height)
        self.display_width = max(1, int(round(image_width * self.scale)))
        self.display_height = max(1, int(round(image_height * self.scale)))
        self.offset_x = (canvas_width - self.display_width) // 2
        self.offset_y = (canvas_height - self.display_height) // 2

        pil_image = Image.fromarray(self.display_rgb, mode="RGB")
        pil_image = pil_image.resize((self.display_width, self.display_height), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw")
        self._draw_points()

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


class RegistrationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DataFusion Image Register")
        self.geometry("1320x820")

        self.fixed_image: np.ndarray | None = None
        self.moving_image: np.ndarray | None = None
        self.modality_image: np.ndarray | None = None
        self.fixed_path: str | None = None
        self.moving_path: str | None = None
        self.modality_path: str | None = None
        self.fixed_points: list[tuple[float, float]] = []
        self.moving_points: list[tuple[float, float]] = []
        self.pending_fixed_point: tuple[float, float] | None = None
        self.current_settings: dict | None = None
        self.preview_overlay: np.ndarray | None = None
        self.preview_registered: np.ndarray | None = None
        self.preview_source: str = "moving"

        self.transform_type_var = tk.StringVar(value="affine")
        self.output_basis_var = tk.StringVar(value="fixed")
        self.fixed_pixel_size_var = tk.StringVar(value="1.0")
        self.moving_pixel_size_var = tk.StringVar(value="1.0")
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.scale_bar_length_var = tk.StringVar(value="10")
        self.scale_bar_units_var = tk.StringVar(value="um")
        self.crop_to_common_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Load fixed and moving images, then click fixed/moving point pairs.")

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=10)
        controls.grid(row=0, column=0, sticky="ns")

        viewer = ttk.Frame(self, padding=(0, 10, 10, 10))
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.columnconfigure(1, weight=1)
        viewer.rowconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)

        self.fixed_panel = ImagePanel(viewer, "Fixed / Reference", self._add_fixed_point)
        self.fixed_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.moving_panel = ImagePanel(viewer, "Moving", self._add_moving_point)
        self.moving_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 8))

        self.overlay_panel = ImagePanel(viewer, "Overlay Preview")
        self.overlay_panel.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self._build_controls(controls)

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_controls(self, parent: ttk.Frame) -> None:
        ttk.Button(parent, text="Load Fixed", command=self._load_fixed).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Load Moving", command=self._load_moving).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Load Modality", command=self._load_modality).pack(fill="x", pady=(0, 12))

        ttk.Label(parent, text="Transform").pack(anchor="w")
        ttk.Combobox(
            parent,
            textvariable=self.transform_type_var,
            values=("affine", "homography"),
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
        ttk.Checkbutton(
            parent,
            text="Crop to common area",
            variable=self.crop_to_common_var,
            command=self._refresh_preview_if_ready,
        ).pack(anchor="w", pady=(8, 0))

        ttk.Separator(parent).pack(fill="x", pady=12)
        ttk.Button(parent, text="Preview / Estimate", command=self._estimate_and_preview).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Apply Modality", command=self._apply_modality).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Undo", command=self._undo).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Reset Points", command=self._reset_points).pack(fill="x", pady=(0, 12))

        ttk.Button(parent, text="Save Settings", command=self._save_settings).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Load Settings", command=self._load_settings).pack(fill="x", pady=(0, 12))

        ttk.Button(parent, text="Export Overlay", command=self._export_overlay).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Export Registered", command=self._export_registered).pack(fill="x")

    def _entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=variable, width=18).pack(fill="x", pady=(0, 2))

    def _load_fixed(self) -> None:
        path = self._ask_image_path()
        if path is None:
            return
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
        self.moving_image = self._read_image_or_alert(path)
        if self.moving_image is None:
            return
        self.moving_path = str(path)
        self.moving_panel.set_image(self.moving_image)
        self._reset_points()
        self.current_settings = None
        self.preview_overlay = None
        self.overlay_panel.set_image(None)
        self._set_status(f"Loaded moving image: {Path(path).name}")

    def _load_modality(self) -> None:
        path = self._ask_image_path()
        if path is None:
            return
        self.modality_image = self._read_image_or_alert(path)
        if self.modality_image is None:
            return
        self.modality_path = str(path)
        self._set_status(f"Loaded modality image: {Path(path).name}")

    def _read_image_or_alert(self, path: Path) -> np.ndarray | None:
        try:
            return read_image(path)
        except Exception as exc:
            messagebox.showerror("Image Load Error", str(exc))
            return None

    def _ask_image_path(self) -> Path | None:
        path = filedialog.askopenfilename(filetypes=IMAGE_FILETYPES)
        return Path(path) if path else None

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

    def _update_point_panels(self) -> None:
        self.fixed_panel.set_points(self.fixed_points, self.pending_fixed_point)
        self.moving_panel.set_points(self.moving_points)

    def _undo(self) -> None:
        if self.pending_fixed_point is not None:
            self.pending_fixed_point = None
        elif self.fixed_points and self.moving_points:
            self.fixed_points.pop()
            self.moving_points.pop()
        self.current_settings = None
        self._update_point_panels()
        self._set_status(f"Point pairs: {len(self.fixed_points)}")

    def _reset_points(self) -> None:
        self.fixed_points.clear()
        self.moving_points.clear()
        self.pending_fixed_point = None
        self.current_settings = None
        self._update_point_panels()
        self._set_status("Points reset.")

    def _estimate_and_preview(self) -> None:
        if self.fixed_image is None or self.moving_image is None:
            self._set_status("Load fixed and moving images first.")
            return
        try:
            self.current_settings = self._build_settings()
            self.preview_source = "moving"
            self._render_preview(self.moving_image)
        except Exception as exc:
            messagebox.showerror("Registration Error", str(exc))

    def _apply_modality(self) -> None:
        if self.modality_image is None:
            self._set_status("Load a modality image first.")
            return
        try:
            if self.current_settings is None:
                self.current_settings = self._build_settings()
            self.preview_source = "modality"
            self._render_preview(self.modality_image)
        except Exception as exc:
            messagebox.showerror("Modality Apply Error", str(exc))

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
        overlay, registered, _fixed_output = make_overlay(
            fixed_image=self.fixed_image,
            moving_or_modality_image=source_image,
            settings=self.current_settings,
            alpha=alpha,
            scale_bar_length=scale_length,
            scale_bar_units=units,
            crop_to_common=bool(self.crop_to_common_var.get()),
        )
        self.preview_overlay = overlay
        self.preview_registered = registered
        self.overlay_panel.set_image(overlay)
        registration = self.current_settings["registration"]
        self._set_status(
            "Preview ready. "
            f"Pairs: {len(self.fixed_points)}, "
            f"mean residual: {registration['mean_residual']:.4g}, "
            f"max residual: {registration['max_residual']:.4g}"
        )

    def _refresh_preview_if_ready(self) -> None:
        if self.current_settings is None:
            return
        if self.preview_source == "modality" and self.modality_image is not None:
            self._render_preview(self.modality_image)
        elif self.moving_image is not None:
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

    def _positive_float(self, value: str, label: str) -> float:
        parsed = float(value)
        if parsed <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        return parsed

    def _optional_positive_float(self, value: str, label: str) -> float | None:
        if not value.strip():
            return None
        return self._positive_float(value, label)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)


def main() -> None:
    app = RegistrationApp()
    app.mainloop()


if __name__ == "__main__":
    main()

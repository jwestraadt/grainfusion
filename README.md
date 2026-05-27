# DataFusion Image Register

Small Python GUI for registering two images with corresponding clicked points.

## Install

Install `uv` first if it is not already available on your PATH.

```powershell
uv sync
```

## Run

```powershell
uv run image-register
```

## Basic Workflow

1. Load a fixed/reference image.
2. Load a moving image.
3. Enter fixed and moving pixel sizes in the same physical unit, for example `um/pixel`.
4. Choose `affine`, `homography`, or `morphops_tps`.
5. Click corresponding points in sequence:
   - fixed point 1
   - moving point 1
   - fixed point 2
   - moving point 2
6. Click `Preview / Estimate`.
7. Adjust the alpha slider.
8. Leave scale-bar font/height blank for automatic sizing, or enter pixel values to override them.
9. Enable `Crop to common area` if you want the preview/export trimmed to the overlap.
10. Save the correction JSON.
11. Load another modality image that shares the moving image coordinate system.
12. Click `Apply Modality`.
13. Export the overlay.

Affine needs at least 3 point pairs. Homography needs at least 4 point pairs.
Morphops TPS needs at least 3 non-collinear point pairs and usually benefits from more landmarks.

## Notes

- `Output pixels = fixed` keeps the output on the fixed image grid.
- `Output pixels = moving` keeps the fixed image physical extent but samples it at the moving image pixel size.
- The scale bar uses the selected output pixel size, so the fixed and moving pixel sizes must be entered in the same unit.
- Scale-bar font size, bar height, and margin are automatic by default and scale with the exported image size.
- Scale-bar labels are rounded to whole-number values.
- The common-area crop uses the rectangular bounding box where the fixed output and warped moving/modality image overlap.
- Large images are shown with downsampled interactive previews, while registration and export still use the full-resolution image data.
- `morphops_tps` uses Morphops thin-plate spline warping. It is non-linear and uses all selected point pairs directly.
- The saved JSON separates geometric registration settings from display settings such as alpha, scale bar length, scale-bar styling, and crop mode.

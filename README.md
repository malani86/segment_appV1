# Lipid Droplet Segmentation Application(segment_appV1)

A PySide6 desktop application for segmenting and quantifying lipid droplets in
microscopy images using a UNetDC model.

## Features

- Segment PNG, JPG, JPEG, TIFF, and TIFF stack images.
- Choose TIFF current slice, max projection, or all slices.
- Save predicted binary masks and green overlays.
- Quantify droplets into per-image CSV, combined CSV, and Excel reports.
- Show per-image summaries, total droplet count, and size statistics.
- Manually delete incorrect droplets after segmentation.
- Save and load analysis sessions as `.joblib` files.

## Project Structure

```text
segment_app/
├── main.py                      application entry point
├── main_window.py               main GUI window and user actions
├── controller.py                app state and run request management
├── backend.py                   command builder for batch segmentation
├── workers.py                   background process runner
├── quantify_droplets_batch.py   segmentation and quantification pipeline
├── preview_service.py           loads original/mask/overlay previews
├── imaging.py                   image and TIFF preview helpers
├── app_models.py                shared dataclasses and constants
├── config.py                    default settings
├── widgets.py                   reusable GUI widgets
├── models/                      neural network definitions
└── utils/                       training/preprocessing/metrics helpers
```

## Install

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the GUI

```bash
python segment_app/main.py
```

## Run the Batch Script Directly

```bash
python segment_app/quantify_droplets_batch.py \
  --img_dir path/to/images \
  --ckpt_path best_UNetDC_focal_model.pth \
  --out_dir quant_results \
  --save_overlays
```

For TIFF stacks, useful options are:

```bash
--tiff_mode current_slice
--tiff_mode max_projection
--tiff_mode all_slices
--tiff_as_png_style
```

## Outputs

The analysis creates files such as:

```text
predicted_masks/*_pred.png
overlays/*_overlay.png
*_droplets.csv
all_droplets.csv
summary_per_image.csv
droplet_size_stats.csv
all_droplets.xlsx
size_histogram.png
```

## License Note

This project root includes an MIT license for original project files unless an
individual file says otherwise.

Before publishing, check whether you want to keep `segment_app/algorithms.py`.
It appears unused by the current app and contains a BlobInspector GPL header, so
remove it from the published project if you want a cleaner MIT-only release.

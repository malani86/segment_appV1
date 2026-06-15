from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

from app_models import AppSettings
from config import SCRIPT_BASENAME, VALID_EXTS


def _is_tiff_path(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _resolve_batch_script() -> Path:
    start = Path(__file__).resolve()
    exe_path = Path(sys.argv[0]).resolve()
    frozen_dir = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else None

    if frozen_dir and frozen_dir.exists():
        search_roots = (frozen_dir, exe_path.parent, Path.cwd(), start.parent, *start.parents)
    else:
        search_roots = (exe_path.parent, Path.cwd(), start.parent, *start.parents)

    exe_name = SCRIPT_BASENAME + (".exe" if getattr(sys, "frozen", False) else ".py")
    candidates = []
    seen = set()
    for directory in search_roots:
        if directory in seen:
            continue
        seen.add(directory)
        candidates.append(directory / exe_name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    locations = "\n - ".join(str(path.parent) for path in candidates)
    raise FileNotFoundError(f"Could not locate {exe_name}. Looked in:\n -  " + locations)


def build_batch_command(
    *,
    input_dir: Path,
    out_dir: Path,
    settings: AppSettings,
    current_slice_index: int,
    tiff_png_reference_dir: Path | None = None,
) -> list[str]:
    script_path = _resolve_batch_script()

    args = ([str(script_path)] if getattr(sys, "frozen", False) else [sys.executable, str(script_path)]) + [
        "--img_dir", str(input_dir),
        "--ckpt_path", str(settings.checkpoint_path),
        "--out_dir", str(out_dir),
        "--batch", str(settings.batch_size),
        "--prob_thresh", str(settings.threshold),
        "--min_area", str(settings.min_area),
        "--background_radius", str(settings.background_radius),
        "--resize_size", str(settings.resize_size),
        "--overlay_alpha", str(settings.overlay_alpha),
        "--tiff_mode", str(settings.tiff_stack_mode),
        "--slice_index", str(max(0, current_slice_index)),
    ]

    if settings.px_per_micron > 0:
        args.extend(["--px_per_micron", str(settings.px_per_micron)])
    if settings.save_overlays:
        args.append("--save_overlays")
    if not settings.save_masks:
        args.append("--skip_mask_save")
    if not settings.automatic_quantification:
        args.append("--skip_quantification")
    if settings.use_watershed_count:
        args.append("--use_watershed_count")
    if settings.tiff_as_png_style:
        args.append("--tiff_as_png_style")
        if tiff_png_reference_dir is not None:
            args.extend(["--tiff_png_reference_dir", str(tiff_png_reference_dir)])
    if not settings.excel_enabled:
        args.append("--skip_excel")
    if not settings.histogram_enabled:
        args.append("--skip_histogram")
    if settings.debug_preprocessed_match:
        args.extend(["--debug_preprocessed_match", str(settings.debug_preprocessed_match)])
    if settings.debug_preprocessed_dir:
        args.extend(["--debug_preprocessed_dir", str(settings.debug_preprocessed_dir)])
    return args


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    except Exception:
        return []


def prepare_run(
    *,
    folder_text: str,
    out_dir_text: str,
    settings: AppSettings,
    current_slice_index: int,
) -> tuple[Path, list[str], Path]:
    ckpt_path = Path(settings.checkpoint_path)
    if not ckpt_path.is_file():
        raise ValueError("Checkpoint file does not exist.")

    base_out_dir = Path(out_dir_text.strip()) if out_dir_text.strip() else Path("quant_results")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    folder_path = Path(folder_text.strip())
    if not folder_path.is_dir():
        raise ValueError("Please select a valid input folder.")
    image_count = sum(1 for p in folder_path.iterdir() if p.suffix.lower() in VALID_EXTS)
    if image_count == 0:
        raise ValueError("The selected folder contains no supported images.")

    input_dir = folder_path
    out_dir = base_out_dir

    command = build_batch_command(
        input_dir=input_dir,
        out_dir=out_dir,
        settings=settings,
        current_slice_index=current_slice_index,
        tiff_png_reference_dir=input_dir,
    )

    return folder_path, command, out_dir


def prepare_single_image_run(
    *,
    image_path: Path,
    out_dir_text: str,
    settings: AppSettings,
    current_slice_index: int,
) -> tuple[Path, list[str], Path, Path]:
    if not image_path.is_file():
        raise ValueError("Please select a valid image file.")
    if image_path.suffix.lower() not in VALID_EXTS:
        raise ValueError("The selected file is not a supported image.")

    ckpt_path = Path(settings.checkpoint_path)
    if not ckpt_path.is_file():
        raise ValueError("Checkpoint file does not exist.")

    base_out_dir = Path(out_dir_text.strip()) if out_dir_text.strip() else Path("quant_results")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    single_out_dir = base_out_dir / "single_runs" / image_path.stem
    single_out_dir.mkdir(parents=True, exist_ok=True)

    temp_input_dir = Path(tempfile.mkdtemp(prefix="segment_app_single_"))
    staged_image = temp_input_dir / image_path.name
    shutil.copy2(image_path, staged_image)

    command = build_batch_command(
        input_dir=temp_input_dir,
        out_dir=single_out_dir,
        settings=settings,
        current_slice_index=current_slice_index,
        tiff_png_reference_dir=image_path.parent if _is_tiff_path(image_path) else None,
    )
    return image_path, command, single_out_dir, temp_input_dir

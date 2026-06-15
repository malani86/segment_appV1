from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_BACKGROUND_RADIUS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    DEFAULT_EXCEL_ENABLED,
    DEFAULT_HISTOGRAM_ENABLED,
    DEFAULT_MIN_AREA,
    DEFAULT_OVERLAY_ALPHA,
    DEFAULT_PX_PER_MICRON,
    DEFAULT_RESIZE_SIZE,
    DEFAULT_SAVE_MASKS,
    DEFAULT_SAVE_OVERLAYS,
    DEFAULT_TIFF_AS_PNG_STYLE,
    DEFAULT_USE_WATERSHED_COUNT,
    DEFAULT_AUTO_QUANTIFICATION,
    DEFAULT_TIFF_STACK_MODE,
    DEFAULT_THRESHOLD,
)


VIEWER_MODE_ORIGINAL = "original"
VIEWER_MODE_MASK = "mask"
VIEWER_MODE_OVERLAY = "overlay"
STACK_VIEW_SLICE = "slice"
STACK_VIEW_PROJECTION = "projection"
WORKFLOW_STEP_LOAD = "load"
WORKFLOW_STEP_PREVIEW = "preview"
WORKFLOW_STEP_SEGMENT = "segment"
WORKFLOW_STEP_REVIEW = "review"
WORKFLOW_STEP_QUANTIFY = "quantify"
WORKFLOW_STEP_EXPORT = "export"
INPUT_MODE_BATCH = "batch"
INPUT_MODE_SINGLE = "single"
TIFF_MODE_CURRENT_SLICE = "current_slice"
TIFF_MODE_MAX_PROJECTION = "max_projection"
TIFF_MODE_ALL_SLICES = "all_slices"


@dataclass
class PreviewBundle:
    pages: list[Any]
    max_projection_image: Any | None
    num_pages: int
    source_mode: str
    is_stack: bool


@dataclass
class BatchRunResult:
    input_path: str
    out_dir: str
    log_text: str
    summary_rows: list[dict[str, str]]
    stats_rows: list[dict[str, str]]
    droplet_rows: list[dict[str, str]]
    histogram_path: str | None


@dataclass
class AppSettings:
    checkpoint_path: Path = Path(DEFAULT_CHECKPOINT)
    batch_size: int = DEFAULT_BATCH_SIZE
    threshold: float = DEFAULT_THRESHOLD
    min_area: int = DEFAULT_MIN_AREA
    background_radius: int = DEFAULT_BACKGROUND_RADIUS
    resize_size: int = DEFAULT_RESIZE_SIZE
    px_per_micron: float = DEFAULT_PX_PER_MICRON
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA
    save_overlays: bool = DEFAULT_SAVE_OVERLAYS
    save_masks: bool = DEFAULT_SAVE_MASKS
    automatic_quantification: bool = DEFAULT_AUTO_QUANTIFICATION
    excel_enabled: bool = DEFAULT_EXCEL_ENABLED
    histogram_enabled: bool = DEFAULT_HISTOGRAM_ENABLED
    use_watershed_count: bool = DEFAULT_USE_WATERSHED_COUNT
    tiff_stack_mode: str = DEFAULT_TIFF_STACK_MODE
    tiff_as_png_style: bool = DEFAULT_TIFF_AS_PNG_STYLE
    debug_preprocessed_match: str = ""
    debug_preprocessed_dir: Path | None = None


@dataclass
class ViewerState:
    current_mode: str = VIEWER_MODE_ORIGINAL
    current_input_index: int = -1
    current_overlay_index: int = -1
    current_slice_index: int = 0
    available_slices: int = 1
    fit_to_window: bool = True
    source_mode: str = ""
    is_stack: bool = False
    stack_view_mode: str = STACK_VIEW_SLICE
    tiff_stack_mode: str = DEFAULT_TIFF_STACK_MODE
    workflow_step: str = WORKFLOW_STEP_LOAD


@dataclass
class SessionState:
    input_dir: Path | None
    output_dir: Path
    input_mode: str = INPUT_MODE_BATCH
    selected_image: Path | None = None
    last_out_dir: Path | None = None
    input_images: list[Path] | None = None
    overlay_paths: list[Path] | None = None
    is_running: bool = False
    last_result: BatchRunResult | None = None

    def __post_init__(self) -> None:
        if self.input_images is None:
            self.input_images = []
        if self.overlay_paths is None:
            self.overlay_paths = []


@dataclass
class AppState:
    session: SessionState
    settings: AppSettings
    viewer: ViewerState


@dataclass
class RunRequest:
    input_path: Path
    command: list[str]
    out_dir: Path
    temp_input_dir: Path | None = None

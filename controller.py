from __future__ import annotations

from pathlib import Path

from app_models import (
    AppSettings,
    AppState,
    INPUT_MODE_BATCH,
    INPUT_MODE_SINGLE,
    WORKFLOW_STEP_LOAD,
    RunRequest,
    SessionState,
    STACK_VIEW_SLICE,
    ViewerState,
    VIEWER_MODE_ORIGINAL,
    TIFF_MODE_CURRENT_SLICE,
)
from backend import prepare_run, prepare_single_image_run
from config import DEFAULT_OUT_DIR, VALID_EXTS


class SegmentAppController:
    def __init__(self) -> None:
        self.state = AppState(
            session=SessionState(
                input_dir=None,
                output_dir=Path(DEFAULT_OUT_DIR),
            ),
            settings=AppSettings(),
            viewer=ViewerState(),
        )

    def set_input_dir(self, path: str | Path | None) -> list[Path]:
        if not path:
            self.state.session.input_dir = None
            self.state.session.selected_image = None
            self.state.session.input_mode = INPUT_MODE_BATCH
            self.state.session.input_images = []
            self.reset_viewer_state()
            return []

        input_dir = Path(path)
        images = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        )
        self.state.session.input_dir = input_dir
        self.state.session.selected_image = None
        self.state.session.input_mode = INPUT_MODE_BATCH
        self.state.session.input_images = images
        self.reset_viewer_state()
        return images

    def set_input_file(self, path: str | Path | None) -> list[Path]:
        if not path:
            self.state.session.selected_image = None
            self.state.session.input_dir = None
            self.state.session.input_mode = INPUT_MODE_SINGLE
            self.state.session.input_images = []
            self.reset_viewer_state()
            return []

        image_path = Path(path)
        self.state.session.selected_image = image_path
        self.state.session.input_dir = image_path.parent
        self.state.session.input_mode = INPUT_MODE_SINGLE
        self.state.session.input_images = [image_path]
        self.reset_viewer_state()
        return [image_path]

    def set_output_dir(self, path: str | Path) -> None:
        self.state.session.output_dir = Path(path)

    def set_last_out_dir(self, path: Path | None) -> None:
        self.state.session.last_out_dir = path

    def set_overlay_paths(self, paths: list[Path]) -> None:
        self.state.session.overlay_paths = list(paths)
        self.state.viewer.current_overlay_index = -1

    def clear_results(self) -> None:
        self.state.session.overlay_paths = []
        self.state.session.last_result = None
        self.state.viewer.current_overlay_index = -1

    def reset_viewer_state(self) -> None:
        self.state.viewer.current_mode = VIEWER_MODE_ORIGINAL
        self.state.viewer.current_input_index = -1
        self.state.viewer.current_overlay_index = -1
        self.state.viewer.current_slice_index = 0
        self.state.viewer.available_slices = 1
        self.state.viewer.fit_to_window = True
        self.state.viewer.source_mode = ""
        self.state.viewer.is_stack = False
        self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
        self.state.viewer.tiff_stack_mode = self.state.settings.tiff_stack_mode or TIFF_MODE_CURRENT_SLICE
        self.state.viewer.workflow_step = WORKFLOW_STEP_LOAD

    def update_settings(
        self,
        *,
        checkpoint_path: str | Path,
        batch_size: int,
        threshold: float,
        min_area: int,
        background_radius: int,
        resize_size: int,
        px_per_micron: float,
        overlay_alpha: float,
        save_overlays: bool,
        save_masks: bool,
        automatic_quantification: bool,
        excel_enabled: bool,
        histogram_enabled: bool,
        use_watershed_count: bool,
        tiff_stack_mode: str,
        tiff_as_png_style: bool,
    ) -> None:
        settings = self.state.settings
        settings.checkpoint_path = Path(checkpoint_path)
        settings.batch_size = batch_size
        settings.threshold = threshold
        settings.min_area = min_area
        settings.background_radius = background_radius
        settings.resize_size = resize_size
        settings.px_per_micron = px_per_micron
        settings.overlay_alpha = overlay_alpha
        settings.save_overlays = save_overlays
        settings.save_masks = save_masks
        settings.automatic_quantification = automatic_quantification
        settings.excel_enabled = excel_enabled
        settings.histogram_enabled = histogram_enabled
        settings.use_watershed_count = use_watershed_count
        settings.tiff_stack_mode = tiff_stack_mode
        settings.tiff_as_png_style = tiff_as_png_style
        self.state.viewer.tiff_stack_mode = tiff_stack_mode

    def build_run_request(self) -> RunRequest:
        session = self.state.session
        settings = self.state.settings
        if session.input_mode == INPUT_MODE_SINGLE:
            image_path = session.selected_image
            if image_path is None:
                raise ValueError("Please select an image to analyze.")
            input_path, command, out_dir, temp_input_dir = prepare_single_image_run(
                image_path=image_path,
                out_dir_text=str(session.output_dir),
                settings=settings,
                current_slice_index=self.state.viewer.current_slice_index,
            )
            return RunRequest(input_path=input_path, command=command, out_dir=out_dir, temp_input_dir=temp_input_dir)

        folder_text = str(session.input_dir) if session.input_dir is not None else ""
        input_path, command, out_dir = prepare_run(
            folder_text=folder_text,
            out_dir_text=str(session.output_dir),
            settings=settings,
            current_slice_index=self.state.viewer.current_slice_index,
        )
        return RunRequest(input_path=input_path, command=command, out_dir=out_dir)

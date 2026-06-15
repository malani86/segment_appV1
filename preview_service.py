from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtGui import QPixmap

from app_models import (
    PreviewBundle,
    STACK_VIEW_PROJECTION,
    ViewerState,
)
from config import VALID_EXTS
from imaging import image_file_to_numpy, load_preview_bundle, preview_image_from_bundle


@dataclass
class ViewerDisplayData:
    original_pixmap: np.ndarray | None
    mask_pixmap: np.ndarray | None
    overlay_pixmap: np.ndarray | None
    preferred_mode: str | None = None


@dataclass
class PreviewSelection:
    input_path: Path | None
    original_pixmap: np.ndarray | None
    mask_path: Path | None
    overlay_path: Path | None
    normalized_stem: str
    matched_result_stem: str | None
    matched_slice_index: int | None
    resolved_slice_index: int
    is_stack_input: bool
    is_projection_result: bool


class PreviewResultService:
    def __init__(self) -> None:
        self._cached_preview_bundle: PreviewBundle | None = None
        self._cached_preview_path: Path | None = None

    def clear_preview_cache(self) -> None:
        self._cached_preview_bundle = None
        self._cached_preview_path = None

    def load_overlay_paths(self, out_dir: Path) -> list[Path]:
        overlay_dir = out_dir / "overlays"
        if not overlay_dir.exists():
            return []
        return sorted(p for p in overlay_dir.glob("*") if p.suffix.lower() in VALID_EXTS)

    def first_overlay_path_from_paths(self, overlay_paths: list[Path]) -> Path | None:
        if not overlay_paths:
            return None
        return overlay_paths[0]

    def prepare_display_for_input(
        self,
        *,
        viewer_state: ViewerState,
        input_images: list[Path],
        current_index: int,
        last_out_dir: Path | None,
    ) -> ViewerDisplayData | None:
        if current_index < 0 or current_index >= len(input_images):
            return None
        return self.prepare_display_for_stem(
            viewer_state=viewer_state,
            input_images=input_images,
            stem=input_images[current_index].stem,
            last_out_dir=last_out_dir,
        )

    def prepare_display_for_stem(
        self,
        *,
        viewer_state: ViewerState,
        input_images: list[Path],
        stem: str,
        last_out_dir: Path | None,
        preferred_mode: str | None = None,
    ) -> ViewerDisplayData:
        selection = self.get_preview_bundle(
            viewer_state=viewer_state,
            input_images=input_images,
            stem=stem,
            last_out_dir=last_out_dir,
        )
        mask_pixmap = image_file_to_numpy(selection.mask_path) if selection.mask_path is not None else None
        overlay_pixmap = image_file_to_numpy(selection.overlay_path, raw=True) if selection.overlay_path is not None else None

        return ViewerDisplayData(
            original_pixmap=selection.original_pixmap,
            mask_pixmap=mask_pixmap,
            overlay_pixmap=overlay_pixmap,
            preferred_mode=preferred_mode,
        )

    def get_preview_bundle(
        self,
        *,
        viewer_state: ViewerState,
        input_images: list[Path],
        stem: str,
        last_out_dir: Path | None,
    ) -> PreviewSelection:
        normalized_stem = self._normalize_result_stem(stem)
        input_index = self.find_input_index_by_stem(normalized_stem, input_images)
        input_path = input_images[input_index] if input_index is not None else None
        original_pixmap = self._load_original_pixmap_for_stem(viewer_state, input_images, normalized_stem)
        mask_path, matched_mask_stem = self.resolve_output_image_path(
            viewer_state,
            input_images,
            last_out_dir,
            "predicted_masks",
            normalized_stem,
            "_pred.png",
        )
        if mask_path is None:
            mask_path, matched_mask_stem = self.resolve_output_image_path(
                viewer_state,
                input_images,
                last_out_dir,
                "predicted_masks",
                normalized_stem,
                "_mask.png",
            )
        overlay_path, matched_overlay_stem = self.resolve_output_image_path(
            viewer_state,
            input_images,
            last_out_dir,
            "overlays",
            normalized_stem,
            "_overlay.png",
        )
        matched_result_stem = matched_overlay_stem or matched_mask_stem
        matched_slice_index = self.result_slice_index(matched_result_stem) if matched_result_stem else None
        is_projection = self.is_projection_result(matched_result_stem or "")
        return PreviewSelection(
            input_path=input_path,
            original_pixmap=original_pixmap,
            mask_path=mask_path,
            overlay_path=overlay_path,
            normalized_stem=normalized_stem,
            matched_result_stem=matched_result_stem,
            matched_slice_index=matched_slice_index,
            resolved_slice_index=viewer_state.current_slice_index,
            is_stack_input=viewer_state.is_stack,
            is_projection_result=is_projection,
        )

    def find_input_index_by_stem(self, stem: str, input_images: list[Path]) -> int | None:
        normalized_stem = self._normalize_result_stem(stem)
        for index, path in enumerate(input_images):
            if path.stem == normalized_stem:
                return index
        return None

    def resolve_output_image_path(
        self,
        viewer_state: ViewerState,
        input_images: list[Path],
        last_out_dir: Path | None,
        folder_name: str,
        stem: str,
        suffix: str,
    ) -> tuple[Path | None, str | None]:
        if last_out_dir is None:
            return None, None

        input_index = self.find_input_index_by_stem(stem, input_images)
        bundle: PreviewBundle | None = None
        is_stack = False
        if input_index is not None:
            bundle = self._get_preview_bundle(input_images[input_index])
            is_stack = bundle.is_stack

        candidates = self._result_name_candidates(
            stem=stem,
            viewer_state=viewer_state,
            is_stack=is_stack,
        )
        for candidate_stem in candidates:
            candidate = last_out_dir / folder_name / f"{candidate_stem}{suffix}"
            if candidate.exists():
                return candidate, candidate_stem
        return None, None

    def _result_name_candidates(
        self,
        *,
        stem: str,
        viewer_state: ViewerState,
        is_stack: bool,
    ) -> list[str]:
        if not is_stack:
            return [stem]

        slice_suffix = f"_z{viewer_state.current_slice_index:03d}"
        if viewer_state.stack_view_mode == STACK_VIEW_PROJECTION:
            return [f"{stem}_maxproj", stem]
        return [f"{stem}{slice_suffix}", stem]

    def _normalize_result_stem(self, stem: str) -> str:
        if stem.endswith("_maxproj"):
            return stem[: -len("_maxproj")]
        marker = "_z"
        slice_index = stem.rfind(marker)
        if slice_index >= 0:
            suffix = stem[slice_index + len(marker):]
            if suffix.isdigit():
                return stem[:slice_index]
        return stem

    def result_slice_index(self, stem: str) -> int | None:
        marker = "_z"
        slice_index = stem.rfind(marker)
        if slice_index < 0:
            return None
        suffix = stem[slice_index + len(marker):]
        if not suffix.isdigit():
            return None
        return max(0, int(suffix))

    def is_projection_result(self, stem: str) -> bool:
        return stem.endswith("_maxproj")

    def _load_original_pixmap_for_stem(
        self,
        viewer_state: ViewerState,
        input_images: list[Path],
        stem: str,
    ) -> np.ndarray | None:
        input_index = self.find_input_index_by_stem(stem, input_images)
        if input_index is None:
            viewer_state.available_slices = 1
            viewer_state.current_slice_index = 0
            viewer_state.source_mode = ""
            viewer_state.is_stack = False
            return None

        input_path = input_images[input_index]
        bundle = self._get_preview_bundle(input_path)
        viewer_state.available_slices = bundle.num_pages
        viewer_state.source_mode = bundle.source_mode
        viewer_state.is_stack = bundle.is_stack
        if not bundle.is_stack:
            viewer_state.current_slice_index = 0
        else:
            viewer_state.current_slice_index = max(0, min(bundle.num_pages - 1, viewer_state.current_slice_index))

        image = preview_image_from_bundle(
            bundle,
            slice_index=viewer_state.current_slice_index,
            stack_view_mode=viewer_state.stack_view_mode,
        )
        if image is None:
            return None
        return image

    def _get_preview_bundle(self, path: Path) -> PreviewBundle:
        if self._cached_preview_bundle is not None and self._cached_preview_path == path:
            return self._cached_preview_bundle

        bundle = load_preview_bundle(str(path))
        self._cached_preview_bundle = bundle
        self._cached_preview_path = path
        return bundle

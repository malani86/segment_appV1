from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app_models import (
    STACK_VIEW_PROJECTION,
    STACK_VIEW_SLICE,
    TIFF_MODE_ALL_SLICES,
    TIFF_MODE_CURRENT_SLICE,
    TIFF_MODE_MAX_PROJECTION,
    VIEWER_MODE_MASK,
    VIEWER_MODE_ORIGINAL,
    VIEWER_MODE_OVERLAY,
)
from app_models import AppSettings

DISPLAY_MODE_FIT = "fit"
DISPLAY_MODE_CUSTOM = "custom"


class BlobInspectorToolbar(NavigationToolbar2QT):
    def __init__(self, canvas: FigureCanvasQTAgg, parent: QWidget, home_callback):
        super().__init__(canvas, parent)
        self._home_callback = home_callback

    def home(self, *args, **kwargs):
        self._home_callback()


class ImagePanel(QWidget):
    zoomChanged = Signal(float)
    fitModeChanged = Signal(bool)
    imageClicked = Signal(float, float)

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("imagePanel")
        self._title = title
        self._image_array: np.ndarray | None = None
        self._image_width = 1
        self._image_height = 1
        self._zoom_factor = 1.0
        self._fit_to_window = True
        self._display_mode = DISPLAY_MODE_FIT
        self._background_color = "#ffffff"
        self._background_text_color = "#6b7280"
        self._panel_background_color = "#ffffff"
        self._panel_border_color = "#d8dee8"
        self._title_color = "#111827"
        self._meta_color = "#6b7280"
        self._toolbar_background_color = "#ffffff"
        self._toolbar_border_color = "#d8dee8"
        self._toolbar_text_color = "#111827"
        self._linked_panels: list[ImagePanel] = []
        self._syncing_view = False
        self._applying_view = False
        self._previous_xlim: tuple[float, float] | None = None
        self._previous_ylim: tuple[float, float] | None = None
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setToolTip("Use the Matplotlib toolbar for BlobInspector-style zoom and pan.")
        self.setStyleSheet(
            self._panel_stylesheet()
        )
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("color: #111827; font-size: 12px; font-weight: 600;")
        self._meta_label = QLabel("No image loaded")
        self._meta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._meta_label.setStyleSheet("color: #6b7280; font-size: 11px;")

        self._figure = Figure(facecolor=self._background_color)
        self._figure.subplots_adjust(0, 0, 1, 1)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        self._axes.set_axis_off()
        self._axes.set_facecolor(self._background_color)
        self._text_artist = self._axes.text(
            0.5,
            0.5,
            title,
            color=self._background_text_color,
            ha="center",
            va="center",
            transform=self._axes.transAxes,
        )
        self._image_artist = None
        self._toolbar = BlobInspectorToolbar(self._canvas, self, self.fit_to_window)
        self._toolbar.setIconSize(self._toolbar.iconSize())
        self._apply_toolbar_theme()

        header = QWidget()
        header.setObjectName("panelHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 2)
        header_layout.setSpacing(8)
        header_layout.addWidget(self._title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self._meta_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        layout.addWidget(header)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        self._canvas.mpl_connect("draw_event", self._on_draw_event)
        self._canvas.mpl_connect("button_press_event", self._on_button_press_event)
        self._apply_panel_text_styles()

    def set_linked_panels(self, panels: list["ImagePanel"]) -> None:
        self._linked_panels = panels

    def set_image_array(self, image: np.ndarray) -> None:
        self._image_array = np.asarray(image).copy()
        self._image_height, self._image_width = self._image_array.shape[:2]
        self._title_label.setText(self._title)
        self._render_image()
        if self._fit_to_window:
            self.fit_to_window()
        else:
            self._apply_zoom(self._zoom_factor)
        self._update_meta_label()

    def clear_to_title(self, title: str) -> None:
        self._title = title
        self._image_array = None
        self._zoom_factor = 1.0
        self._fit_to_window = True
        self._display_mode = DISPLAY_MODE_FIT
        self._previous_xlim = None
        self._previous_ylim = None
        self._title_label.setText(title)
        self._meta_label.setText("No image loaded")
        self._axes.clear()
        self._axes.set_axis_off()
        self._axes.set_facecolor(self._background_color)
        self._text_artist = self._axes.text(
            0.5,
            0.5,
            title,
            color=self._background_text_color,
            ha="center",
            va="center",
            transform=self._axes.transAxes,
        )
        self._image_artist = None
        self._canvas.draw_idle()
        self.zoomChanged.emit(self._zoom_factor)
        self.fitModeChanged.emit(self._fit_to_window)

    def has_image(self) -> bool:
        return self._image_array is not None and self._image_array.size > 0

    def is_fit_to_window(self) -> bool:
        return self._fit_to_window

    def fit_to_window(self) -> None:
        if not self.has_image():
            return
        self._fit_to_window = True
        self._display_mode = DISPLAY_MODE_FIT
        self._apply_display_mode_view()
        self._canvas.draw_idle()
        self._update_zoom_state()
        self._update_meta_label()
        self.fitModeChanged.emit(True)

    def reset_zoom(self) -> None:
        if not self.has_image():
            return
        self._fit_to_window = False
        self._display_mode = DISPLAY_MODE_CUSTOM
        self._apply_zoom(1.0)
        self._update_meta_label()
        self.fitModeChanged.emit(False)

    def zoom_in(self) -> None:
        self._step_zoom(1 / 1.15)

    def zoom_out(self) -> None:
        self._step_zoom(1.15)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.has_image():
            return
        if self._display_mode == DISPLAY_MODE_FIT:
            self._apply_display_mode_view()
            self._canvas.draw_idle()
            self._update_zoom_state()
            self._update_meta_label()

    def _step_zoom(self, multiplier: float) -> None:
        if not self.has_image():
            return
        self._fit_to_window = False
        self._display_mode = DISPLAY_MODE_CUSTOM
        self.fitModeChanged.emit(False)
        self._apply_zoom(self._zoom_factor * multiplier)
        self._update_meta_label()

    def _apply_zoom(self, zoom_factor: float) -> None:
        if not self.has_image():
            return
        self._zoom_factor = max(0.1, min(12.0, zoom_factor))
        self._set_centered_view_for_zoom(self._zoom_factor)
        self._canvas.draw_idle()
        self.zoomChanged.emit(self._zoom_factor)
        self._update_meta_label()

    def _render_image(self) -> None:
        self._axes.clear()
        self._axes.set_axis_off()
        self._axes.set_facecolor(self._background_color)
        if self._image_array is None:
            self._image_artist = None
            self._text_artist = self._axes.text(
                0.5,
                0.5,
                self._title,
                color=self._background_text_color,
                ha="center",
                va="center",
                transform=self._axes.transAxes,
            )
        else:
            if self._image_array.ndim == 2:
                self._image_artist = self._axes.imshow(self._image_array, cmap="gray", interpolation="nearest")
            else:
                self._image_artist = self._axes.imshow(self._image_array, interpolation="nearest")
            self._text_artist = None
        self._canvas.draw_idle()

    def _set_full_extent(self) -> None:
        self._axes.set_xlim(-0.5, self._image_width - 0.5)
        self._axes.set_ylim(self._image_height - 0.5, -0.5)

    def _apply_display_mode_view(self) -> None:
        if not self.has_image():
            return
        self._applying_view = True
        try:
            self._set_full_extent()
        finally:
            self._applying_view = False

    def _set_centered_view_for_zoom(self, zoom_factor: float) -> None:
        canvas_width = max(1, self._canvas.width())
        canvas_height = max(1, self._canvas.height())

        view_width = min(self._image_width, canvas_width / zoom_factor)
        view_height = min(self._image_height, canvas_height / zoom_factor)

        center_x = (self._image_width - 1) / 2
        center_y = (self._image_height - 1) / 2
        half_w = view_width / 2
        half_h = view_height / 2

        left = max(-0.5, center_x - half_w)
        right = min(self._image_width - 0.5, center_x + half_w)
        top = max(-0.5, center_y - half_h)
        bottom = min(self._image_height - 0.5, center_y + half_h)

        self._axes.set_xlim(left, right)
        self._axes.set_ylim(bottom, top)

    def _window_fit_zoom(self) -> float:
        return min(
            self._canvas.width() / max(1, self._image_width),
            self._canvas.height() / max(1, self._image_height),
        )

    def _display_mode_text(self) -> str:
        if self._display_mode == DISPLAY_MODE_FIT:
            return "Fit"
        return "Custom"

    def _update_meta_label(self) -> None:
        if not self.has_image():
            self._meta_label.setText("No image loaded")
            return
        self._meta_label.setText(
            f"{self._image_width}x{self._image_height} px  |  {self._display_mode_text()}  |  {self._zoom_factor * 100:.0f}%"
        )

    def _update_zoom_state(self) -> None:
        if not self.has_image():
            self._zoom_factor = 1.0
            self.zoomChanged.emit(self._zoom_factor)
            self._update_meta_label()
            return

        if self._display_mode == DISPLAY_MODE_FIT:
            self._zoom_factor = max(0.01, self._window_fit_zoom())
        else:
            x0, x1 = self._axes.get_xlim()
            current_width = max(1e-6, abs(x1 - x0))
            self._zoom_factor = max(0.1, min(12.0, self._image_width / current_width))
        self.zoomChanged.emit(self._zoom_factor)
        self._update_meta_label()

    def _on_draw_event(self, _event) -> None:
        self._update_zoom_state()
        if self._syncing_view or not self.has_image():
            self._previous_xlim = self._axes.get_xlim()
            self._previous_ylim = self._axes.get_ylim()
            return

        current_xlim = self._axes.get_xlim()
        current_ylim = self._axes.get_ylim()
        if current_xlim == self._previous_xlim and current_ylim == self._previous_ylim:
            return

        if not self._applying_view and self._display_mode != DISPLAY_MODE_CUSTOM:
            self._display_mode = DISPLAY_MODE_CUSTOM
            self._fit_to_window = False
            self.fitModeChanged.emit(False)
            self._update_meta_label()

        for panel in self._linked_panels:
            if panel is self or not panel.has_image():
                continue
            panel._sync_view_from_peer(current_xlim, current_ylim)

        self._previous_xlim = current_xlim
        self._previous_ylim = current_ylim

    def _sync_view_from_peer(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        self._syncing_view = True
        self._fit_to_window = False
        self._display_mode = DISPLAY_MODE_CUSTOM
        self._axes.set_xlim(xlim)
        self._axes.set_ylim(ylim)
        self._canvas.draw_idle()
        self._previous_xlim = xlim
        self._previous_ylim = ylim
        self._syncing_view = False
        self._update_zoom_state()

    def set_background(self, color: str, text_color: str = "#6b7280") -> None:
        self._background_color = color
        self._background_text_color = text_color
        self._figure.set_facecolor(color)
        self._axes.set_facecolor(color)
        if self._text_artist is not None:
            self._text_artist.set_color(text_color)
        self._canvas.draw_idle()

    def set_theme(
        self,
        *,
        panel_background: str,
        panel_border: str,
        title_color: str,
        meta_color: str,
    ) -> None:
        self._panel_background_color = panel_background
        self._panel_border_color = panel_border
        self._title_color = title_color
        self._meta_color = meta_color
        self.setStyleSheet(self._panel_stylesheet())
        self._apply_panel_text_styles()
        if panel_background.lower() == "#111827":
            self._toolbar_background_color = "#f8fafc"
            self._toolbar_border_color = "#cbd5e1"
            self._toolbar_text_color = "#111827"
        else:
            self._toolbar_background_color = "#ffffff"
            self._toolbar_border_color = "#d8dee8"
            self._toolbar_text_color = "#111827"
        self._apply_toolbar_theme()

    def _panel_stylesheet(self) -> str:
        return f"""
            QWidget#imagePanel {{
                border: 1px solid {self._panel_border_color};
                border-radius: 10px;
                background: {self._panel_background_color};
            }}
            """

    def _apply_panel_text_styles(self) -> None:
        self._title_label.setStyleSheet(f"color: {self._title_color}; font-size: 12px; font-weight: 600;")
        self._meta_label.setStyleSheet(f"color: {self._meta_color}; font-size: 11px;")

    def _apply_toolbar_theme(self) -> None:
        self._toolbar.setStyleSheet(
            f"""
            QToolBar {{
                border: 1px solid {self._toolbar_border_color};
                border-radius: 8px;
                spacing: 2px;
                background: {self._toolbar_background_color};
                color: {self._toolbar_text_color};
                padding: 2px 4px;
            }}
            QToolButton {{
                background: transparent;
                border: none;
                padding: 4px;
                color: {self._toolbar_text_color};
            }}
            QToolButton:hover {{
                background: rgba(148, 163, 184, 0.2);
                border-radius: 6px;
            }}
            QLabel {{
                color: {self._toolbar_text_color};
                background: transparent;
            }}
            """
        )

    def _on_button_press_event(self, event) -> None:
        if (
            event.inaxes != self._axes
            or event.xdata is None
            or event.ydata is None
            or not self.has_image()
        ):
            return
        if event.button != 1:
            return
        self.imageClicked.emit(float(event.xdata), float(event.ydata))

class InspectionViewer(QWidget):
    modeChanged = Signal(str)
    fitModeChanged = Signal(bool)
    sliceIndexChanged = Signal(int)
    stackViewModeChanged = Signal(str)
    imageClicked = Signal(str, float, float)

    def __init__(self) -> None:
        super().__init__()
        self._images: dict[str, np.ndarray | None] = {
            VIEWER_MODE_ORIGINAL: None,
            VIEWER_MODE_MASK: None,
            VIEWER_MODE_OVERLAY: None,
        }
        self._titles: dict[str, str] = {
            VIEWER_MODE_ORIGINAL: "Original image",
            VIEWER_MODE_MASK: "Predicted mask",
            VIEWER_MODE_OVERLAY: "Overlay",
        }
        self._source_mode = ""
        self._slice_summary = "Single image"
        self._slice_count = 1
        self._slice_index = 0
        self._is_stack = False
        self._stack_view_mode = STACK_VIEW_SLICE
        self._background_name = "Light"
        self._dark_mode = False
        self._build_ui()
        self._connect_signals()
        self.set_mode(VIEWER_MODE_ORIGINAL)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        toolbar = QFrame()
        toolbar.setObjectName("viewerToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(6)

        self.original_btn = QPushButton("Original")
        self.mask_btn = QPushButton("Mask")
        self.overlay_btn = QPushButton("Overlay")
        self.fit_btn = QPushButton("Fit")
        for button in (self.original_btn, self.mask_btn, self.overlay_btn):
            button.setCheckable(True)
        for button in (self.fit_btn,):
            button.setCheckable(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.original_btn)
        self.mode_group.addButton(self.mask_btn)
        self.mode_group.addButton(self.overlay_btn)

        self.info_label = QLabel("No image selected")
        self.info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.info_label.setObjectName("viewerInfo")
        self.zoom_status_label = QLabel("Fit")
        self.zoom_status_label.setAlignment(Qt.AlignCenter)
        self.zoom_status_label.setObjectName("viewerChip")
        self.background_label = QLabel("Background")
        self.background_label.setObjectName("viewerFieldLabel")
        self.background_combo = QComboBox()
        self.background_combo.setObjectName("viewerCombo")
        self.background_combo.addItems(["Light", "Slate", "Dark"])
        self.background_combo.setCurrentText(self._background_name)

        self.setStyleSheet(self._theme_stylesheet())

        for button in (
            self.original_btn,
            self.mask_btn,
            self.overlay_btn,
            self.fit_btn,
        ):
            button.setMinimumHeight(26)

        toolbar_layout.addWidget(self.original_btn)
        toolbar_layout.addWidget(self.mask_btn)
        toolbar_layout.addWidget(self.overlay_btn)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self.fit_btn)
        toolbar_layout.addWidget(self.zoom_status_label)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self.background_label)
        toolbar_layout.addWidget(self.background_combo)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.info_label)

        self.original_compare_canvas = ImagePanel("Original image")
        self.mask_compare_canvas = ImagePanel("Predicted mask")
        self.overlay_compare_canvas = ImagePanel("Overlay")
        compare_panels = [
            self.original_compare_canvas,
            self.mask_compare_canvas,
            self.overlay_compare_canvas,
        ]
        for panel in compare_panels:
            panel.set_linked_panels(compare_panels)
            panel.zoomChanged.connect(self._update_zoom_controls)
        self.original_compare_canvas.imageClicked.connect(
            lambda x, y: self.imageClicked.emit(VIEWER_MODE_ORIGINAL, x, y)
        )
        self.mask_compare_canvas.imageClicked.connect(
            lambda x, y: self.imageClicked.emit(VIEWER_MODE_MASK, x, y)
        )
        self.overlay_compare_canvas.imageClicked.connect(
            lambda x, y: self.imageClicked.emit(VIEWER_MODE_OVERLAY, x, y)
        )

        self.compare_row = QWidget()
        compare_layout = QHBoxLayout(self.compare_row)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        compare_layout.setSpacing(10)
        compare_layout.addWidget(self.original_compare_canvas, 1)
        compare_layout.addWidget(self.mask_compare_canvas, 1)
        compare_layout.addWidget(self.overlay_compare_canvas, 1)

        stack_bar = QFrame()
        stack_bar.setObjectName("stackBar")
        stack_layout = QHBoxLayout(stack_bar)
        stack_layout.setContentsMargins(10, 8, 10, 8)
        stack_layout.setSpacing(6)

        self.slice_view_btn = QPushButton("Slice")
        self.projection_view_btn = QPushButton("Projection")
        self.slice_view_btn.setCheckable(True)
        self.projection_view_btn.setCheckable(True)
        self.stack_view_group = QButtonGroup(self)
        self.stack_view_group.setExclusive(True)
        self.stack_view_group.addButton(self.slice_view_btn)
        self.stack_view_group.addButton(self.projection_view_btn)

        self.prev_slice_btn = QPushButton("Prev")
        self.next_slice_btn = QPushButton("Next")
        self.slice_label = QLabel("1/1")
        self.slice_label.setObjectName("viewerChip")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 0)

        for button in (
            self.slice_view_btn,
            self.projection_view_btn,
            self.prev_slice_btn,
            self.next_slice_btn,
        ):
            button.setMinimumHeight(26)

        stack_layout.addWidget(self.slice_view_btn)
        stack_layout.addWidget(self.projection_view_btn)
        stack_layout.addSpacing(6)
        stack_layout.addWidget(self.prev_slice_btn)
        stack_layout.addWidget(self.slice_slider, 1)
        stack_layout.addWidget(self.next_slice_btn)
        stack_layout.addWidget(self.slice_label)

        root.addWidget(toolbar)
        root.addWidget(stack_bar)
        root.addWidget(self.compare_row, 1)
        self.set_theme(False)

    def _connect_signals(self) -> None:
        self.original_btn.clicked.connect(lambda: self.set_mode(VIEWER_MODE_ORIGINAL))
        self.mask_btn.clicked.connect(lambda: self.set_mode(VIEWER_MODE_MASK))
        self.overlay_btn.clicked.connect(lambda: self.set_mode(VIEWER_MODE_OVERLAY))
        self.fit_btn.clicked.connect(self.fit_all_panels)
        self.background_combo.currentTextChanged.connect(self._apply_background_preset)
        self.prev_slice_btn.clicked.connect(self._go_to_previous_slice)
        self.next_slice_btn.clicked.connect(self._go_to_next_slice)
        self.slice_slider.valueChanged.connect(self._emit_slice_index_changed)
        self.slice_view_btn.clicked.connect(lambda: self._set_stack_view_mode(STACK_VIEW_SLICE))
        self.projection_view_btn.clicked.connect(lambda: self._set_stack_view_mode(STACK_VIEW_PROJECTION))

    def set_images(
        self,
        *,
        original: np.ndarray | None,
        mask: np.ndarray | None,
        overlay: np.ndarray | None,
        source_mode: str = "",
        is_stack: bool = False,
        available_slices: int = 1,
        current_slice_index: int = 0,
        stack_view_mode: str = STACK_VIEW_SLICE,
    ) -> None:
        self._images = {
            VIEWER_MODE_ORIGINAL: original,
            VIEWER_MODE_MASK: mask,
            VIEWER_MODE_OVERLAY: overlay,
        }
        self._source_mode = source_mode
        self._is_stack = is_stack
        self._slice_count = max(1, available_slices)
        self._slice_index = max(0, min(self._slice_count - 1, current_slice_index))
        self._stack_view_mode = stack_view_mode
        self._sync_stack_controls()
        self._slice_summary = self._format_slice_summary(available_slices, current_slice_index)
        self._update_mode_buttons()
        self._refresh_display()
        self._update_zoom_controls()

    def clear(self, title: str = "Select an image to inspect.") -> None:
        self._images = {
            VIEWER_MODE_ORIGINAL: None,
            VIEWER_MODE_MASK: None,
            VIEWER_MODE_OVERLAY: None,
        }
        self._source_mode = ""
        self._slice_summary = "Single image"
        self._is_stack = False
        self._slice_count = 1
        self._slice_index = 0
        self._stack_view_mode = STACK_VIEW_SLICE
        self.original_compare_canvas.clear_to_title("Original image")
        self.mask_compare_canvas.clear_to_title("Predicted mask")
        self.overlay_compare_canvas.clear_to_title("Overlay")
        self._sync_stack_controls()
        self._update_mode_buttons()
        self._update_info_label()
        self._update_zoom_controls()
        self.original_btn.setChecked(True)
        self.mask_btn.setChecked(False)
        self.overlay_btn.setChecked(False)
        if title:
            self.info_label.setText(title)

    def set_mode(self, mode: str) -> None:
        self.original_btn.setChecked(mode == VIEWER_MODE_ORIGINAL)
        self.mask_btn.setChecked(mode == VIEWER_MODE_MASK)
        self.overlay_btn.setChecked(mode == VIEWER_MODE_OVERLAY)
        self._refresh_display()
        self._update_mode_buttons()
        self._update_info_label(mode)
        self.modeChanged.emit(mode)

    def current_mode(self) -> str:
        if self.mask_btn.isChecked():
            return VIEWER_MODE_MASK
        if self.overlay_btn.isChecked():
            return VIEWER_MODE_OVERLAY
        return VIEWER_MODE_ORIGINAL

    def show_original(self) -> None:
        self.set_mode(VIEWER_MODE_ORIGINAL)

    def show_mask(self) -> None:
        self.set_mode(VIEWER_MODE_MASK)

    def show_overlay(self) -> None:
        self.set_mode(VIEWER_MODE_OVERLAY)

    def fit_all_panels(self) -> None:
        for panel in self._panels():
            panel.fit_to_window()
        self._update_zoom_controls()
        self.fitModeChanged.emit(True)

    def set_stack_state(
        self,
        *,
        is_stack: bool,
        available_slices: int,
        current_slice_index: int,
        stack_view_mode: str,
    ) -> None:
        self._is_stack = is_stack
        self._slice_count = max(1, available_slices)
        self._slice_index = max(0, min(self._slice_count - 1, current_slice_index))
        self._stack_view_mode = stack_view_mode
        self._slice_summary = self._format_slice_summary(self._slice_count, self._slice_index)
        self._sync_stack_controls()
        self._update_info_label()

    def _first_available_mode(self) -> str:
        for mode in (VIEWER_MODE_ORIGINAL, VIEWER_MODE_MASK, VIEWER_MODE_OVERLAY):
            if self._images.get(mode) is not None:
                return mode
        return VIEWER_MODE_ORIGINAL

    def _update_mode_buttons(self) -> None:
        self.original_btn.setEnabled(self._images[VIEWER_MODE_ORIGINAL] is not None)
        self.mask_btn.setEnabled(self._images[VIEWER_MODE_MASK] is not None)
        self.overlay_btn.setEnabled(self._images[VIEWER_MODE_OVERLAY] is not None)

    def _update_info_label(self, mode: str | None = None) -> None:
        selected_mode = mode or self.current_mode()
        details = ["Comparison view", self._titles[selected_mode].replace(" image", "")]
        if self._source_mode:
            details.append(self._source_mode)
        if self._slice_summary:
            details.append(self._slice_summary)
        self.info_label.setText(" | ".join(details))
        self._update_mode_buttons()

    def _refresh_display(self) -> None:
        self._set_panel_pixmap(
            self.original_compare_canvas,
            self._images[VIEWER_MODE_ORIGINAL],
            self._titles[VIEWER_MODE_ORIGINAL],
        )
        self._set_panel_pixmap(
            self.mask_compare_canvas,
            self._images[VIEWER_MODE_MASK],
            self._titles[VIEWER_MODE_MASK],
        )
        self._set_panel_pixmap(
            self.overlay_compare_canvas,
            self._images[VIEWER_MODE_OVERLAY],
            self._titles[VIEWER_MODE_OVERLAY],
        )
        self._apply_background_preset(self.background_combo.currentText())
        self._update_zoom_controls()

    @staticmethod
    def _set_panel_pixmap(panel: ImagePanel, pixmap: np.ndarray | None, title: str) -> None:
        if pixmap is None:
            panel.clear_to_title(f"{title} not available.")
        else:
            panel.set_image_array(pixmap)

    def _format_slice_summary(self, available_slices: int, current_slice_index: int) -> str:
        if available_slices <= 1:
            return ""
        if self._stack_view_mode == STACK_VIEW_PROJECTION:
            return f"Projection {available_slices}"
        return f"{current_slice_index + 1}/{available_slices}"

    def _sync_stack_controls(self) -> None:
        is_stack = self._is_stack and self._slice_count > 1
        self.slice_view_btn.setEnabled(is_stack)
        self.projection_view_btn.setEnabled(is_stack)
        self.prev_slice_btn.setEnabled(is_stack and self._stack_view_mode == STACK_VIEW_SLICE and self._slice_index > 0)
        self.next_slice_btn.setEnabled(
            is_stack and self._stack_view_mode == STACK_VIEW_SLICE and self._slice_index < self._slice_count - 1
        )
        self.slice_slider.setEnabled(is_stack and self._stack_view_mode == STACK_VIEW_SLICE)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max(0, self._slice_count - 1))
        self.slice_slider.setValue(self._slice_index)
        self.slice_slider.blockSignals(False)
        self.slice_view_btn.setChecked(self._stack_view_mode == STACK_VIEW_SLICE)
        self.projection_view_btn.setChecked(self._stack_view_mode == STACK_VIEW_PROJECTION)
        if is_stack:
            if self._stack_view_mode == STACK_VIEW_PROJECTION:
                self.slice_label.setText(f"P {self._slice_count}")
            else:
                self.slice_label.setText(f"{self._slice_index + 1}/{self._slice_count}")
        else:
            self.slice_label.setText("1/1")

    def _go_to_previous_slice(self) -> None:
        if self._slice_index > 0:
            self.slice_slider.setValue(self._slice_index - 1)

    def _go_to_next_slice(self) -> None:
        if self._slice_index < self._slice_count - 1:
            self.slice_slider.setValue(self._slice_index + 1)

    def _emit_slice_index_changed(self, value: int) -> None:
        self._slice_index = value
        self._slice_summary = self._format_slice_summary(self._slice_count, self._slice_index)
        self._sync_stack_controls()
        self._update_info_label()
        self.sliceIndexChanged.emit(value)

    def _set_stack_view_mode(self, mode: str) -> None:
        if self._stack_view_mode == mode:
            self._sync_stack_controls()
            return
        self._stack_view_mode = mode
        self._slice_summary = self._format_slice_summary(self._slice_count, self._slice_index)
        self._sync_stack_controls()
        self._update_info_label()
        self.stackViewModeChanged.emit(mode)

    def _panels(self) -> list[ImagePanel]:
        return [
            self.original_compare_canvas,
            self.mask_compare_canvas,
            self.overlay_compare_canvas,
        ]

    def _active_panel(self) -> ImagePanel:
        for mode, panel in (
            (VIEWER_MODE_ORIGINAL, self.original_compare_canvas),
            (VIEWER_MODE_MASK, self.mask_compare_canvas),
            (VIEWER_MODE_OVERLAY, self.overlay_compare_canvas),
        ):
            if self._images.get(mode) is not None:
                return panel
        return self.original_compare_canvas

    def _update_zoom_controls(self, *_args) -> None:
        panel = self._active_panel()
        has_image = any(item is not None for item in self._images.values())
        self.fit_btn.setEnabled(has_image)
        if not has_image:
            self.fit_btn.setChecked(False)
            self.zoom_status_label.setText("No image")
            return

        mode = panel._display_mode
        self.fit_btn.setChecked(mode == DISPLAY_MODE_FIT)
        self.zoom_status_label.setText(f"{panel._display_mode_text()}  {panel._zoom_factor * 100:.0f}%")

    def _apply_background_preset(self, preset: str) -> None:
        palette = {
            "Light": ("#ffffff", "#6b7280"),
            "Slate": ("#e2e8f0", "#475569"),
            "Dark": ("#0f172a", "#cbd5e1"),
        }
        self._background_name = preset
        background_color, text_color = palette.get(preset, palette["Light"])
        for panel in self._panels():
            panel.set_background(background_color, text_color)

    def set_theme(self, dark: bool) -> None:
        self._dark_mode = dark
        self.setStyleSheet(self._theme_stylesheet())
        if dark:
            panel_background = "#111827"
            panel_border = "#334155"
            title_color = "#e5e7eb"
            meta_color = "#94a3b8"
        else:
            panel_background = "#ffffff"
            panel_border = "#d8dee8"
            title_color = "#111827"
            meta_color = "#6b7280"
        for panel in self._panels():
            panel.set_theme(
                panel_background=panel_background,
                panel_border=panel_border,
                title_color=title_color,
                meta_color=meta_color,
            )

    def _theme_stylesheet(self) -> str:
        if self._dark_mode:
            return """
                QFrame#viewerToolbar, QFrame#stackBar {
                    background: #0f172a;
                    border: 1px solid #334155;
                    border-radius: 10px;
                }
                QPushButton {
                    min-height: 26px;
                    padding: 2px 10px;
                    border: 1px solid #475569;
                    border-radius: 7px;
                    background: #1e293b;
                    color: #e5e7eb;
                    font-weight: 500;
                }
                QPushButton:checked {
                    background: #1d4ed8;
                    border-color: #60a5fa;
                    color: #eff6ff;
                    font-weight: 600;
                }
                QPushButton:disabled {
                    color: #64748b;
                    background: #0f172a;
                    border-color: #334155;
                }
                QLabel#viewerInfo {
                    color: #cbd5e1;
                    font-size: 12px;
                    padding: 0 2px;
                }
                QLabel#viewerChip {
                    color: #e2e8f0;
                    font-size: 11px;
                    font-weight: 700;
                    padding: 4px 10px;
                    background: #1e293b;
                    border: 1px solid #475569;
                    border-radius: 999px;
                }
                QLabel#viewerFieldLabel {
                    color: #94a3b8;
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    padding-left: 2px;
                }
                QComboBox#viewerCombo {
                    min-height: 28px;
                    padding: 2px 10px;
                    border: 1px solid #475569;
                    border-radius: 7px;
                    background: #1e293b;
                    color: #e5e7eb;
                }
            """
        return """
            QFrame#viewerToolbar, QFrame#stackBar {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
            QPushButton {
                min-height: 26px;
                padding: 2px 10px;
                border: 1px solid #d1d5db;
                border-radius: 7px;
                background: #ffffff;
                color: #111827;
                font-weight: 500;
            }
            QPushButton:checked {
                background: #eef2ff;
                border-color: #64748b;
                font-weight: 600;
            }
            QPushButton:disabled {
                color: #9ca3af;
                background: #f9fafb;
                border-color: #e5e7eb;
            }
            QLabel#viewerInfo {
                color: #475569;
                font-size: 12px;
                padding: 0 2px;
            }
            QLabel#viewerChip {
                color: #334155;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                background: #ffffff;
                border: 1px solid #dbe4ee;
                border-radius: 999px;
            }
            QLabel#viewerFieldLabel {
                color: #64748b;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                padding-left: 2px;
            }
            QComboBox#viewerCombo {
                min-height: 28px;
                padding: 2px 10px;
                border: 1px solid #d1d5db;
                border-radius: 7px;
                background: #ffffff;
                color: #111827;
            }
        """


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 420)

        self.checkpoint_edit = QLineEdit(str(settings.checkpoint_path))
        self.checkpoint_browse_btn = QPushButton("Browse...")

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(settings.threshold)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10_000)
        self.batch_size_spin.setValue(settings.batch_size)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10_000_000)
        self.min_area_spin.setValue(settings.min_area)

        self.resize_size_spin = QSpinBox()
        self.resize_size_spin.setRange(32, 4096)
        self.resize_size_spin.setSingleStep(32)
        self.resize_size_spin.setValue(settings.resize_size)

        self.background_radius_spin = QSpinBox()
        self.background_radius_spin.setRange(0, 10_000)
        self.background_radius_spin.setValue(settings.background_radius)

        self.px_per_micron_spin = QDoubleSpinBox()
        self.px_per_micron_spin.setRange(0.0, 10000.0)
        self.px_per_micron_spin.setDecimals(4)
        self.px_per_micron_spin.setSingleStep(0.1)
        self.px_per_micron_spin.setValue(settings.px_per_micron)

        self.overlay_alpha_spin = QDoubleSpinBox()
        self.overlay_alpha_spin.setRange(0.0, 1.0)
        self.overlay_alpha_spin.setDecimals(2)
        self.overlay_alpha_spin.setSingleStep(0.05)
        self.overlay_alpha_spin.setValue(settings.overlay_alpha)

        self.tiff_mode_combo = QComboBox()
        self.tiff_mode_combo.addItem("Current slice", TIFF_MODE_CURRENT_SLICE)
        self.tiff_mode_combo.addItem("Max projection", TIFF_MODE_MAX_PROJECTION)
        self.tiff_mode_combo.addItem("All slices", TIFF_MODE_ALL_SLICES)
        current_tiff_index = self.tiff_mode_combo.findData(settings.tiff_stack_mode)
        if current_tiff_index >= 0:
            self.tiff_mode_combo.setCurrentIndex(current_tiff_index)

        self.save_overlays_check = QCheckBox("Save overlays")
        self.save_overlays_check.setChecked(settings.save_overlays)
        self.save_masks_check = QCheckBox("Save masks")
        self.save_masks_check.setChecked(settings.save_masks)
        self.automatic_quant_check = QCheckBox("Run quantification automatically")
        self.automatic_quant_check.setChecked(settings.automatic_quantification)
        self.watershed_count_check = QCheckBox("Use watershed for droplet counting")
        self.watershed_count_check.setChecked(settings.use_watershed_count)
        self.tiff_as_png_style_check = QCheckBox("Segment TIFFs using PNG-style input")
        self.tiff_as_png_style_check.setChecked(settings.tiff_as_png_style)
        self.excel_check = QCheckBox("Generate Excel workbook")
        self.excel_check.setChecked(settings.excel_enabled)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        checkpoint_row = QWidget()
        checkpoint_layout = QHBoxLayout(checkpoint_row)
        checkpoint_layout.setContentsMargins(0, 0, 0, 0)
        checkpoint_layout.addWidget(self.checkpoint_edit, 1)
        checkpoint_layout.addWidget(self.checkpoint_browse_btn)

        form.addRow("Checkpoint path", checkpoint_row)
        form.addRow("Batch size", self.batch_size_spin)
        form.addRow("Probability threshold", self.threshold_spin)
        form.addRow("Minimum area", self.min_area_spin)
        form.addRow("Resize size", self.resize_size_spin)
        form.addRow("Background radius", self.background_radius_spin)
        form.addRow("Pixels per micron", self.px_per_micron_spin)
        form.addRow("Overlay alpha", self.overlay_alpha_spin)
        form.addRow("TIFF stack mode", self.tiff_mode_combo)

        toggles = QWidget()
        toggles_layout = QVBoxLayout(toggles)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.addWidget(self.save_overlays_check)
        toggles_layout.addWidget(self.save_masks_check)
        toggles_layout.addWidget(self.automatic_quant_check)
        toggles_layout.addWidget(self.watershed_count_check)
        toggles_layout.addWidget(self.tiff_as_png_style_check)
        toggles_layout.addWidget(self.excel_check)
        toggles_layout.addStretch(1)
        form.addRow("Output options", toggles)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root.addLayout(form)
        root.addStretch(1)
        root.addWidget(buttons)

    def _connect_signals(self) -> None:
        self.checkpoint_browse_btn.clicked.connect(self._browse_checkpoint)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", self.checkpoint_edit.text())
        if path:
            self.checkpoint_edit.setText(path)

    def to_settings(self, current_settings: AppSettings) -> AppSettings:
        return AppSettings(
            checkpoint_path=Path(self.checkpoint_edit.text().strip()),
            batch_size=int(self.batch_size_spin.value()),
            threshold=float(self.threshold_spin.value()),
            min_area=int(self.min_area_spin.value()),
            background_radius=int(self.background_radius_spin.value()),
            resize_size=int(self.resize_size_spin.value()),
            px_per_micron=float(self.px_per_micron_spin.value()),
            overlay_alpha=float(self.overlay_alpha_spin.value()),
            save_overlays=self.save_overlays_check.isChecked(),
            save_masks=self.save_masks_check.isChecked(),
            automatic_quantification=self.automatic_quant_check.isChecked(),
            excel_enabled=self.excel_check.isChecked(),
            histogram_enabled=current_settings.histogram_enabled,
            use_watershed_count=self.watershed_count_check.isChecked(),
            tiff_stack_mode=str(self.tiff_mode_combo.currentData()),
            tiff_as_png_style=self.tiff_as_png_style_check.isChecked(),
        )

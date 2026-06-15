from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import shutil
import shlex
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import QUrl, Qt, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices
from skimage.measure import label
from PySide6.QtWidgets import (
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app_models import (
    BatchRunResult,
    INPUT_MODE_BATCH,
    INPUT_MODE_SINGLE,
    STACK_VIEW_PROJECTION,
    STACK_VIEW_SLICE,
    TIFF_MODE_ALL_SLICES,
    TIFF_MODE_CURRENT_SLICE,
    TIFF_MODE_MAX_PROJECTION,
    VIEWER_MODE_OVERLAY,
    WORKFLOW_STEP_EXPORT,
    WORKFLOW_STEP_LOAD,
    WORKFLOW_STEP_PREVIEW,
    WORKFLOW_STEP_QUANTIFY,
    WORKFLOW_STEP_REVIEW,
    WORKFLOW_STEP_SEGMENT,
)
from controller import SegmentAppController
from preview_service import PreviewResultService, ViewerDisplayData
from widgets import InspectionViewer, SettingsDialog
from workers import BatchProcessWorker


class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UNetDC Segmenter")
        self.resize(1450, 900)

        self.controller = SegmentAppController()
        self.preview_service = PreviewResultService()
        self.worker: BatchProcessWorker | None = None
        self._pending_temp_input_dir: Path | None = None
        self._droplet_rows_data: list[dict[str, str]] = []
        self._selected_droplet: dict[str, str] | None = None
        self._suppress_droplet_selection = False
        self._dark_mode = False

        self.viewer = InspectionViewer()
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_image_btn = QPushButton("Open Image")
        self.settings_btn = QPushButton("Settings")
        self.open_output_btn = QPushButton("Open Output")
        self.save_analysis_btn = QPushButton("Save Analysis")
        self.save_analysis_btn.setEnabled(False)
        self.dark_mode_btn = QPushButton("Dark Mode")
        self.dark_mode_btn.setCheckable(True)
        self.current_image_label = ClickableLabel("No image selected")
        self.current_image_label.setStyleSheet(
            """
            color: #6b7280;
            font-size: 12px;
            padding: 0 10px;
            min-height: 28px;
            border: 1px solid #d8dee8;
            border-radius: 7px;
            background: #ffffff;
            """
        )
        self.current_image_label.setCursor(Qt.PointingHandCursor)
        self.current_image_label.setMinimumWidth(220)
        self.current_image_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.summary_table = QTableWidget()
        self.stats_table = QTableWidget()
        self.droplets_table = QTableWidget()
        for table in (self.summary_table, self.stats_table, self.droplets_table):
            table.horizontalHeader().setStretchLastSection(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.summary_message = QLabel("Run the pipeline to see summary tables.")
        self.summary_message.setAlignment(Qt.AlignCenter)
        self.summary_message.setWordWrap(True)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_clear_btn = QPushButton("Clear log")
        self.delete_droplet_btn = QPushButton("Delete selected droplet")
        self.delete_droplet_btn.setEnabled(False)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.show_images_btn = QPushButton("Image View")
        self.show_summary_btn = QPushButton("Analysis Summary")
        self.show_images_btn.setCheckable(True)
        self.show_summary_btn.setCheckable(True)
        self.show_summary_btn.setEnabled(False)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setMinimumHeight(28)
        self.analyze_btn.setStyleSheet(
            """
            QPushButton {
                background: #111827;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 3px 14px;
                font-weight: 600;
            }
            QPushButton:hover:!disabled {
                background: #1f2937;
            }
            QPushButton:disabled {
                background: #9ca3af;
                color: #f9fafb;
            }
            """
        )

        self._build_ui()
        self._connect_signals()
        self._build_menu()
        self._set_workflow_step(WORKFLOW_STEP_LOAD)
        self._refresh_tiff_mode_label()
        self._apply_theme()

    def _build_ui(self) -> None:
        self.action_bar = QFrame()
        self.action_bar.setObjectName("actionBar")
        action_layout = QHBoxLayout(self.action_bar)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_layout.setSpacing(8)

        secondary_button_style = """
            QPushButton {
                min-height: 28px;
                padding: 3px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: #ffffff;
                color: #111827;
                font-weight: 500;
            }
            QPushButton:disabled {
                color: #9ca3af;
                background: #f9fafb;
                border-color: #e5e7eb;
            }
            QPushButton:hover:!disabled {
                background: #f8fafc;
            }
        """
        for button in (
            self.open_folder_btn,
            self.open_image_btn,
            self.settings_btn,
            self.open_output_btn,
            self.save_analysis_btn,
            self.dark_mode_btn,
        ):
            button.setStyleSheet(secondary_button_style)
            button.setMinimumHeight(28)

        action_layout.addWidget(self.open_folder_btn)
        action_layout.addWidget(self.open_image_btn)
        action_layout.addSpacing(4)
        action_layout.addWidget(self.current_image_label, 1)
        action_layout.addWidget(self.analyze_btn)
        action_layout.addWidget(self.open_output_btn)
        action_layout.addWidget(self.save_analysis_btn)
        action_layout.addWidget(self.dark_mode_btn)
        action_layout.addWidget(self.settings_btn)

        self.viewer_frame = QFrame()
        self.viewer_frame.setObjectName("viewerFrame")
        viewer_layout = QVBoxLayout(self.viewer_frame)
        viewer_layout.setContentsMargins(12, 12, 12, 12)
        viewer_layout.setSpacing(10)

        self.summary_page = QWidget()
        summary_page_layout = QVBoxLayout(self.summary_page)
        summary_page_layout.setContentsMargins(0, 0, 0, 0)
        summary_page_layout.setSpacing(10)
        summary_page_layout.addWidget(self.summary_message)
        summary_page_layout.addWidget(QLabel("Per-image summary"))
        summary_page_layout.addWidget(self.summary_table, 1)
        summary_page_layout.addWidget(QLabel("Size statistics"))
        summary_page_layout.addWidget(self.stats_table, 1)

        self.display_stack = QStackedWidget()
        self.display_stack.addWidget(self.viewer)
        self.display_stack.addWidget(self.summary_page)
        self.show_images_btn.setChecked(True)

        viewer_layout.addWidget(self.display_stack, 1)

        self.droplets_tab = QWidget()
        droplets_layout = QVBoxLayout(self.droplets_tab)
        droplets_hint = QLabel("Select a droplet row to jump to it in the image viewer, then delete it if it is incorrect.")
        droplets_hint.setStyleSheet("color: #64748b; font-size: 12px;")
        droplets_toolbar = QHBoxLayout()
        droplets_toolbar.addStretch(1)
        droplets_toolbar.addWidget(self.delete_droplet_btn)
        droplets_layout.addWidget(droplets_hint)
        droplets_layout.addLayout(droplets_toolbar)
        droplets_layout.addWidget(QLabel("Per-droplet table"))
        droplets_layout.addWidget(self.droplets_table)
        self.tabs.addTab(self.droplets_tab, "Droplets")

        self.log_tab = QWidget()
        log_layout = QVBoxLayout(self.log_tab)
        log_toolbar = QHBoxLayout()
        log_toolbar.addStretch(1)
        log_toolbar.addWidget(self.log_clear_btn)
        log_layout.addLayout(log_toolbar)
        log_layout.addWidget(self.log_output)
        self.tabs.addTab(self.log_tab, "Log")

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self.action_bar)
        root.addWidget(self.viewer_frame, 9)
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.clear()
        file_menu = menu_bar.addMenu("File")

        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(self.open_folder_action)

        self.open_image_action = QAction("Open Image", self)
        self.open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(self.open_image_action)

        file_menu.addSeparator()

        self.select_output_action = QAction("Select Output Folder", self)
        self.select_output_action.triggered.connect(self.browse_output_dir)
        file_menu.addAction(self.select_output_action)

        self.open_output_action = QAction("Open Output Folder", self)
        self.open_output_action.triggered.connect(self.open_output_folder)
        self.open_output_action.setEnabled(False)
        file_menu.addAction(self.open_output_action)

        self.save_analysis_action = QAction("Save Analysis As...", self)
        self.save_analysis_action.triggered.connect(self.save_analysis_as)
        self.save_analysis_action.setEnabled(False)
        file_menu.addAction(self.save_analysis_action)

        self.load_analysis_action = QAction("Load Analysis...", self)
        self.load_analysis_action.triggered.connect(self.load_analysis)
        file_menu.addAction(self.load_analysis_action)

        file_menu.addSeparator()

        self.run_action = QAction("Analyze", self)
        self.run_action.triggered.connect(self.run_pipeline)
        file_menu.addAction(self.run_action)

        file_menu.addSeparator()

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        settings_menu = menu_bar.addMenu("Settings")
        self.settings_action = QAction("Open Settings", self)
        self.settings_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(self.settings_action)
        settings_menu.addSeparator()
        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.triggered.connect(self.dark_mode_btn.setChecked)
        settings_menu.addAction(self.dark_mode_action)

        help_menu = menu_bar.addMenu("Help")

        self.about_action = QAction("About", self)
        self.about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(self.about_action)

    def _connect_signals(self) -> None:
        self.log_clear_btn.clicked.connect(self.clear_log)
        self.analyze_btn.clicked.connect(self.run_pipeline)
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.open_image_btn.clicked.connect(self.open_image)
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.save_analysis_btn.clicked.connect(self.save_analysis_as)
        self.dark_mode_btn.toggled.connect(self._toggle_dark_mode)
        self.current_image_label.clicked.connect(self._show_current_image_menu)
        self.delete_droplet_btn.clicked.connect(self.delete_selected_droplet)
        self.show_images_btn.clicked.connect(lambda: self._set_display_mode("images"))
        self.show_summary_btn.clicked.connect(lambda: self._set_display_mode("summary"))
        self.droplets_table.itemSelectionChanged.connect(self._on_droplet_table_selection_changed)
        self.viewer.imageClicked.connect(self._on_viewer_image_clicked)
        self.viewer.modeChanged.connect(self._on_viewer_mode_changed)
        self.viewer.fitModeChanged.connect(self._on_viewer_fit_mode_changed)
        self.viewer.sliceIndexChanged.connect(self._on_viewer_slice_index_changed)
        self.viewer.stackViewModeChanged.connect(self._on_viewer_stack_view_mode_changed)

    @property
    def state(self):
        return self.controller.state

    def _set_workflow_step(self, step_key: str) -> None:
        self.state.viewer.workflow_step = step_key

    @Slot(str)
    def _on_viewer_mode_changed(self, mode: str) -> None:
        self.state.viewer.current_mode = mode
        self._set_workflow_step(WORKFLOW_STEP_PREVIEW)

    @Slot(bool)
    def _on_viewer_fit_mode_changed(self, enabled: bool) -> None:
        self.state.viewer.fit_to_window = enabled

    @Slot(int)
    def _on_viewer_slice_index_changed(self, index: int) -> None:
        self.state.viewer.current_slice_index = index
        self._refresh_viewer()

    @Slot(str)
    def _on_viewer_stack_view_mode_changed(self, mode: str) -> None:
        self.state.viewer.stack_view_mode = mode
        self._refresh_viewer()

    def _sync_settings_to_state(self) -> None:
        self.controller.update_settings(
            checkpoint_path=self.state.settings.checkpoint_path,
            batch_size=int(self.state.settings.batch_size),
            threshold=float(self.state.settings.threshold),
            min_area=int(self.state.settings.min_area),
            background_radius=int(self.state.settings.background_radius),
            resize_size=int(self.state.settings.resize_size),
            px_per_micron=float(self.state.settings.px_per_micron),
            overlay_alpha=float(self.state.settings.overlay_alpha),
            save_overlays=self.state.settings.save_overlays,
            save_masks=self.state.settings.save_masks,
            automatic_quantification=self.state.settings.automatic_quantification,
            excel_enabled=self.state.settings.excel_enabled,
            histogram_enabled=self.state.settings.histogram_enabled,
            use_watershed_count=self.state.settings.use_watershed_count,
            tiff_stack_mode=self.state.settings.tiff_stack_mode,
            tiff_as_png_style=self.state.settings.tiff_as_png_style,
        )

    def _toggle_dark_mode(self, enabled: bool) -> None:
        self._dark_mode = enabled
        if hasattr(self, "dark_mode_action"):
            self.dark_mode_action.setChecked(enabled)
        self._apply_theme()

    def _apply_theme(self) -> None:
        if self._dark_mode:
            self.setStyleSheet(
                """
                QMainWindow, QWidget {
                    background: #020617;
                    color: #e2e8f0;
                }
                QTabWidget::pane {
                    border: 1px solid #334155;
                    background: #0f172a;
                }
                QTabBar::tab {
                    background: #0f172a;
                    color: #cbd5e1;
                    border: 1px solid #334155;
                    padding: 8px 14px;
                    margin-right: 4px;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                }
                QTabBar::tab:selected {
                    background: #1e293b;
                    color: #f8fafc;
                }
                QTableWidget, QPlainTextEdit {
                    background: #0f172a;
                    color: #e2e8f0;
                    border: 1px solid #334155;
                    gridline-color: #334155;
                }
                QHeaderView::section {
                    background: #1e293b;
                    color: #e2e8f0;
                    border: 1px solid #334155;
                    padding: 4px 6px;
                }
                QMenuBar, QMenu {
                    background: #0f172a;
                    color: #e2e8f0;
                }
                QMenu::item:selected {
                    background: #1d4ed8;
                }
                """
            )
            secondary_button_style = """
                QPushButton {
                    min-height: 28px;
                    padding: 3px 12px;
                    border: 1px solid #475569;
                    border-radius: 8px;
                    background: #1e293b;
                    color: #e2e8f0;
                    font-weight: 500;
                }
                QPushButton:checked {
                    background: #1d4ed8;
                    border-color: #60a5fa;
                    color: #eff6ff;
                }
                QPushButton:disabled {
                    color: #64748b;
                    background: #0f172a;
                    border-color: #334155;
                }
                QPushButton:hover:!disabled {
                    background: #334155;
                }
            """
            current_label_style = """
                color: #cbd5e1;
                font-size: 12px;
                padding: 0 10px;
                min-height: 28px;
                border: 1px solid #475569;
                border-radius: 7px;
                background: #1e293b;
            """
            analyze_style = """
                QPushButton {
                    background: #2563eb;
                    color: white;
                    border: none;
                    border-radius: 7px;
                    padding: 3px 14px;
                    font-weight: 600;
                }
                QPushButton:hover:!disabled {
                    background: #3b82f6;
                }
                QPushButton:disabled {
                    background: #334155;
                    color: #94a3b8;
                }
            """
            action_bar_style = """
                #actionBar {
                    background: #0f172a;
                    border: 1px solid #334155;
                    border-radius: 12px;
                }
            """
            viewer_frame_style = """
                #viewerFrame {
                    border: 1px solid #334155;
                    border-radius: 14px;
                    background: #111827;
                }
            """
        else:
            self.setStyleSheet("")
            secondary_button_style = """
                QPushButton {
                    min-height: 28px;
                    padding: 3px 12px;
                    border: 1px solid #d1d5db;
                    border-radius: 8px;
                    background: #ffffff;
                    color: #111827;
                    font-weight: 500;
                }
                QPushButton:disabled {
                    color: #9ca3af;
                    background: #f9fafb;
                    border-color: #e5e7eb;
                }
                QPushButton:hover:!disabled {
                    background: #f8fafc;
                }
            """
            current_label_style = """
                color: #6b7280;
                font-size: 12px;
                padding: 0 10px;
                min-height: 28px;
                border: 1px solid #d8dee8;
                border-radius: 7px;
                background: #ffffff;
            """
            analyze_style = """
                QPushButton {
                    background: #111827;
                    color: white;
                    border: none;
                    border-radius: 7px;
                    padding: 3px 14px;
                    font-weight: 600;
                }
                QPushButton:hover:!disabled {
                    background: #1f2937;
                }
                QPushButton:disabled {
                    background: #9ca3af;
                    color: #f9fafb;
                }
            """
            action_bar_style = """
                #actionBar {
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                }
            """
            viewer_frame_style = """
                #viewerFrame {
                    border: 1px solid #d7dbe2;
                    border-radius: 14px;
                    background: #ffffff;
                }
            """

        for button in (
            self.open_folder_btn,
            self.open_image_btn,
            self.settings_btn,
            self.open_output_btn,
            self.save_analysis_btn,
            self.dark_mode_btn,
            self.show_images_btn,
            self.show_summary_btn,
            self.delete_droplet_btn,
        ):
            button.setStyleSheet(secondary_button_style)
        self.current_image_label.setStyleSheet(current_label_style)
        self.analyze_btn.setStyleSheet(analyze_style)
        self.action_bar.setStyleSheet(action_bar_style)
        self.viewer_frame.setStyleSheet(viewer_frame_style)
        self.viewer.set_theme(self._dark_mode)

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.state.settings, self)
        if not dialog.exec():
            return

        updated_settings = dialog.to_settings(self.state.settings)
        self.controller.update_settings(
            checkpoint_path=updated_settings.checkpoint_path,
            batch_size=updated_settings.batch_size,
            threshold=updated_settings.threshold,
            min_area=updated_settings.min_area,
            background_radius=updated_settings.background_radius,
            resize_size=updated_settings.resize_size,
            px_per_micron=updated_settings.px_per_micron,
            overlay_alpha=updated_settings.overlay_alpha,
            save_overlays=updated_settings.save_overlays,
            save_masks=updated_settings.save_masks,
            automatic_quantification=updated_settings.automatic_quantification,
            excel_enabled=updated_settings.excel_enabled,
            histogram_enabled=updated_settings.histogram_enabled,
            use_watershed_count=updated_settings.use_watershed_count,
            tiff_stack_mode=updated_settings.tiff_stack_mode,
            tiff_as_png_style=updated_settings.tiff_as_png_style,
        )
        self._append_log_line("Settings updated.")

    def _describe_tiff_mode(self) -> str:
        mode = self.state.settings.tiff_stack_mode
        if mode == TIFF_MODE_MAX_PROJECTION:
            return "TIFF inference mode: max projection for multi-slice TIFF stacks."
        if mode == TIFF_MODE_ALL_SLICES:
            return "TIFF inference mode: all slices for multi-slice TIFF stacks."
        return f"TIFF inference mode: current slice ({self.state.viewer.current_slice_index + 1}) for multi-slice TIFF stacks."

    def _refresh_tiff_mode_label(self) -> None:
        return

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About segment_app",
            "segment_app\n\nA microscopy image analysis GUI for previewing, segmenting, and reviewing results.",
        )

    def open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not path:
            return
        self._clear_tables_and_outputs()
        self.controller.set_input_dir(path)
        self._set_workflow_step(WORKFLOW_STEP_LOAD)
        if not self.state.session.input_images:
            self._append_log_line(f"Selected folder: {path}")
            self._append_log_line("Detected images: 0")
            self.viewer.clear("Select an image to inspect.")
            self._update_input_navigation()
            return

        self._set_current_input_index(0)
        self._append_log_line(f"Selected folder: {path}")
        self._append_log_line(f"Detected images: {len(self.state.session.input_images)}")

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if not path:
            return
        self._clear_tables_and_outputs()
        self.controller.set_input_file(path)
        self._set_workflow_step(WORKFLOW_STEP_LOAD)
        self._set_current_input_index(0)
        self._append_log_line(f"Selected image: {path}")

    def browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.controller.set_output_dir(path)

    def clear_log(self) -> None:
        self.log_output.clear()

    def run_pipeline(self) -> None:
        if self.worker is not None:
            return

        self._sync_settings_to_state()
        try:
            run_request = self.controller.build_run_request()
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.log_output.clear()
        self._append_log_line("Running: " + " ".join(shlex.quote(a) for a in run_request.command))
        self._set_workflow_step(WORKFLOW_STEP_SEGMENT)
        self._set_running(True)
        self._clear_tables_and_outputs()
        self.controller.set_last_out_dir(run_request.out_dir)
        self._pending_temp_input_dir = run_request.temp_input_dir

        self.worker = BatchProcessWorker(
            input_path=run_request.input_path,
            command=run_request.command,
            out_dir=run_request.out_dir,
            parent=self,
        )
        self.worker.output.connect(self._append_log_line)
        self.worker.succeeded.connect(self.on_run_succeeded)
        self.worker.failed.connect(self.on_run_failed)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _set_running(self, running: bool) -> None:
        self.state.session.is_running = running
        widgets = (
            self.open_folder_action,
            self.open_image_action,
            self.select_output_action,
            self.settings_action,
        )
        for widget in widgets:
            widget.setEnabled(not running)
        self.run_action.setEnabled(not running)
        self.analyze_btn.setEnabled(not running)
        self.analyze_btn.setText("Analyzing..." if running else "Analyze")
        self.open_folder_btn.setEnabled(not running)
        self.open_image_btn.setEnabled(not running)
        self.settings_btn.setEnabled(not running)
        if not running:
            self.open_folder_action.setEnabled(True)
            self.open_image_action.setEnabled(True)
        self.open_output_btn.setEnabled(
            (not running)
            and self.state.session.last_out_dir is not None
            and self.state.session.last_out_dir.exists()
        )
        self.save_analysis_btn.setEnabled(self.open_output_btn.isEnabled())
        self._update_input_navigation(enabled=not running)
        self.open_output_action.setEnabled(
            (not running)
            and self.state.session.last_out_dir is not None
            and self.state.session.last_out_dir.exists()
        )
        self.save_analysis_action.setEnabled(self.open_output_action.isEnabled())

    @Slot()
    def _cleanup_worker(self) -> None:
        self._set_running(False)
        self.worker = None
        if self._pending_temp_input_dir is not None:
            shutil.rmtree(self._pending_temp_input_dir, ignore_errors=True)
            self._pending_temp_input_dir = None

    @Slot(str)
    def _append_log_line(self, line: str) -> None:
        self.log_output.appendPlainText(line)

    @Slot(object)
    def on_run_succeeded(self, result: BatchRunResult) -> None:
        self.state.session.last_result = result
        self.controller.set_last_out_dir(Path(result.out_dir))
        self._append_log_line("Finished successfully.")
        self._populate_summary(result.summary_rows, result.stats_rows)
        self._populate_droplets(result.droplet_rows)
        self._load_result_images(Path(result.out_dir))
        self._load_result_previews()
        self._set_workflow_step(WORKFLOW_STEP_QUANTIFY)

        self.open_output_action.setEnabled(True)
        self.save_analysis_action.setEnabled(True)
        self.save_analysis_btn.setEnabled(True)
        self.show_summary_btn.setEnabled(bool(result.summary_rows or result.stats_rows))
        self.tabs.setCurrentWidget(self.droplets_tab if result.droplet_rows else self.log_tab)
        self.raise_()
        self.activateWindow()
        switch_dialog = QMessageBox(self)
        switch_dialog.setWindowTitle("Processing complete")
        switch_dialog.setText("Analysis finished.")
        switch_dialog.setInformativeText("Replace the image display with the summary and size statistics?")
        show_summary_button = switch_dialog.addButton("Show Summary", QMessageBox.AcceptRole)
        keep_images_button = switch_dialog.addButton("Keep Images", QMessageBox.RejectRole)
        switch_dialog.setDefaultButton(keep_images_button)
        switch_dialog.exec()

        if switch_dialog.clickedButton() is show_summary_button and self.show_summary_btn.isEnabled():
            self._set_display_mode("summary")
        else:
            self._set_display_mode("images")

    @Slot(str)
    def on_run_failed(self, message: str) -> None:
        self.log_output.appendPlainText("ERROR: " + message)
        self._set_workflow_step(WORKFLOW_STEP_SEGMENT)
        QMessageBox.critical(self, "Error", message)

    def _clear_tables_and_outputs(self) -> None:
        self.controller.clear_results()
        self._reset_table(self.summary_table)
        self._reset_table(self.stats_table)
        self._reset_table(self.droplets_table)
        self._droplet_rows_data = []
        self._selected_droplet = None
        self.summary_message.setText("Run the pipeline to see summary tables.")
        self.controller.set_overlay_paths([])
        self.preview_service.clear_preview_cache()
        self.viewer.clear("Select an image to inspect.")
        self.state.viewer.current_input_index = -1
        self._set_workflow_step(WORKFLOW_STEP_LOAD)
        self.show_summary_btn.setEnabled(False)
        self.save_analysis_btn.setEnabled(False)
        if hasattr(self, "save_analysis_action"):
            self.save_analysis_action.setEnabled(False)
        self._set_display_mode("images")
        self._update_input_navigation()

    def _populate_summary(self, summary_rows: list[dict[str, str]], stats_rows: list[dict[str, str]]) -> None:
        if summary_rows:
            headers = list(summary_rows[0].keys())
            self._populate_table(self.summary_table, headers, summary_rows)
            self.summary_message.setText("")
        else:
            self.summary_message.setText("Summary files were not generated.")

        if stats_rows:
            headers = list(stats_rows[0].keys())
            self._populate_table(self.stats_table, headers, stats_rows)

    def _populate_droplets(self, droplet_rows: list[dict[str, str]]) -> None:
        self._droplet_rows_data = droplet_rows
        self._selected_droplet = None
        self.delete_droplet_btn.setEnabled(False)
        if not droplet_rows:
            self._reset_table(self.droplets_table)
            return
        headers = list(droplet_rows[0].keys())
        self._populate_table(self.droplets_table, headers, droplet_rows)

    def _load_result_images(self, out_dir: Path) -> None:
        overlay_paths = self.preview_service.load_overlay_paths(out_dir)
        self.controller.set_overlay_paths(overlay_paths)

    @Slot(int)
    def on_input_selected(self, index: int) -> None:
        self.state.viewer.current_input_index = index
        if index < 0 or index >= len(self.state.session.input_images):
            self.viewer.clear("Select an image to inspect.")
            self._update_input_navigation()
            return

        self.state.viewer.current_slice_index = 0
        self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
        self.preview_service.clear_preview_cache()
        self._set_display_mode("images")
        self._refresh_viewer()
        self._set_workflow_step(WORKFLOW_STEP_PREVIEW)
        self._update_input_navigation()

    def _set_current_input_index(self, index: int) -> None:
        self.on_input_selected(index)

    def _show_current_image_menu(self) -> None:
        images = self.state.session.input_images or []
        if not images:
            return
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background: #ffffff;
                border: 1px solid #d8dee8;
                padding: 6px 0;
            }
            QMenu::item {
                padding: 6px 14px;
            }
            QMenu::item:selected {
                background: #eef2ff;
            }
            """
        )
        current_index = self.state.viewer.current_input_index
        for index, image_path in enumerate(images):
            action = menu.addAction(image_path.name)
            action.setCheckable(True)
            action.setChecked(index == current_index)
            action.triggered.connect(lambda checked=False, idx=index: self._set_current_input_index(idx))

        menu.exec(self.current_image_label.mapToGlobal(self.current_image_label.rect().bottomLeft()))

    def _update_input_navigation(self, *, enabled: bool | None = None) -> None:
        can_interact = bool(self.state.session.input_images) if enabled is None else (enabled and bool(self.state.session.input_images))
        images = self.state.session.input_images or []
        current = self.state.viewer.current_input_index
        if hasattr(self, "prev_image_action"):
            self.prev_image_action.setEnabled(False)
        if hasattr(self, "next_image_action"):
            self.next_image_action.setEnabled(False)
        if 0 <= current < len(images):
            self.current_image_label.setText(images[current].name)
        elif images:
            self.current_image_label.setText(f"{len(images)} image(s)")
        else:
            self.current_image_label.setText("No image selected")

    def _load_result_previews(self) -> None:
        current_index = self.state.viewer.current_input_index
        if 0 <= current_index < len(self.state.session.input_images):
            if self.state.settings.tiff_stack_mode == TIFF_MODE_MAX_PROJECTION:
                self.state.viewer.stack_view_mode = STACK_VIEW_PROJECTION
            else:
                self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
            self._refresh_viewer()
            return

        overlay_path = self.preview_service.first_overlay_path_from_paths(self.state.session.overlay_paths)
        if overlay_path is not None:
            result_stem = overlay_path.stem.removesuffix("_overlay")
            selection = self.preview_service.get_preview_bundle(
                viewer_state=self.state.viewer,
                input_images=self.state.session.input_images,
                stem=result_stem,
                last_out_dir=self.state.session.last_out_dir,
            )
            matched_index = self.preview_service.find_input_index_by_stem(
                selection.normalized_stem,
                self.state.session.input_images,
            )
            if matched_index is not None:
                if selection.matched_slice_index is not None:
                    self.state.viewer.current_slice_index = selection.matched_slice_index
                    self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
                if selection.is_projection_result:
                    self.state.viewer.stack_view_mode = STACK_VIEW_PROJECTION
                self.state.viewer.current_mode = VIEWER_MODE_OVERLAY
                self._set_current_input_index(matched_index)
                return

            display = self.preview_service.prepare_display_for_stem(
                viewer_state=self.state.viewer,
                input_images=self.state.session.input_images,
                stem=selection.normalized_stem,
                last_out_dir=self.state.session.last_out_dir,
                preferred_mode=VIEWER_MODE_OVERLAY,
            )
            self._apply_viewer_display(display)
            return

        self.viewer.clear("Select an image to inspect.")

    def _refresh_viewer(self) -> None:
        display = self.preview_service.prepare_display_for_input(
            viewer_state=self.state.viewer,
            input_images=self.state.session.input_images,
            current_index=self.state.viewer.current_input_index,
            last_out_dir=self.state.session.last_out_dir,
        )
        if display is None:
            self.viewer.clear("Select an image to inspect.")
            return
        self._apply_viewer_display(display)

    def _apply_viewer_display(self, display: ViewerDisplayData) -> None:
        self.viewer.set_images(
            original=display.original_pixmap,
            mask=display.mask_pixmap,
            overlay=display.overlay_pixmap,
            source_mode=self.state.viewer.source_mode,
            is_stack=self.state.viewer.is_stack,
            available_slices=self.state.viewer.available_slices,
            current_slice_index=self.state.viewer.current_slice_index,
            stack_view_mode=self.state.viewer.stack_view_mode,
        )
        target_mode = display.preferred_mode or self.state.viewer.current_mode
        self.viewer.set_mode(target_mode)

    def _set_display_mode(self, mode: str) -> None:
        show_summary = mode == "summary" and self.show_summary_btn.isEnabled()
        self.display_stack.setCurrentWidget(self.summary_page if show_summary else self.viewer)
        self.show_images_btn.setChecked(not show_summary)
        self.show_summary_btn.setChecked(show_summary)

    def _reset_table(self, table: QTableWidget) -> None:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)

    def _populate_table(self, table: QTableWidget, headers: list[str], rows: list[dict[str, str]]) -> None:
        self._reset_table(table)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, header in enumerate(headers):
                item = QTableWidgetItem(str(row.get(header, "")))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()

    def open_output_folder(self) -> None:
        if self.state.session.last_out_dir is None:
            return
        self._set_workflow_step(WORKFLOW_STEP_EXPORT)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.state.session.last_out_dir)))

    def save_analysis_as(self) -> None:
        source_dir = self.state.session.last_out_dir
        if source_dir is None or not source_dir.exists():
            QMessageBox.information(self, "Save analysis", "Run an analysis before saving.")
            return

        analysis_dir = Path("analysis")
        analysis_dir.mkdir(exist_ok=True)
        default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_analysis.joblib")
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Save analysis",
            str(analysis_dir / default_name),
            "Analysis files (*.joblib)",
        )
        if not destination:
            return

        destination_path = Path(destination)
        if destination_path.suffix.lower() != ".joblib":
            destination_path = destination_path.with_suffix(".joblib")

        try:
            payload = self._analysis_payload(source_dir)
            joblib.dump(payload, destination_path, compress=True)
        except Exception as exc:
            QMessageBox.critical(self, "Save analysis", f"Could not save analysis:\n{exc}")
            return

        self._append_log_line(f"Saved analysis to: {destination_path}")
        QMessageBox.information(self, "Save analysis", f"Analysis saved to:\n{destination_path}")

    def load_analysis(self) -> None:
        analysis_dir = Path("analysis")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load analysis",
            str(analysis_dir),
            "Analysis files (*.joblib)",
        )
        if not filename:
            return

        try:
            payload = joblib.load(filename)
            out_dir = self._restore_analysis_payload(payload, Path(filename))
        except Exception as exc:
            QMessageBox.critical(self, "Load analysis", f"Could not load analysis:\n{exc}")
            return

        self.controller.set_last_out_dir(out_dir)
        self.state.session.input_images = [Path(path) for path in payload.get("input_images", []) if Path(path).exists()]
        self.state.session.input_dir = Path(payload["input_dir"]) if payload.get("input_dir") else None
        self.state.session.selected_image = Path(payload["selected_image"]) if payload.get("selected_image") else None
        self.state.session.input_mode = payload.get("input_mode", self.state.session.input_mode)
        self._apply_loaded_settings(payload.get("settings", {}))
        self._apply_loaded_viewer_state(payload.get("viewer", {}))
        if self.state.session.input_images and self.state.viewer.current_input_index < 0:
            self.state.viewer.current_input_index = 0

        result_data = payload.get("last_result") or {}
        result = BatchRunResult(
            input_path=result_data.get("input_path", ""),
            out_dir=str(out_dir),
            log_text=result_data.get("log_text", ""),
            summary_rows=payload.get("summary_rows", []),
            stats_rows=payload.get("stats_rows", []),
            droplet_rows=payload.get("droplet_rows", []),
            histogram_path=str(out_dir / "size_histogram.png") if (out_dir / "size_histogram.png").exists() else None,
        )
        self.state.session.last_result = result

        self._populate_summary(result.summary_rows, result.stats_rows)
        self._populate_droplets(result.droplet_rows)
        self._load_result_images(out_dir)
        self._load_result_previews()
        self._set_workflow_step(WORKFLOW_STEP_REVIEW)
        self.open_output_action.setEnabled(True)
        self.open_output_btn.setEnabled(True)
        self.save_analysis_action.setEnabled(True)
        self.save_analysis_btn.setEnabled(True)
        self.show_summary_btn.setEnabled(bool(result.summary_rows or result.stats_rows))
        self._append_log_line(f"Loaded analysis from: {filename}")

    def _apply_loaded_settings(self, values: object) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            if not hasattr(self.state.settings, key):
                continue
            if key in {"checkpoint_path", "debug_preprocessed_dir"}:
                setattr(self.state.settings, key, Path(value) if value else None)
            else:
                setattr(self.state.settings, key, value)

    def _apply_loaded_viewer_state(self, values: object) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            if hasattr(self.state.viewer, key):
                setattr(self.state.viewer, key, value)

    def _analysis_payload(self, source_dir: Path) -> dict[str, object]:
        result = self.state.session.last_result
        return {
            "format": "segment_app_analysis",
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir_name": source_dir.name,
            "input_dir": str(self.state.session.input_dir) if self.state.session.input_dir else "",
            "selected_image": str(self.state.session.selected_image) if self.state.session.selected_image else "",
            "input_mode": self.state.session.input_mode,
            "input_images": [str(path) for path in self.state.session.input_images or []],
            "settings": self._json_safe_mapping(asdict(self.state.settings)),
            "viewer": self._json_safe_mapping(asdict(self.state.viewer)),
            "summary_rows": self._read_rows(source_dir / "summary_per_image.csv"),
            "stats_rows": self._read_rows(source_dir / "droplet_size_stats.csv"),
            "droplet_rows": self._read_rows(source_dir / "all_droplets.csv"),
            "last_result": asdict(result) if result is not None else {},
            "files": self._collect_analysis_files(source_dir),
        }

    @staticmethod
    def _json_safe_mapping(values: dict[str, object]) -> dict[str, object]:
        safe_values: dict[str, object] = {}
        for key, value in values.items():
            safe_values[key] = str(value) if isinstance(value, Path) else value
        return safe_values

    @staticmethod
    def _collect_analysis_files(source_dir: Path) -> dict[str, bytes]:
        files: dict[str, bytes] = {}
        for path in source_dir.rglob("*"):
            if path.is_file():
                files[path.relative_to(source_dir).as_posix()] = path.read_bytes()
        return files

    def _restore_analysis_payload(self, payload: dict[str, object], analysis_path: Path) -> Path:
        if payload.get("format") != "segment_app_analysis":
            raise ValueError("This is not a segment_app analysis file.")

        loaded_root = Path("quant_results") / "loaded_analyses"
        loaded_root.mkdir(parents=True, exist_ok=True)
        base_name = analysis_path.stem.removesuffix("_analysis")
        out_dir = self._unique_directory(loaded_root, base_name)
        out_dir.mkdir(parents=True, exist_ok=False)

        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("Analysis file does not contain result files.")
        for relative_path, content in files.items():
            target_path = out_dir / str(relative_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(content)
        return out_dir

    @staticmethod
    def _unique_directory(parent: Path, folder_name: str) -> Path:
        base_name = folder_name or "analysis"
        candidate = parent / base_name
        if not candidate.exists():
            return candidate

        index = 1
        while True:
            candidate = parent / f"{base_name}_{index}"
            if not candidate.exists():
                return candidate
            index += 1

    def _on_droplet_table_selection_changed(self) -> None:
        if self._suppress_droplet_selection:
            return
        selected_indexes = self.droplets_table.selectionModel().selectedRows() if self.droplets_table.selectionModel() else []
        if not selected_indexes:
            self._selected_droplet = None
            self.delete_droplet_btn.setEnabled(False)
            self._refresh_viewer()
            return

        row_index = selected_indexes[0].row()
        if row_index < 0 or row_index >= len(self._droplet_rows_data):
            return

        droplet = self._droplet_rows_data[row_index]
        self._selected_droplet = droplet
        self.delete_droplet_btn.setEnabled(True)
        self._focus_viewer_on_droplet(droplet)

    def _focus_viewer_on_droplet(self, droplet: dict[str, str]) -> None:
        filename = droplet.get("filename", "").strip()
        if not filename:
            return

        normalized_stem, slice_index, is_projection = self._parse_droplet_filename(filename)
        matched_index = self.preview_service.find_input_index_by_stem(
            normalized_stem,
            self.state.session.input_images,
        )
        if matched_index is None:
            return

        self.state.viewer.current_input_index = matched_index
        if is_projection:
            self.state.viewer.stack_view_mode = STACK_VIEW_PROJECTION
        elif slice_index is not None:
            self.state.viewer.current_slice_index = slice_index
            self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
        else:
            self.state.viewer.stack_view_mode = STACK_VIEW_SLICE
        self.state.viewer.current_mode = VIEWER_MODE_OVERLAY
        self._set_display_mode("images")
        self._refresh_viewer()
        self.viewer.set_mode(VIEWER_MODE_OVERLAY)
        self._set_workflow_step(WORKFLOW_STEP_REVIEW)
        self._update_input_navigation()

    def _on_viewer_image_clicked(self, _mode: str, x: float, y: float) -> None:
        nearest_row_index = self._find_nearest_droplet_row(x, y)
        if nearest_row_index is None:
            return

        self.tabs.setCurrentWidget(self.droplets_tab)
        self._suppress_droplet_selection = True
        try:
            self.droplets_table.selectRow(nearest_row_index)
            self.droplets_table.scrollToItem(self.droplets_table.item(nearest_row_index, 0))
        finally:
            self._suppress_droplet_selection = False

        if 0 <= nearest_row_index < len(self._droplet_rows_data):
            droplet = self._droplet_rows_data[nearest_row_index]
            self._selected_droplet = droplet
            self.delete_droplet_btn.setEnabled(True)
            self._focus_viewer_on_droplet(droplet)

    def delete_selected_droplet(self) -> None:
        selected_indexes = self.droplets_table.selectionModel().selectedRows() if self.droplets_table.selectionModel() else []
        if not selected_indexes:
            QMessageBox.information(self, "Delete droplet", "Select a droplet row first.")
            return

        row_index = selected_indexes[0].row()
        if row_index < 0 or row_index >= len(self._droplet_rows_data):
            return

        droplet = self._droplet_rows_data[row_index]
        if QMessageBox.question(
            self,
            "Delete droplet",
            "Remove the selected droplet from the mask, overlay, and CSV outputs?",
        ) != QMessageBox.Yes:
            return

        try:
            self._delete_droplet_from_outputs(droplet, row_index)
        except Exception as exc:
            QMessageBox.critical(self, "Delete droplet", str(exc))
            return

        self._append_log_line(f"Deleted droplet from {droplet.get('filename', '')}.")
        self._refresh_viewer()

    def _delete_droplet_from_outputs(self, droplet: dict[str, str], row_index: int) -> None:
        if self.state.session.last_out_dir is None:
            raise ValueError("No output folder is available.")

        result_stem = self._result_stem_for_droplet(droplet)
        if not result_stem:
            raise ValueError("Could not resolve the selected droplet output file.")

        out_dir = self.state.session.last_out_dir
        mask_path = out_dir / "predicted_masks" / f"{result_stem}_pred.png"
        overlay_path = out_dir / "overlays" / f"{result_stem}_overlay.png"
        droplet_csv_path = out_dir / f"{result_stem}_droplets.csv"
        all_droplets_path = out_dir / "all_droplets.csv"
        summary_path = out_dir / "summary_per_image.csv"
        stats_path = out_dir / "droplet_size_stats.csv"

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        component_mask = self._component_mask_for_droplet(mask, droplet)
        if not np.any(component_mask):
            raise ValueError("Could not find this droplet in the saved mask.")

        updated_mask = (mask > 0).astype(np.uint8)
        updated_mask[component_mask] = 0
        cv2.imwrite(str(mask_path), updated_mask * 255)

        if overlay_path.exists():
            overlay = self._build_corrected_overlay(updated_mask, result_stem, overlay_path)
            if overlay is not None:
                cv2.imwrite(str(overlay_path), overlay)

        self._remove_row_from_csv(droplet_csv_path, droplet)
        combined = self._remove_row_from_csv(all_droplets_path, droplet)
        self._update_summary_after_delete(summary_path, droplet, int(component_mask.sum()))
        self._rewrite_size_stats(stats_path, combined)
        droplet_rows = self._refresh_result_tables_from_disk(summary_path, stats_path, all_droplets_path)
        if droplet_rows:
            self._droplet_rows_data = droplet_rows
        elif 0 <= row_index < len(self._droplet_rows_data):
            del self._droplet_rows_data[row_index]
        self._selected_droplet = None
        self._suppress_droplet_selection = True
        try:
            self._populate_droplets(self._droplet_rows_data)
        finally:
            self._suppress_droplet_selection = False
        self.delete_droplet_btn.setEnabled(False)
        self.preview_service.clear_preview_cache()

    def _build_corrected_overlay(self, mask: np.ndarray, result_stem: str, overlay_path: Path) -> np.ndarray | None:
        source = self._source_image_for_result_stem(result_stem)
        if source is None:
            source = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
            if source is None:
                return None
            green_pixels = (
                (source[:, :, 1] > 180)
                & (source[:, :, 0] < 80)
                & (source[:, :, 2] < 80)
            )
            source[green_pixels] = 0
        elif source.ndim == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        elif source.ndim == 3 and source.shape[2] >= 3:
            source = cv2.cvtColor(source[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            return None

        if source.shape[:2] != mask.shape:
            source = cv2.resize(source, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        overlay = source.copy()
        overlay[mask > 0] = (0, 255, 0)
        return overlay

    def _source_image_for_result_stem(self, result_stem: str) -> np.ndarray | None:
        selection = self.preview_service.get_preview_bundle(
            viewer_state=self.state.viewer,
            input_images=self.state.session.input_images,
            stem=result_stem,
            last_out_dir=self.state.session.last_out_dir,
        )
        return selection.original_pixmap

    def _component_mask_for_droplet(self, mask: np.ndarray, droplet: dict[str, str]) -> np.ndarray:
        centroid_y = self._safe_float(droplet.get("centroid-0"))
        centroid_x = self._safe_float(droplet.get("centroid-1"))
        if centroid_y is None or centroid_x is None:
            return np.zeros(mask.shape, dtype=bool)

        binary = mask > 0
        labeled = label(binary, connectivity=1)
        if labeled.max() == 0:
            return np.zeros(mask.shape, dtype=bool)

        y = int(round(centroid_y))
        x = int(round(centroid_x))
        if 0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1] and labeled[y, x] > 0:
            return labeled == labeled[y, x]

        search_radius = 8
        y0 = max(0, y - search_radius)
        y1 = min(labeled.shape[0], y + search_radius + 1)
        x0 = max(0, x - search_radius)
        x1 = min(labeled.shape[1], x + search_radius + 1)
        window = labeled[y0:y1, x0:x1]
        labels = np.unique(window[window > 0])
        if labels.size == 0:
            return np.zeros(mask.shape, dtype=bool)

        best_label = min(
            labels,
            key=lambda value: self._component_centroid_distance(labeled, int(value), centroid_x, centroid_y),
        )
        return labeled == int(best_label)

    @staticmethod
    def _component_centroid_distance(labeled: np.ndarray, component: int, x: float, y: float) -> float:
        ys, xs = np.where(labeled == component)
        if ys.size == 0:
            return float("inf")
        return float(np.hypot(float(xs.mean()) - x, float(ys.mean()) - y))

    def _remove_row_from_csv(self, path: Path, droplet: dict[str, str]) -> pd.DataFrame | None:
        if not path.exists():
            return None
        table = pd.read_csv(path)
        if table.empty:
            return table

        row_mask = self._droplet_row_mask(table, droplet)

        if row_mask.any():
            table = table.loc[~row_mask]
            table.to_csv(path, index=False)
        return table

    def _droplet_row_mask(self, table: pd.DataFrame, droplet: dict[str, str]) -> pd.Series:
        row_mask = pd.Series(True, index=table.index)
        if "filename" in table.columns and "filename" in droplet:
            row_mask &= table["filename"].astype(str) == str(droplet["filename"])
        if "label" in table.columns and "label" in droplet:
            row_mask &= table["label"].astype(str) == str(droplet["label"])

        for column in ("centroid-0", "centroid-1"):
            if column not in table.columns or column not in droplet:
                continue
            value = self._safe_float(droplet.get(column))
            if value is None:
                row_mask &= table[column].astype(str) == str(droplet[column])
            else:
                close = np.isclose(pd.to_numeric(table[column], errors="coerce"), value, atol=1e-3)
                row_mask &= pd.Series(close, index=table.index)

        return row_mask

    def _update_summary_after_delete(self, path: Path, droplet: dict[str, str], removed_area: int) -> None:
        if not path.exists():
            return
        table = pd.read_csv(path)
        if table.empty or "filename" not in table.columns:
            return

        filename = droplet.get("filename", "")
        matches = table["filename"].astype(str) == str(filename)
        total_matches = table["filename"].astype(str).str.upper() == "TOTAL"
        affected_rows = matches | total_matches
        if not affected_rows.any():
            return

        if "droplet_count" in table.columns:
            table.loc[affected_rows, "droplet_count"] = (
                table.loc[affected_rows, "droplet_count"].astype(float) - 1
            ).clip(lower=0)
        if "total_area_px" in table.columns:
            table.loc[affected_rows, "total_area_px"] = (
                table.loc[affected_rows, "total_area_px"].astype(float) - float(removed_area)
            ).clip(lower=0)
        table.to_csv(path, index=False)

    def _rewrite_size_stats(self, path: Path, combined: pd.DataFrame | None) -> None:
        if combined is None:
            return
        size_column = "eq_diam_micron" if "eq_diam_micron" in combined.columns else "equivalent_diameter"
        if not combined.empty and size_column in combined.columns and combined[size_column].notna().any():
            stats = combined[size_column].describe()[["mean", "50%", "std"]].rename({"50%": "median"})
            stats.to_csv(path)
        else:
            pd.Series(dtype=float, name=size_column).to_csv(path)

    def _refresh_result_tables_from_disk(
        self,
        summary_path: Path,
        stats_path: Path,
        droplets_path: Path,
    ) -> list[dict[str, str]]:
        summary_rows = self._read_rows(summary_path)
        stats_rows = self._read_rows(stats_path)
        droplet_rows = self._read_rows(droplets_path)
        self._populate_summary(summary_rows, stats_rows)
        if self.state.session.last_result is not None:
            self.state.session.last_result.summary_rows = summary_rows
            self.state.session.last_result.stats_rows = stats_rows
            self.state.session.last_result.droplet_rows = droplet_rows
        return droplet_rows

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        try:
            return pd.read_csv(path).fillna("").astype(str).to_dict(orient="records")
        except pd.errors.EmptyDataError:
            return []

    def _result_stem_for_droplet(self, droplet: dict[str, str]) -> str:
        normalized_stem, slice_index, is_projection = self._parse_droplet_filename(droplet.get("filename", ""))
        if is_projection:
            return f"{normalized_stem}_maxproj"
        if slice_index is not None:
            return f"{normalized_stem}_z{slice_index:03d}"
        return normalized_stem

    def _parse_droplet_filename(self, filename: str) -> tuple[str, int | None, bool]:
        cleaned = filename.strip()
        if not cleaned:
            return "", None, False

        if " [slice " in cleaned and cleaned.endswith("]"):
            stem_part, _, suffix = cleaned.partition(" [slice ")
            slice_text = suffix[:-1].split(",", 1)[0].strip()
            if slice_text.isdigit():
                return Path(stem_part).stem, int(slice_text), False

        if cleaned.endswith(" [max projection]"):
            stem_part = cleaned[: -len(" [max projection]")]
            return Path(stem_part).stem, None, True

        return self._normalize_result_stem(Path(cleaned).stem), None, False

    @staticmethod
    def _normalize_result_stem(stem: str) -> str:
        if stem.endswith("_maxproj"):
            return stem[: -len("_maxproj")]
        marker = "_z"
        slice_index = stem.rfind(marker)
        if slice_index >= 0:
            suffix = stem[slice_index + len(marker):]
            if suffix.isdigit():
                return stem[:slice_index]
        return stem

    def _find_nearest_droplet_row(self, x: float, y: float) -> int | None:
        current_index = self.state.viewer.current_input_index
        input_images = self.state.session.input_images or []
        if current_index < 0 or current_index >= len(input_images):
            return None

        current_stem = input_images[current_index].stem
        best_index: int | None = None
        best_distance: float | None = None
        max_distance = 35.0

        for index, droplet in enumerate(self._droplet_rows_data):
            droplet_stem, droplet_slice, is_projection = self._parse_droplet_filename(droplet.get("filename", ""))
            if droplet_stem != current_stem:
                continue
            if is_projection and self.state.viewer.stack_view_mode != STACK_VIEW_PROJECTION:
                continue
            if droplet_slice is not None and droplet_slice != self.state.viewer.current_slice_index:
                continue

            centroid_y = self._safe_float(droplet.get("centroid-0"))
            centroid_x = self._safe_float(droplet.get("centroid-1"))
            if centroid_y is None or centroid_x is None:
                continue

            distance = float(np.hypot(centroid_x - x, centroid_y - y))
            if distance > max_distance:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index

    @staticmethod
    def _safe_float(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

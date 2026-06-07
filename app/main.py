from __future__ import annotations
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import TumorInferenceEngine
from PySide6.QtCore import QThread, Qt, Signal, QRectF

from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "Breast Tumor Analysis Workstation"
PREVIEW_WIDTH = 420
PREVIEW_HEIGHT = 300
RESULT_WIDTH = 360
RESULT_HEIGHT = 250
SUPPORTED_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)" #example supported formats, adjust as needed


@dataclass
class AnalysisResult:
    image_path: str
    predicted_label: str
    confidence: float
    original_pixmap: QPixmap
    segmentation_overlay: QPixmap
    gradcam_overlay: QPixmap
    summary_text: str


class AnalysisEngine:
    def __init__(self):
        self.inference_engine = TumorInferenceEngine()

    def analyze(self, image_path: str, progress_callback) -> AnalysisResult:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            raise ValueError("The selected file could not be opened as an image.")

        progress_callback(10, "Loading image...")
        progress_callback(30, "Preprocessing image...")
        progress_callback(55, "Running segmentation model...")
        progress_callback(75, "Running classification model...")
        progress_callback(90, "Generating Grad-CAM visualization...")

        predicted_label, confidence, segmentation_overlay, gradcam_overlay = (
            self.inference_engine.predict(image_path)
        )

        progress_callback(100, "Analysis completed.")

        summary_text = (
            f"Prediction: {predicted_label}\n"
            f"Confidence: {confidence * 100:.1f}%\n\n"
            "Result generated using the trained deep learning model.\n"
            "Segmentation overlay highlights the predicted suspicious region.\n"
            "Grad-CAM visualization shows the image regions that influenced the classification output."
        )

        return AnalysisResult(
            image_path=image_path,
            predicted_label=predicted_label,
            confidence=confidence,
            original_pixmap=pixmap,
            segmentation_overlay=segmentation_overlay,
            gradcam_overlay=gradcam_overlay,
            summary_text=summary_text,
        )
    #next 3 methods are placeholders to simulate the behavior of the actual analysis components.
    def _classify_placeholder(self, image_path: str) -> tuple[str, float]:
        # Deterministic placeholder result derived from the file path.
        digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16)
        malignant = value % 2 == 0
        confidence = 0.72 + ((value % 20) / 100)
        confidence = min(confidence, 0.97)
        label = "Malignant" if malignant else "Benign"
        return label, confidence

    def _create_segmentation_overlay(self, base: QPixmap) -> QPixmap:
        overlay = QPixmap(base.size())
        overlay.fill(Qt.GlobalColor.transparent)

        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = base.width()
        height = base.height()

        rect = QRectF(
            width * 0.28,
            height * 0.22,
            width * 0.36,
            height * 0.42,
        )

        fill_color = QColor(220, 40, 40, 90)
        border_color = QColor(220, 40, 40, 220)

        painter.setBrush(fill_color)
        painter.setPen(QPen(border_color, 4))
        painter.drawEllipse(rect)

        painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine))
        painter.drawText(rect.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, "Tumor Region")
        painter.end()
        return overlay

    def _create_gradcam_overlay(self, base: QPixmap) -> QPixmap:
        overlay = QPixmap(base.size())
        overlay.fill(Qt.GlobalColor.transparent)

        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = base.width()
        height = base.height()
        center_x = width * 0.48
        center_y = height * 0.45
        radius = min(width, height) * 0.24

        gradient = QRadialGradient(center_x, center_y, radius)
        gradient.setColorAt(0.0, QColor(255, 0, 0, 175))
        gradient.setColorAt(0.45, QColor(255, 140, 0, 120))
        gradient.setColorAt(0.75, QColor(255, 255, 0, 65))
        gradient.setColorAt(1.0, QColor(255, 255, 0, 0))

        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))
        painter.end()
        return overlay


class AnalysisThread(QThread):
    progress_changed = Signal(int, str)
    analysis_finished = Signal(object)
    analysis_failed = Signal(str)

    def __init__(self, image_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.engine = AnalysisEngine()

    def run(self) -> None:
        try:
            result = self.engine.analyze(self.image_path, self._emit_progress)
            self.analysis_finished.emit(result)
        except Exception as exc:  
            self.analysis_failed.emit(str(exc))

    def _emit_progress(self, value: int, message: str) -> None:
        self.progress_changed.emit(value, message)


class ImagePanel(QWidget):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = title
        self._pixmap: Optional[QPixmap] = None

        self.setObjectName("imagePanel")
        self.setMinimumSize(RESULT_WIDTH, RESULT_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("imagePanelTitle")

        self.image_label = QLabel("Image preview will appear here.")
        self.image_label.setObjectName("imageCanvas")
        self.image_label.setMinimumSize(RESULT_WIDTH, RESULT_HEIGHT)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)

    def set_display_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._refresh()

    def clear_panel(self) -> None:
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText("Image preview will appear here.")

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI behavior
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return

        rect = self.image_label.contentsRect()
        target_width = max(1, rect.width())
        target_height = max(1, rect.height())

        scaled = self._pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setText("")
        self.image_label.setPixmap(scaled)


class UploadPage(QWidget):
    choose_clicked = Signal()
    analyze_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 34, 36, 28)
        root.setSpacing(20)

        header = QFrame()
        header.setObjectName("pageHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        title = QLabel("Breast Tumor Scan Analysis")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Upload a breast scan image to run segmentation, classification, and Grad-CAM visualization."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("pageSubtitle")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        content_card = QFrame()
        content_card.setObjectName("card")
        content_layout = QHBoxLayout(content_card)
        content_layout.setContentsMargins(22, 22, 22, 22)
        content_layout.setSpacing(24)

        self.preview = ImagePanel("Selected Scan")
        self.preview.setMinimumSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        content_layout.addWidget(self.preview, stretch=3)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(20, 18, 20, 18)
        side_layout.setSpacing(14)

        section_title = QLabel("Scan Selection")
        section_title.setObjectName("sectionTitle")

        section_note = QLabel(
            "Choose a supported image file, then start the trained analysis pipeline."
        )
        section_note.setObjectName("bodyText")
        section_note.setWordWrap(True)

        self.path_label = QLabel("No file selected")
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("filePathLabel")

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.choose_button = QPushButton("Select Scan")
        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.setEnabled(False)
        self.choose_button.setProperty("role", "secondary")
        self.analyze_button.setProperty("role", "primary")
        self.choose_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.choose_button.clicked.connect(self.choose_clicked.emit)
        self.analyze_button.clicked.connect(self.analyze_clicked.emit)

        button_row.addWidget(self.choose_button)
        button_row.addWidget(self.analyze_button)

        notes = QGroupBox("Review Notes")
        notes_layout = QVBoxLayout(notes)
        notes_layout.setSpacing(8)
        for note in (
            "Supported formats include PNG, JPG, BMP, and TIFF.",
            "Outputs are generated by the trained deep learning model.",
            "This prototype supports review and does not replace clinical diagnosis.",
        ):
            note_label = QLabel(note)
            note_label.setObjectName("guidanceText")
            note_label.setWordWrap(True)
            notes_layout.addWidget(note_label)

        side_layout.addWidget(section_title)
        side_layout.addWidget(section_note)
        side_layout.addWidget(self.path_label)
        side_layout.addLayout(button_row)
        side_layout.addWidget(notes)
        side_layout.addStretch(1)

        content_layout.addWidget(side_panel, stretch=2)

        footer_note = QLabel("AI-assisted decision support prototype")
        footer_note.setObjectName("footerNote")
        footer_note.setAlignment(Qt.AlignmentFlag.AlignCenter)

        root.addWidget(header)
        root.addWidget(content_card, stretch=1)
        root.addWidget(footer_note)

    def set_selected_image(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.preview.clear_panel()
            self.path_label.setText("No file selected")
            self.analyze_button.setEnabled(False)
            return

        self.preview.set_display_pixmap(pixmap)
        self.path_label.setText(f"Selected file:\n{image_path}")
        self.analyze_button.setEnabled(True)


class ProcessingPage(QWidget):
    cancel_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 34, 36, 28)
        root.setSpacing(18)
        root.addStretch(1)

        processing_card = QFrame()
        processing_card.setObjectName("processingCard")
        processing_card.setMinimumWidth(560)
        processing_card.setMaximumWidth(680)
        card_layout = QVBoxLayout(processing_card)
        card_layout.setContentsMargins(34, 32, 34, 32)
        card_layout.setSpacing(18)

        title = QLabel("Analysis in Progress")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Preparing analysis...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(26)

        hint = QLabel(
            "Segmentation, classification, and visualization are being generated."
        )
        hint.setObjectName("bodyText")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.back_button = QPushButton("Back to Upload")
        self.back_button.setProperty("role", "secondary")
        self.back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_button.clicked.connect(self.cancel_clicked.emit)
        self.back_button.setEnabled(False)

        card_layout.addWidget(title)
        card_layout.addWidget(self.status_label)
        card_layout.addWidget(self.progress_bar)
        card_layout.addWidget(hint)
        card_layout.addWidget(self.back_button, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addWidget(processing_card, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addStretch(1)

    def update_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def prepare(self) -> None:
        self.progress_bar.setValue(0)
        self.status_label.setText("Preparing analysis...")
        self.back_button.setEnabled(False)

    def allow_back(self) -> None:
        self.back_button.setEnabled(True)


class ResultsPage(QWidget):
    back_clicked = Signal()
    rerun_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 28, 30, 26)
        root.setSpacing(18)

        header = QFrame()
        header.setObjectName("pageHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(18)

        title_column = QVBoxLayout()
        title_column.setSpacing(8)

        title = QLabel("Analysis Results")
        title.setObjectName("pageTitle")

        description = QLabel(
            "Review the scan, segmentation overlay, classification output, and Grad-CAM visualization side by side."
        )
        description.setWordWrap(True)
        description.setObjectName("pageSubtitle")

        title_column.addWidget(title)
        title_column.addWidget(description)

        header_note = QLabel("AI-assisted decision support prototype")
        header_note.setObjectName("footerNote")
        header_note.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        header_layout.addLayout(title_column, stretch=1)
        header_layout.addWidget(header_note)

        self.original_panel = ImagePanel("Original Scan")
        self.segmentation_panel = ImagePanel("Segmentation Overlay")
        self.gradcam_panel = ImagePanel("Grad-CAM Overlay")

        self.summary_box = QGroupBox("Classification Summary")
        self.summary_box.setObjectName("summaryBox")
        summary_layout = QVBoxLayout(self.summary_box)
        summary_layout.setContentsMargins(18, 20, 18, 18)
        summary_layout.setSpacing(12)

        summary_header = QHBoxLayout()
        summary_header.setSpacing(10)
        prediction_label = QLabel("Model Prediction")
        prediction_label.setObjectName("eyebrowLabel")
        self.prediction_badge = QLabel("Awaiting Result")
        self.prediction_badge.setObjectName("predictionBadge")
        self.prediction_badge.setProperty("tone", "neutral")
        self.prediction_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_header.addWidget(prediction_label)
        summary_header.addStretch(1)
        summary_header.addWidget(self.prediction_badge)

        confidence_label = QLabel("Confidence")
        confidence_label.setObjectName("eyebrowLabel")
        self.confidence_value = QLabel("--")
        self.confidence_value.setObjectName("confidenceValue")

        self.summary_label = QLabel("No result available yet.")
        self.summary_label.setObjectName("summaryText")
        self.summary_label.setWordWrap(True)

        disclaimer = QLabel(
            "Final interpretation should be reviewed by a qualified clinician."
        )
        disclaimer.setObjectName("disclaimerText")
        disclaimer.setWordWrap(True)

        summary_layout.addLayout(summary_header)
        summary_layout.addWidget(confidence_label)
        summary_layout.addWidget(self.confidence_value)
        summary_layout.addWidget(self.summary_label, stretch=1)
        summary_layout.addWidget(disclaimer)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.addWidget(self.original_panel, 0, 0)
        grid.addWidget(self.segmentation_panel, 0, 1)
        grid.addWidget(self.gradcam_panel, 1, 0)
        grid.addWidget(self.summary_box, 1, 1)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.back_button = QPushButton("Analyze Another Scan")
        self.rerun_button = QPushButton("Re-run Current Scan")
        self.back_button.setProperty("role", "secondary")
        self.rerun_button.setProperty("role", "primary")
        self.back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.rerun_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_button.clicked.connect(self.back_clicked.emit)
        self.rerun_button.clicked.connect(self.rerun_clicked.emit)
        button_row.addStretch(1)
        button_row.addWidget(self.back_button)
        button_row.addWidget(self.rerun_button)

        root.addWidget(header)
        root.addLayout(grid, stretch=1)
        root.addLayout(button_row)

    def set_result(self, result: AnalysisResult) -> None:
        segmentation_view = blend_pixmaps(result.original_pixmap, result.segmentation_overlay, 1.0)
        gradcam_view = blend_pixmaps(result.original_pixmap, result.gradcam_overlay, 1.0)

        self.original_panel.set_display_pixmap(result.original_pixmap)
        self.segmentation_panel.set_display_pixmap(segmentation_view)
        self.gradcam_panel.set_display_pixmap(gradcam_view)
        self.summary_label.setText(result.summary_text)
        self.confidence_value.setText(f"{result.confidence * 100:.1f}%")
        self.prediction_badge.setText(result.predicted_label)
        self._set_prediction_tone(result.predicted_label)

    def clear_result(self) -> None:
        self.original_panel.clear_panel()
        self.segmentation_panel.clear_panel()
        self.gradcam_panel.clear_panel()
        self.summary_label.setText("No result available yet.")
        self.confidence_value.setText("--")
        self.prediction_badge.setText("Awaiting Result")
        self._set_prediction_tone("neutral")

    def _set_prediction_tone(self, label: str) -> None:
        normalized = label.strip().lower()
        if normalized == "malignant":
            tone = "malignant"
        elif normalized == "benign":
            tone = "benign"
        else:
            tone = "neutral"

        self.prediction_badge.setProperty("tone", tone)
        self.prediction_badge.style().unpolish(self.prediction_badge)
        self.prediction_badge.style().polish(self.prediction_badge)
        self.prediction_badge.update()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1120, 860)

        self.current_image_path: Optional[str] = None
        self.analysis_thread: Optional[AnalysisThread] = None
        self.latest_result: Optional[AnalysisResult] = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.upload_page = UploadPage()
        self.processing_page = ProcessingPage()
        self.results_page = ResultsPage()

        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.processing_page)
        self.stack.addWidget(self.results_page)

        self._create_menu()
        self._connect_signals()
        self._apply_styles()

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Open Scan", self)
        open_action.triggered.connect(self.select_image)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = self.menuBar().addMenu("View")

        fullscreen_action = QAction("Toggle Full Screen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self.toggle_full_screen)
        view_menu.addAction(fullscreen_action)

    def _connect_signals(self) -> None:
        self.upload_page.choose_clicked.connect(self.select_image)
        self.upload_page.analyze_clicked.connect(self.start_analysis)
        self.processing_page.cancel_clicked.connect(self.show_upload_page)
        self.results_page.back_clicked.connect(self.show_upload_page)
        self.results_page.rerun_clicked.connect(self.start_analysis)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow,
            QStackedWidget {
                background: #eef5f7;
                color: #172033;
            }

            QWidget {
                color: #172033;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }

            QFrame#pageHeader {
                background: transparent;
                border: none;
            }

            QLabel#pageTitle {
                color: #102a43;
                font-size: 28px;
                font-weight: 800;
            }

            QLabel#pageSubtitle {
                color: #526a76;
                font-size: 14px;
                line-height: 145%;
            }

            QLabel#sectionTitle {
                color: #102a43;
                font-size: 18px;
                font-weight: 750;
            }

            QLabel#bodyText,
            QLabel#guidanceText,
            QLabel#summaryText {
                color: #405763;
                line-height: 145%;
            }

            QLabel#footerNote {
                color: #6a7f89;
                font-size: 12px;
                font-weight: 600;
            }

            QLabel#filePathLabel {
                background: #f8fbfc;
                border: 1px solid #d7e3e8;
                border-radius: 8px;
                color: #314a56;
                padding: 12px;
            }

            QFrame#card,
            QFrame#processingCard {
                background: #ffffff;
                border: 1px solid #d7e3e8;
                border-radius: 14px;
            }

            QFrame#sidePanel {
                background: #f8fbfc;
                border: 1px solid #dce8ed;
                border-radius: 12px;
            }

            QWidget#imagePanel {
                background: #ffffff;
                border: 1px solid #d7e3e8;
                border-radius: 12px;
            }

            QLabel#imagePanelTitle {
                color: #102a43;
                font-size: 14px;
                font-weight: 750;
            }

            QLabel#imageCanvas {
                background: #f8fbfc;
                border: 1px solid #d7e3e8;
                border-radius: 8px;
                color: #8a99a3;
                padding: 10px;
            }

            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7e3e8;
                border-radius: 10px;
                color: #102a43;
                font-weight: 750;
                margin-top: 14px;
                padding-top: 12px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }

            QGroupBox#summaryBox {
                background: #ffffff;
            }

            QLabel#eyebrowLabel {
                color: #6a7f89;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 0px;
                text-transform: uppercase;
            }

            QLabel#confidenceValue {
                color: #0f766e;
                font-size: 34px;
                font-weight: 850;
            }

            QLabel#predictionBadge {
                border-radius: 12px;
                font-size: 12px;
                font-weight: 800;
                min-width: 104px;
                padding: 6px 12px;
            }

            QLabel#predictionBadge[tone="neutral"] {
                background: #eef2f7;
                border: 1px solid #d6e0e7;
                color: #526a76;
            }

            QLabel#predictionBadge[tone="benign"] {
                background: #ecfdf5;
                border: 1px solid #a7f3d0;
                color: #047857;
            }

            QLabel#predictionBadge[tone="malignant"] {
                background: #fff1f2;
                border: 1px solid #fecdd3;
                color: #be123c;
            }

            QLabel#disclaimerText {
                background: #f6fafb;
                border: 1px solid #dce8ed;
                border-radius: 8px;
                color: #526a76;
                padding: 10px;
            }

            QLabel#statusLabel {
                color: #102a43;
                font-size: 15px;
                font-weight: 650;
            }

            QPushButton {
                border-radius: 8px;
                font-weight: 750;
                min-height: 22px;
                min-width: 132px;
                padding: 10px 16px;
            }

            QPushButton[role="primary"] {
                background: #0f766e;
                border: 1px solid #0f766e;
                color: #ffffff;
            }

            QPushButton[role="primary"]:hover {
                background: #0d5f59;
                border-color: #0d5f59;
            }

            QPushButton[role="primary"]:pressed {
                background: #0b4f4a;
                border-color: #0b4f4a;
            }

            QPushButton[role="secondary"] {
                background: #ffffff;
                border: 1px solid #bfd1d9;
                color: #0f3d4a;
            }

            QPushButton[role="secondary"]:hover {
                background: #f2f8fa;
                border-color: #9eb9c5;
            }

            QPushButton[role="secondary"]:pressed {
                background: #e7f1f4;
                border-color: #8faebc;
            }

            QPushButton:disabled,
            QPushButton[role="primary"]:disabled,
            QPushButton[role="secondary"]:disabled {
                background: #dbe4e8;
                border: 1px solid #dbe4e8;
                color: #7d8d96;
            }

            QProgressBar {
                background: #e4edf1;
                border: none;
                border-radius: 13px;
                color: #102a43;
                font-weight: 800;
                text-align: center;
            }

            QProgressBar::chunk {
                background: #0f766e;
                border-radius: 13px;
            }

            QMenuBar {
                background: #ffffff;
                border-bottom: 1px solid #d7e3e8;
                color: #314a56;
                padding: 2px;
            }

            QMenuBar::item {
                background: transparent;
                border-radius: 6px;
                padding: 6px 10px;
            }

            QMenuBar::item:selected {
                background: #e7f1f4;
            }

            QMenu {
                background: #ffffff;
                border: 1px solid #d7e3e8;
                color: #314a56;
                padding: 6px;
            }

            QMenu::item {
                border-radius: 6px;
                padding: 7px 24px;
            }

            QMenu::item:selected {
                background: #e7f1f4;
            }
            """
        )

    def select_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Scan Image",
            str(Path.home()),
            SUPPORTED_FILTER,
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.upload_page.set_selected_image(file_path)
        self.stack.setCurrentWidget(self.upload_page)

    def start_analysis(self) -> None:
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please select an image before starting the analysis.")
            return

        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            QMessageBox.information(self, "Analysis Running", "Please wait until the current analysis completes.")
            return

        self.processing_page.prepare()
        self.stack.setCurrentWidget(self.processing_page)

        self.analysis_thread = AnalysisThread(self.current_image_path, self)
        self.analysis_thread.progress_changed.connect(self.processing_page.update_progress)
        self.analysis_thread.analysis_finished.connect(self._on_analysis_finished)
        self.analysis_thread.analysis_failed.connect(self._on_analysis_failed)
        self.analysis_thread.finished.connect(self.processing_page.allow_back)
        self.analysis_thread.start()

    def _on_analysis_finished(self, result: AnalysisResult) -> None:
        self.latest_result = result
        self.results_page.set_result(result)
        self.stack.setCurrentWidget(self.results_page)

    def _on_analysis_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Analysis Failed", message)
        self.show_upload_page()

    def show_upload_page(self) -> None:
        self.stack.setCurrentWidget(self.upload_page)

    def toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event) -> None:  # pragma: no cover - GUI behavior
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showMaximized()
            event.accept()
            return

        super().keyPressEvent(event)


def blend_pixmaps(base: QPixmap, overlay: QPixmap, overlay_opacity: float = 1.0) -> QPixmap:
    if base.isNull():
        return QPixmap()

    blended = QPixmap(base.size())
    blended.fill(Qt.GlobalColor.transparent)

    painter = QPainter(blended)
    painter.drawPixmap(0, 0, base)
    painter.setOpacity(overlay_opacity)
    painter.drawPixmap(0, 0, overlay)
    painter.end()
    return blended


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

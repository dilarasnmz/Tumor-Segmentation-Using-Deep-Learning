from __future__ import annotations
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Qt, Signal, QRectF
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
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


APP_TITLE = "Tumor Analysis Desktop App"
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
    """
    1. load image
    2. preprocess
    3. segmentation inference
    4. classification inference
    5. Grad-CAM generation
    6. return visualization assets
    """

    def analyze(self, image_path: str, progress_callback) -> AnalysisResult:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            raise ValueError("The selected file could not be opened as an image.")

        progress_callback(10, "Loading image...")
        time.sleep(0.3)

        progress_callback(30, "Preprocessing image...")
        time.sleep(0.5)

        progress_callback(55, "Running segmentation model...")
        segmentation_overlay = self._create_segmentation_overlay(pixmap)
        time.sleep(0.7)

        progress_callback(75, "Running classification model...")
        predicted_label, confidence = self._classify_placeholder(image_path)
        time.sleep(0.6)

        progress_callback(90, "Generating Grad-CAM visualization...")
        gradcam_overlay = self._create_gradcam_overlay(pixmap)
        time.sleep(0.5)

        progress_callback(100, "Analysis completed.")

        summary_text = (
            f"Prediction: {predicted_label}\n"
            f"Confidence: {confidence * 100:.1f}%\n\n"
            "This is currently a UI prototype using placeholder inference logic. "
            "Replace the AnalysisEngine methods with the trained segmentation, "
            "classification, and Grad-CAM modules from your project."
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


class ImagePanel(QLabel):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = title
        self._pixmap: Optional[QPixmap] = None
        self.setMinimumSize(RESULT_WIDTH, RESULT_HEIGHT)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            QLabel {
                background: #fafafa;
                border: 1px solid #d9d9d9;
                border-radius: 10px;
                color: #666;
                padding: 10px;
            }
            """
        )
        self.setText(f"{self.title}\n\nNo image loaded")

    def set_display_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._refresh()

    def clear_panel(self) -> None:
        self._pixmap = None
        self.setText(f"{self.title}\n\nNo image loaded")

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI behavior
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return

        rect = self.contentsRect()
        target_width = max(1, rect.width())
        target_height = max(1, rect.height())

        scaled = self._pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class UploadPage(QWidget):
    choose_clicked = Signal()
    analyze_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        title = QLabel("Breast Scan Upload")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QLabel(
            "Upload a scan image to start the analysis pipeline: segmentation → classification → Grad-CAM."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #555; font-size: 13px;")

        self.preview = ImagePanel("Selected Scan")
        self.preview.setMinimumSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)

        self.path_label = QLabel("No file selected")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #444; padding: 4px;")

        button_row = QHBoxLayout()
        self.choose_button = QPushButton("Choose Scan")
        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.setEnabled(False)

        self.choose_button.clicked.connect(self.choose_clicked.emit)
        self.analyze_button.clicked.connect(self.analyze_clicked.emit)

        button_row.addWidget(self.choose_button)
        button_row.addWidget(self.analyze_button)
        button_row.addStretch(1)

        tips = QGroupBox("User Guidance")
        tips_layout = QVBoxLayout(tips)
        tips_layout.addWidget(QLabel("• Use a supported image format such as PNG, JPG, BMP, or TIFF."))
        tips_layout.addWidget(QLabel("• The application is designed for fast review by medical professionals."))
        tips_layout.addWidget(QLabel("• Results shown in this starter version are placeholders until the trained models are connected."))

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(self.preview, stretch=1)
        root.addWidget(self.path_label)
        root.addLayout(button_row)
        root.addWidget(tips)

    def set_selected_image(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.preview.clear_panel()
            self.path_label.setText("No file selected")
            self.analyze_button.setEnabled(False)
            return

        self.preview.set_display_pixmap(pixmap)
        self.path_label.setText(f"Selected file: {image_path}")
        self.analyze_button.setEnabled(True)


class ProcessingPage(QWidget):
    cancel_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(18)
        root.addStretch(1)

        title = QLabel("Processing Analysis")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Preparing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #444;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(24)

        hint = QLabel(
            "The UI remains responsive while the analysis runs in a background thread."
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #666;")

        self.back_button = QPushButton("Back to Upload")
        self.back_button.clicked.connect(self.cancel_clicked.emit)
        self.back_button.setEnabled(False)

        root.addWidget(title)
        root.addWidget(self.status_label)
        root.addWidget(self.progress_bar)
        root.addWidget(hint)
        root.addWidget(self.back_button, alignment=Qt.AlignmentFlag.AlignCenter)
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
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        title = QLabel("Analysis Results")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)

        description = QLabel(
            "Review the original scan, segmentation overlay, classification output, and Grad-CAM heatmap side by side."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #555; font-size: 13px;")

        self.original_panel = ImagePanel("Original Scan")
        self.segmentation_panel = ImagePanel("Segmentation Overlay")
        self.gradcam_panel = ImagePanel("Grad-CAM Overlay")

        self.summary_box = QGroupBox("Classification Summary")
        summary_layout = QVBoxLayout(self.summary_box)
        self.summary_label = QLabel("No result available yet.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size: 14px; color: #333;")
        summary_layout.addWidget(self.summary_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)
        grid.addWidget(self.original_panel, 0, 0)
        grid.addWidget(self.segmentation_panel, 0, 1)
        grid.addWidget(self.gradcam_panel, 1, 0)
        grid.addWidget(self.summary_box, 1, 1)

        button_row = QHBoxLayout()
        self.back_button = QPushButton("Choose Another Scan")
        self.rerun_button = QPushButton("Run Again")
        self.back_button.clicked.connect(self.back_clicked.emit)
        self.rerun_button.clicked.connect(self.rerun_clicked.emit)
        button_row.addWidget(self.back_button)
        button_row.addWidget(self.rerun_button)
        button_row.addStretch(1)

        root.addWidget(title)
        root.addWidget(description)
        root.addLayout(grid)
        root.addLayout(button_row)

    def set_result(self, result: AnalysisResult) -> None:
        segmentation_view = blend_pixmaps(result.original_pixmap, result.segmentation_overlay, 1.0)
        gradcam_view = blend_pixmaps(result.original_pixmap, result.gradcam_overlay, 1.0)

        self.original_panel.set_display_pixmap(result.original_pixmap)
        self.segmentation_panel.set_display_pixmap(segmentation_view)
        self.gradcam_panel.set_display_pixmap(gradcam_view)
        self.summary_label.setText(result.summary_text)

    def clear_result(self) -> None:
        self.original_panel.clear_panel()
        self.segmentation_panel.clear_panel()
        self.gradcam_panel.clear_panel()
        self.summary_label.setText("No result available yet.")


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

    def _connect_signals(self) -> None:
        self.upload_page.choose_clicked.connect(self.select_image)
        self.upload_page.analyze_clicked.connect(self.start_analysis)
        self.processing_page.cancel_clicked.connect(self.show_upload_page)
        self.results_page.back_clicked.connect(self.show_upload_page)
        self.results_page.rerun_clicked.connect(self.start_analysis)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4f6f8;
            }
            QGroupBox {
                border: 1px solid #d7dce1;
                border-radius: 10px;
                margin-top: 12px;
                background: white;
                font-weight: 600;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 4px;
            }
            QPushButton {
                background: #165dff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 16px;
                font-weight: 600;
                min-width: 130px;
            }
            QPushButton:hover {
                background: #0f4bd0;
            }
            QPushButton:disabled {
                background: #b9c7e6;
                color: #eef2fb;
            }
            QMenuBar {
                background: white;
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
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

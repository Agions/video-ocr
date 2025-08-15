import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QLabel, QRubberBand


class VideoPlayer(QLabel):
    """
    A custom QLabel widget to display video frames and handle interactive ROI selection.
    It emits a signal whenever the user draws or modifies the selection rectangle.
    """
    # Signal emitted when the Region of Interest (ROI) is changed by the user.
    # It carries the new ROI as a QRect object.
    roi_changed = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("请打开视频文件。")

        self._pixmap: QPixmap | None = None
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.current_roi = QRect()

    def update_frame(self, frame: np.ndarray):
        """
        Updates the displayed video frame.

        Args:
            frame: The new frame to display, as a NumPy array (in BGR format from OpenCV).
        """
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            # Convert NumPy array (BGR) to QImage (RGB)
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            self._pixmap = QPixmap.fromImage(q_image)
            self.setPixmap(self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        except Exception as e:
            print(f"Error updating frame: {e}")

    def set_roi(self, roi: QRect):
        """
        Sets the ROI rectangle from an external source (e.g., ViewModel).
        """
        if self.current_roi != roi:
            self.current_roi = roi
            self.update() # Trigger a repaint

    def mousePressEvent(self, event: QMouseEvent):
        """Handles the start of a mouse drag to define the ROI."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handles mouse movement during ROI selection."""
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handles the end of a mouse drag, finalizing the ROI."""
        if event.button() == Qt.MouseButton.LeftButton and not self.origin.isNull():
            self.rubber_band.hide()

            # Get the geometry of the rubber band relative to the widget
            rect_widget = self.rubber_band.geometry()

            # Convert widget coordinates to pixmap (original image) coordinates
            if self._pixmap and not self._pixmap.isNull():
                pixmap_rect = self._map_widget_to_pixmap(rect_widget)
                self.current_roi = pixmap_rect
                self.roi_changed.emit(self.current_roi)

            self.origin = QPoint()

    def _map_widget_to_pixmap(self, widget_rect: QRect) -> QRect:
        """
        Maps a rectangle from the widget's coordinate system to the original
        pixmap's coordinate system, accounting for scaling and letterboxing.
        """
        if not self._pixmap or self._pixmap.isNull():
            return QRect()

        widget_size = self.size()
        pixmap_size = self._pixmap.size()

        # Find the scaled pixmap size and position within the widget
        scaled_pixmap = self._pixmap.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
        scaled_size = scaled_pixmap.size()

        offset_x = (widget_size.width() - scaled_size.width()) / 2
        offset_y = (widget_size.height() - scaled_size.height()) / 2

        # Remove the offset from the widget rectangle
        adjusted_rect = widget_rect.translated(-offset_x, -offset_y)

        # Scale the adjusted rectangle coordinates back to the original pixmap size
        scale_x = pixmap_size.width() / scaled_size.width()
        scale_y = pixmap_size.height() / scaled_size.height()

        pixmap_x = int(adjusted_rect.x() * scale_x)
        pixmap_y = int(adjusted_rect.y() * scale_y)
        pixmap_width = int(adjusted_rect.width() * scale_x)
        pixmap_height = int(adjusted_rect.height() * scale_y)

        return QRect(pixmap_x, pixmap_y, pixmap_width, pixmap_height)

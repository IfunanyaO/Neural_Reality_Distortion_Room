import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

app = QApplication([])

class FullscreenViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Distortion Room Viewer")
        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        # FORCE ALWAYS ON TOP + FULLSCREEN
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.showFullScreen()

    def update_frame(self, frame):
        """Display NumPy BGR frame in Qt window."""
        h, w, ch = frame.shape
        bytes_per_line = w * ch
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

# instantiate viewer
viewer = FullscreenViewer()
viewer.show()

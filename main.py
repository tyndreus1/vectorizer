import os
import sys
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from gui import LineDrawingApp  # Import your main application class

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Splash Screen
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
    else:
        # Running as a script
        base_path = os.path.dirname(os.path.abspath(__file__))

    splash_path = os.path.join(base_path, "splash.png")
    splash_pix = QPixmap(splash_path)
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()

    # Initialize and show main window
    window = LineDrawingApp()
    QTimer.singleShot(1000, splash.close)  # Close splash after 1 second
    QTimer.singleShot(1000, window.show)
    sys.exit(app.exec_())
import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QTransform, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QSlider, QFileDialog,
    QHBoxLayout, QVBoxLayout, QWidget, QGridLayout, QComboBox, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsColorizeEffect,
    QMessageBox, QGraphicsItem
)
from processing import apply_enhancements, process_with_ai_model
import subprocess
import sys  # Import the sys module

class LineDrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None             # Original image (cv2 BGR)
        self.processed_image = None   # Processed image (cv2; may be grayscale or color)
        self.pixmap_item = None       # QGraphicsPixmapItem for the processed image
        self.is_displaying = False   # Flag to prevent re-entrant calls to display_image
        self.is_updating_all = False # Flag to prevent re-entrant calls to update_all
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Line Drawing App')
        self.setGeometry(100, 100, 1400, 800)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create layouts for left and right panels
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        slider_layout = QGridLayout()

        # Left panel: original image preview and enhancement controls
        self.loaded_image_label = QLabel(self)
        self.loaded_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Changed to Fixed
        # self.loaded_image_label.setFixedSize(600, 400)  # Set a fixed size
        self.loaded_image_label.setAlignment(Qt.AlignCenter)
        self.loaded_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #000;")

        self.load_button = QPushButton('Load Image', self)
        self.save_button = QPushButton('Save Image', self)

        # Enhancement sliders
        self.brightness_label = QLabel('Brightness', self)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)

        self.contrast_label = QLabel('Contrast', self)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(50)

        self.sharpness_label = QLabel('Sharpness', self)
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 100)
        self.sharpness_slider.setValue(50)

        self.blur_label = QLabel('Blur', self)
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 20)
        self.blur_slider.setValue(0)

        self.edge_sensitivity_label = QLabel('Edge Sensitivity', self)
        self.edge_sensitivity_slider = QSlider(Qt.Horizontal)
        self.edge_sensitivity_slider.setRange(0, 100)
        self.edge_sensitivity_slider.setValue(50)

        self.threshold_label = QLabel('Threshold', self)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)

        self.line_thickness_label = QLabel('Line Thickness', self)
        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setRange(1, 10)
        self.line_thickness_slider.setValue(1)

        # Connect buttons and sliders
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(lambda: self.save_image())
        self.brightness_slider.valueChanged.connect(self.update_display_loaded_image)  # Connect to update_display_loaded_image
        self.contrast_slider.valueChanged.connect(self.update_display_loaded_image)   # Connect to update_display_loaded_image
        self.sharpness_slider.valueChanged.connect(self.update_display_loaded_image)  # Connect to update_display_loaded_image
        self.blur_slider.valueChanged.connect(self.update_display_loaded_image)       # Connect to update_display_loaded_image
        self.edge_sensitivity_slider.valueChanged.connect(self.update_display_loaded_image) # Connect to update_display_loaded_image
        self.threshold_slider.valueChanged.connect(self.update_display_loaded_image)    # Connect to update_display_loaded_image
        self.line_thickness_slider.valueChanged.connect(self.update_display_loaded_image) # Connect to update_display_loaded_image

        # Combo box for processing method
        self.processing_method_label = QLabel('Processing Method', self)
        self.processing_method_combo = QComboBox(self)
        self.processing_method_combo.addItem('Threshold')
        self.processing_method_combo.addItem('Edge Detection')
        self.processing_method_combo.currentIndexChanged.connect(self.update_all)  # Connect to update_all

        # Set size policies to Fixed
        self.processing_method_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.processing_method_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        left_layout.addWidget(self.loaded_image_label)
        left_layout.addWidget(self.load_button)
        self.load_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.load_button.setFixedSize(150, 30)
        left_layout.addWidget(self.save_button)
        self.save_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.save_button.setFixedSize(150, 30)

        slider_layout.addWidget(self.brightness_label, 0, 0)
        slider_layout.addWidget(self.brightness_slider, 0, 1)
        slider_layout.addWidget(self.contrast_label, 1, 0)
        slider_layout.addWidget(self.contrast_slider, 1, 1)
        slider_layout.addWidget(self.sharpness_label, 2, 0)
        slider_layout.addWidget(self.sharpness_slider, 2, 1)
        slider_layout.addWidget(self.blur_label, 3, 0)
        slider_layout.addWidget(self.blur_slider, 3, 1)
        slider_layout.addWidget(self.edge_sensitivity_label, 4, 0)
        slider_layout.addWidget(self.edge_sensitivity_slider, 4, 1)
        slider_layout.addWidget(self.threshold_label, 5, 0)
        slider_layout.addWidget(self.threshold_slider, 5, 1)
        slider_layout.addWidget(self.line_thickness_label, 6, 0)
        slider_layout.addWidget(self.line_thickness_slider, 6, 1)

        left_layout.addLayout(slider_layout)

        # Add additional controls to left layout
        left_layout.addWidget(self.processing_method_label)
        left_layout.addWidget(self.processing_method_combo)

        main_layout.addLayout(left_layout)

        # Right panel: processed image preview
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setStyleSheet("background-color: #ffffff; border: 1px solid #000;")

        right_layout.addWidget(self.view)
        main_layout.addLayout(right_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


    def load_image(self):
        """
        Loads an image from a file, handling potential errors.

        Displays an error message if the image cannot be loaded.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            try:
                self.image = cv2.imread(file_name)
                if self.image is None:
                    raise ValueError("Could not read image file.")
                self.display_image(self.image)
                self.update_all()  # Apply initial enhancements and processing
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {e}")

    def display_image(self, image):
        """Displays the image in the loaded_image_label, scaling it to fit."""
        if self.is_displaying:
            print("display_image: Re-entrant call prevented")
            return  # Prevent re-entrant calls

        self.is_displaying = True
        print("display_image: Starting...")
        try:
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
            pixmap = QPixmap(qImg)

            # Scale the pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.loaded_image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.loaded_image_label.setPixmap(scaled_pixmap)
            print("display_image: Image displayed successfully")
        except Exception as e:
            print(f"display_image: Error displaying image: {e}")
        finally:
            self.is_displaying = True
            print("display_image: Finished")

    def update_display_loaded_image(self):
         if self.image is not None:
            self.display_image(self.image)
            
            
    def update_loaded_image(self):
        if self.image is not None:
            if not hasattr(self, '_display_timer'):
                self._display_timer = QTimer()
                self._display_timer.setSingleShot(True)
                self._display_timer.timeout.connect(self._delayed_display)
            self._display_timer.start(50)  # 50 ms delay

    def _delayed_display(self):
        self.display_image(self.image, self.loaded_image_label, scale=0.7)        

    def resizeEvent(self, event):
        """Override resizeEvent to scale the image when the window is resized."""
        print("resizeEvent: Triggered")
        if self.image is not None:
            self.display_image(self.image)
        print("resizeEvent: Finished")


    def update_enhancements(self):
        """Applies brightness, contrast, sharpness, and blur enhancements to the image."""
        if self.image is None:
            return

        params = {
            'brightness': self.brightness_slider.value(),
            'contrast': self.contrast_slider.value(),
            'sharpness': self.sharpness_slider.value(),
            'blur': self.blur_slider.value()
        }

        self.processed_image = apply_enhancements(self.image, params)
        self.update_display()

    def update_processing(self):
        """Applies thresholding or edge detection based on the selected method."""
        if self.image is None:
            return

        params = {
            'method': self.processing_method_combo.currentText(),
            'edge_sensitivity': self.edge_sensitivity_slider.value(),
            'threshold': self.threshold_slider.value(),
            'line_thickness': self.line_thickness_slider.value()
        }

        self.processed_image = process_with_ai_model(self.image, params)
        self.update_display()

    def update_all(self):
        """Applies enhancements and processing, then displays the result."""
        if self.is_updating_all:
            print("update_all: Re-entrant call prevented")
          #  return  # Prevent re-entrant calls

        self.is_updating_all = True
        print("update_all: Starting...")

        try:
            if self.image is None:
                return

            # Apply Enhancements
            enhancement_params = {
                'brightness': self.brightness_slider.value(),
                'contrast': self.contrast_slider.value(),
                'sharpness': self.sharpness_slider.value(),
                'blur': self.blur_slider.value()
            }
            enhanced_image = apply_enhancements(self.image, enhancement_params)

            # Apply Processing
            processing_params = {
                'method': self.processing_method_combo.currentText(),
                'edge_sensitivity': self.edge_sensitivity_slider.value(),
                'threshold': self.threshold_slider.value(),
                'line_thickness': self.line_thickness_slider.value()
            }
            self.processed_image = process_with_ai_model(enhanced_image, processing_params)  # Process enhanced image

            self.update_display()

        finally:
            self.is_updating_all = False
            print("update_all: Finished")


    def update_display(self):
        """Displays the processed image in the right panel."""
        if self.processed_image is None:
            return

        height, width = self.processed_image.shape[:2]
        if self.processed_image.ndim == 2:  # Grayscale image
            bytesPerLine = width
            qImg = QImage(self.processed_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        else:  # Color image
            bytesPerLine = 3 * width
            qImg = QImage(self.processed_image.data, width, height, bytesPerLine, QImage.Format_BGR888)

        pixmap = QPixmap(qImg)
        if self.pixmap_item is None:
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
        else:
            self.pixmap_item.setPixmap(pixmap)

        self.scene.setSceneRect(QRectF(0, 0, width, height))
        self.view.fitInView(QRectF(0, 0, width, height), Qt.KeepAspectRatio)

    def save_image(self):
        """Saves the processed image to a file with format options and automatic extension."""
        print("save_image: Save button clicked!")

        if self.processed_image is None:
            print("save_image: No processed image to save.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, selected_filter = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg);;BMP Files (*.bmp);;SVG Files (*.svg)", options=options)

        if file_name:
            print(f"save_image: Saving to {file_name}")
            try:
                if selected_filter == "PNG Files (*.png)" and not file_name.lower().endswith(".png"):
                    file_name += ".png"
                elif selected_filter == "JPG Files (*.jpg)" and not file_name.lower().endswith(".jpg"):
                    file_name += ".jpg"
                elif selected_filter == "BMP Files (*.bmp)" and not file_name.lower().endswith(".bmp"):
                    file_name += ".bmp"
                elif selected_filter == "SVG Files (*.svg)" and not file_name.lower().endswith(".svg"):
                    self.convert_to_vector(file_name, self.processed_image, "svg")
                else:
                    cv2.imwrite(file_name, self.processed_image)
                print("save_image: Image saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving image: {e}")
                print(f"save_image: Error - {e}")

    def convert_to_vector(self, file_name, image, output_format="svg"):
        """Converts the image to SVG using Potrace."""
        # Save the image to a temporary file
        temp_image_path = "temp.bmp"
        cv2.imwrite(temp_image_path, image)

        # Determine the path to potrace.exe
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_path = sys._MEIPASS
        else:
            # Running as a script
            base_path = os.path.dirname(os.path.abspath(__file__))

        # First, try the same directory as the script
        potrace_path = os.path.join(base_path, "potrace.exe")

        # If not found, try the /potrace subfolder
        if not os.path.exists(potrace_path):
            potrace_path = os.path.join(base_path, "potrace", "potrace.exe")

        # Construct the Potrace command
        command = [potrace_path, "-s", temp_image_path, "-o", file_name]


        try:
            # Execute Potrace
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', timeout=60)

            # Print the standard output and standard error for debugging
            print("Potrace Output:", result.stdout)
            print("Potrace Error:", result.stderr)

            # Remove the temporary image file
            os.remove(temp_image_path)

            print(f"{output_format.upper()} saved to {file_name}")

        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Potrace error: {e.stderr}")
            print(f"Potrace error: {e.stderr}")
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Potrace not found. Please ensure it is bundled with the application.")
            print("Potrace not found. Please ensure it is bundled with the application.")
        except subprocess.TimeoutExpired:
             QMessageBox.critical(self, "Error", "Potrace process timed out.")
             print("Potrace process timed out.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error converting to {output_format.upper()}: {e}")
            print(f"Error converting to {output_format.upper()}: {e}")
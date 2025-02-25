import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QSize
from PyQt5.QtGui import QImage, QPixmap, QTransform, QColor, QPainter
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QSlider, QFileDialog,
    QHBoxLayout, QVBoxLayout, QWidget, QGridLayout, QComboBox, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsColorizeEffect,
    QMessageBox, QGraphicsItem
)
from PyQt5.QtSvg import QSvgRenderer
from processing import apply_enhancements, process_with_ai_model

class LineDrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None             # Original image (cv2 BGR)
        self.processed_image = None   # Processed image (cv2; may be grayscale or color)
        self.cropped_image = None     # Final cropped image (numpy array)
        self.pixmap_item = None       # QGraphicsPixmapItem for the processed image
        self.overlay_item = None      # QGraphicsPixmapItem for the overlay border (now a QGraphicsPixmapItem)
        self.current_svg_path = None  # Currently loaded SVG file path for border overlay
        self.is_displaying = False   # Flag to prevent re-entrant calls to display_image
        self.svg_scale = 1.0          # Scale factor for the SVG
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
        self.loaded_image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed) # Changed to Fixed
        self.loaded_image_label.setFixedSize(600, 400)  # Set a fixed size
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
        self.save_button.clicked.connect(self.save_image)
        self.brightness_slider.valueChanged.connect(self.update_all)  # Connect to update_all
        self.contrast_slider.valueChanged.connect(self.update_all)   # Connect to update_all
        self.sharpness_slider.valueChanged.connect(self.update_all)  # Connect to update_all
        self.blur_slider.valueChanged.connect(self.update_all)       # Connect to update_all
        self.edge_sensitivity_slider.valueChanged.connect(self.update_all) # Connect to update_all
        self.threshold_slider.valueChanged.connect(self.update_all)    # Connect to update_all
        self.line_thickness_slider.valueChanged.connect(self.update_all) # Connect to update_all

        # Load SVG Border Overlay
        self.load_svg_button = QPushButton('Load SVG Border', self)
        self.load_svg_button.clicked.connect(self.load_svg_border)

        # Clear Border Button
        self.clear_border_button = QPushButton('Clear Border', self)
        self.clear_border_button.clicked.connect(self.clear_border)

        # SVG Scale Slider
        self.svg_scale_label = QLabel('SVG Scale', self)
        self.svg_scale_slider = QSlider(Qt.Horizontal)
        self.svg_scale_slider.setRange(1, 200)  # Scale from 0.01 to 2.00
        self.svg_scale_slider.setValue(100)
        self.svg_scale_slider.valueChanged.connect(self.update_svg_scale)

        # Combo box for processing method
        self.processing_method_label = QLabel('Processing Method', self)
        self.processing_method_combo = QComboBox(self)
        self.processing_method_combo.addItem('Threshold')
        self.processing_method_combo.addItem('Edge Detection')
        self.processing_method_combo.currentIndexChanged.connect(self.update_all)  # Connect to update_all

        # Crop Button
        self.crop_button = QPushButton('Crop Image', self)
        self.crop_button.clicked.connect(self.crop_image)

        left_layout.addWidget(self.loaded_image_label)
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.save_button)

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
        left_layout.addWidget(self.load_svg_button)
        left_layout.addWidget(self.clear_border_button)
        left_layout.addWidget(self.svg_scale_label)
        left_layout.addWidget(self.svg_scale_slider)
        left_layout.addWidget(self.processing_method_label)
        left_layout.addWidget(self.processing_method_combo)
        left_layout.addWidget(self.crop_button)

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
            self.is_displaying = False
            print("display_image: Finished")

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

        # Update the border overlay AFTER the image is updated
        self.update_border_overlay()
        self.pixmap_item.setZValue(0) # Ensure image is behind the SVG



    def load_svg_border(self):
        """Loads an SVG border overlay."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load SVG Border", "", "SVG Files (*.svg);;All Files (*)", options=options)
        if file_name:
            try:
                self.current_svg_path = file_name
                self.update_border_overlay()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading SVG: {e}")

    def clear_border(self):
        """Clears the border overlay."""
        self.current_svg_path = None
        self.update_border_overlay()

    def update_border_overlay(self):
        """Updates the border overlay on the processed image."""
        if self.overlay_item:
            self.scene.removeItem(self.overlay_item)
            self.overlay_item = None

        if self.current_svg_path and self.pixmap_item:
            try:
                renderer = QSvgRenderer(self.current_svg_path)
                if not renderer.isValid():
                    raise ValueError(f"Invalid SVG file: {self.current_svg_path}")

                # Get the default size of the SVG
                svg_size = renderer.defaultSize()

                # Calculate the scaled size
                scaled_size = svg_size * self.svg_scale

                # Create a QPixmap to render the SVG onto
                pixmap = QPixmap(scaled_size)
                pixmap.fill(Qt.transparent)  # Make the background transparent

                # Use a QPainter to draw the SVG onto the QPixmap
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.Antialiasing)

                # Apply a color filter (red)
                painter.setCompositionMode(QPainter.CompositionMode_SourceAtop)
                painter.fillRect(pixmap.rect(), Qt.red)

                renderer.render(painter)  # Render the SVG
                painter.end()

                # Create a QGraphicsPixmapItem to display the QPixmap
                self.overlay_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.overlay_item)
                self.overlay_item.setPos(self.pixmap_item.boundingRect().topLeft())
                self.overlay_item.setZValue(1) # Ensure SVG is on top


            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error displaying SVG: {e}")


    def update_svg_scale(self, value):
        """Updates the SVG scale based on the slider value."""
        self.svg_scale = value / 100.0
        self.update_border_overlay()

    def crop_image(self):
        """Crops the image."""
        if self.processed_image is None:
            return

        # Define a fixed crop area (adjust these values as needed)
        crop_x1 = 50
        crop_y1 = 50
        crop_x2 = 350
        crop_y2 = 250

        try:
            self.cropped_image = self.processed_image[crop_y1:crop_y2, crop_x1:crop_x2]
            self.update_display()  # Update display with cropped image
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error cropping image: {e}")

    def save_image(self):
        """Saves the processed image to a file."""
        if self.processed_image is None:
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            try:
                if self.processed_image.ndim == 2:  # Grayscale image
                    cv2.imwrite(file_name, self.processed_image)
                else:
                    cv2.imwrite(file_name, self.processed_image)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving image: {e}")
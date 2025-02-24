import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QLabel, QPushButton, QSlider, QFileDialog,
                             QHBoxLayout, QVBoxLayout, QWidget, QGridLayout, QComboBox, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from processing import (
    apply_enhancements,
    process_with_ai_model,
    crop_with_svg
)

class LineDrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.processed_image = None
        self.cropped_image = None
        self.initUI()

    def initUI(self):
        # Main window settings
        self.setWindowTitle('Line Drawing App')
        self.setGeometry(100, 100, 1400, 800)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        slider_layout = QGridLayout()

        # Labels to display images
        self.loaded_image_label = QLabel(self)
        self.loaded_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.loaded_image_label.setAlignment(Qt.AlignCenter)
        self.loaded_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #000;")

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #000;")

        # Load and Save buttons
        self.load_button = QPushButton('Load Image', self)
        self.save_button = QPushButton('Save Image', self)

        # Enhancement sliders with labels
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

        self.method_label = QLabel('Method', self)
        self.method_combo = QComboBox(self)
        self.method_combo.addItem('Threshold')
        self.method_combo.addItem('Edge Detection')

        # Cropper drop menu and button
        self.cropper_label = QLabel('Crop Shape', self)
        self.cropper_combo = QComboBox(self)
        self.load_cropper_files()  # Load cropper SVG files from /cropper folder
        self.crop_button = QPushButton('Crop Image', self)

        # Connect buttons and sliders
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_image)
        self.brightness_slider.valueChanged.connect(self.update_image)
        self.contrast_slider.valueChanged.connect(self.update_image)
        self.sharpness_slider.valueChanged.connect(self.update_image)
        self.blur_slider.valueChanged.connect(self.update_image)
        self.edge_sensitivity_slider.valueChanged.connect(self.update_image)
        self.threshold_slider.valueChanged.connect(self.update_image)
        self.line_thickness_slider.valueChanged.connect(self.update_image)
        self.method_combo.currentIndexChanged.connect(self.update_image)
        self.crop_button.clicked.connect(self.crop_image)

        # Add widgets to slider layout
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
        slider_layout.addWidget(self.method_label, 7, 0)
        slider_layout.addWidget(self.method_combo, 7, 1)

        # Add widgets to left layout
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.loaded_image_label)
        left_layout.addLayout(slider_layout)
        left_layout.addWidget(self.save_button)

        # Add widgets to right layout
        right_layout.addWidget(self.processed_image_label)
        right_layout.addWidget(self.cropper_label)
        right_layout.addWidget(self.cropper_combo)
        right_layout.addWidget(self.crop_button)

        # Add left and right layouts to main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Set the main layout in a central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_cropper_files(self):
        """
        Loads SVG files from the /cropper directory and populates the cropper combo box.
        """
        cropper_folder = os.path.join(os.getcwd(), "cropper")
        if os.path.exists(cropper_folder):
            for file in os.listdir(cropper_folder):
                if file.lower().endswith(".svg"):
                    self.cropper_combo.addItem(file, os.path.join(cropper_folder, file))
        else:
            # If folder doesn't exist, add a default item
            self.cropper_combo.addItem("No cropper files found")

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "",
                                                   "All Files (*);;Image Files (*.png;*.jpg;*.bmp)",
                                                   options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            if self.image is None:
                print("Error: Unable to load image!")
                return
            self.display_image(self.image, self.loaded_image_label, scale=0.7)
            self.update_image()

    def save_image(self):
        if self.cropped_image is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                       "PNG Files (*.png);;JPG Files (*.jpg);;BMP Files (*.bmp);;SVG Files (*.svg)",
                                                       options=options)
            if file_name:
                if file_name.endswith('.svg'):
                    # For saving as SVG, you may call an SVG export function (not implemented here)
                    print("Saving as SVG is not implemented for cropped images")
                else:
                    cv2.imwrite(file_name, self.cropped_image)
        elif self.processed_image is not None:
            # Fallback: save the processed image if no cropping was performed
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                       "PNG Files (*.png);;JPG Files (*.jpg);;BMP Files (*.bmp)",
                                                       options=options)
            if file_name:
                cv2.imwrite(file_name, self.processed_image)

    def update_image(self):
        if self.image is not None:
            params = {
                'brightness': self.brightness_slider.value(),
                'contrast': self.contrast_slider.value(),
                'sharpness': self.sharpness_slider.value(),
                'blur': self.blur_slider.value(),
                'threshold': self.threshold_slider.value(),
                'edge_sensitivity': self.edge_sensitivity_slider.value(),
                'line_thickness': self.line_thickness_slider.value(),
                'method': self.method_combo.currentText()
            }
            enhanced_image = apply_enhancements(self.image, params)
            self.processed_image = process_with_ai_model(enhanced_image, params)
            self.display_image(enhanced_image, self.loaded_image_label, scale=0.7)
            self.display_image(self.processed_image, self.processed_image_label, is_gray=True, scale=0.7)

    def crop_image(self):
        """
        Crops the processed image using the selected cropping vector (SVG).
        """
        if self.processed_image is not None:
            cropper_index = self.cropper_combo.currentIndex()
            cropper_path = self.cropper_combo.itemData(cropper_index)
            if cropper_path is None:
                print("No valid cropper file selected.")
                return
            self.cropped_image = crop_with_svg(self.processed_image, cropper_path)
            self.display_image(self.cropped_image, self.processed_image_label, is_gray=False, scale=0.7)

    def display_image(self, image, label, is_gray=False, scale=1.0):
        height, width = image.shape[:2]
        new_size = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        qformat = QImage.Format_Indexed8 if is_gray else QImage.Format_RGB888
        if len(resized_image.shape) == 3 and not is_gray:
            if resized_image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(resized_image, resized_image.shape[1], resized_image.shape[0],
                           resized_image.strides[0], qformat)
        if not is_gray:
            out_image = out_image.rgbSwapped()
        label.setPixmap(QPixmap.fromImage(out_image))
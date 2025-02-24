import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QSlider, QFileDialog,
                             QHBoxLayout, QVBoxLayout, QWidget, QGridLayout, QComboBox, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from svgpathtools import parse_path, wsvg
from svgpathtools.paths2svg import disvg


class LineDrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None
        self.processed_image = None

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

        # Add left and right layouts to main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Set the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self):
        # Load image
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "All Files (*);;Image Files (*.png;*.jpg;*.bmp)", options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image, self.loaded_image_label, scale=0.7)
            self.update_image()

    def save_image(self):
        # Save processed image
        if self.processed_image is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg);;BMP Files (*.bmp);;SVG Files (*.svg)", options=options)
            if file_name:
                if file_name.endswith('.svg'):
                    self.save_as_svg(file_name, self.processed_image)
                else:
                    cv2.imwrite(file_name, self.processed_image)

    def save_as_svg(self, file_name, image):
        height, width = image.shape[:2]
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        paths = []
        for contour in contours:
            if len(contour) >= 2:  # Ensure there are at least 2 points in the contour
                path = "M "
                for point in contour:
                    x, y = point[0]
                    path += f"{x},{y} "
                path += "Z"
                paths.append(parse_path(path))
        
        if not paths:
            print("No valid contours found.")
            return

        disvg(paths, filename=file_name)

    def update_image(self):
        if self.image is not None:
            # Apply enhancements
            enhanced_image = self.apply_enhancements(self.image)
            # Process image with AI model
            self.processed_image = self.process_with_ai_model(enhanced_image)
            self.display_image(enhanced_image, self.loaded_image_label, scale=0.7)
            self.display_image(self.processed_image, self.processed_image_label, is_gray=True, scale=0.7)

    def apply_enhancements(self, image):
        # Convert to PIL image for enhancements
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(self.brightness_slider.value() / 50.0)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(self.contrast_slider.value() / 50.0)
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(self.sharpness_slider.value() / 50.0)
        enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Apply blur
        blur_value = self.blur_slider.value()
        if blur_value > 0:
            enhanced_image = cv2.GaussianBlur(enhanced_image, (blur_value*2+1, blur_value*2+1), 0)

        return enhanced_image

    def process_with_ai_model(self, image):
        method = self.method_combo.currentText()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == 'Threshold':
            _, binary_image = cv2.threshold(gray_image, self.threshold_slider.value(), 255, cv2.THRESH_BINARY)
        else:  # Edge Detection
            edge_sensitivity = self.edge_sensitivity_slider.value()
            edges = cv2.Canny(gray_image, edge_sensitivity, edge_sensitivity * 2)
            thickness = self.line_thickness_slider.value()
            kernel = np.ones((thickness, thickness), np.uint8)
            thick_edges = cv2.dilate(edges, kernel, iterations=1)
            _, binary_image = cv2.threshold(thick_edges, self.threshold_slider.value(), 255, cv2.THRESH_BINARY_INV)

        return binary_image

    def display_image(self, image, label, is_gray=False, scale=1.0):
        height, width = image.shape[:2]
        new_size = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        qformat = QImage.Format_Indexed8 if is_gray else QImage.Format_RGB888
        if len(resized_image.shape) == 3 and not is_gray:  # rows, cols, channels
            if resized_image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(resized_image, resized_image.shape[1], resized_image.shape[0], resized_image.strides[0], qformat)
        if not is_gray:
            out_image = out_image.rgbSwapped()
        label.setPixmap(QPixmap.fromImage(out_image))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LineDrawingApp()
    ex.show()
    sys.exit(app.exec_())
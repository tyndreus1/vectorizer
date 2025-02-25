import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QImage, QPixmap, QTransform
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QSlider, QFileDialog,
    QHBoxLayout, QVBoxLayout, QWidget, QGridLayout, QComboBox, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsColorizeEffect
)
from PyQt5.QtSvg import QGraphicsSvgItem
from processing import apply_enhancements, process_with_ai_model

class LineDrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None             # Original image (cv2 BGR)
        self.processed_image = None   # Processed image (cv2; may be grayscale or color)
        self.cropped_image = None     # Final cropped image (numpy array)
        self.pixmap_item = None       # QGraphicsPixmapItem for the processed image
        self.overlay_item = None      # QGraphicsSvgItem for the overlay border
        self.current_svg_path = None  # Currently loaded SVG file path for border overlay
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
        self.loaded_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

        self.method_label = QLabel('Method', self)
        self.method_combo = QComboBox(self)
        self.method_combo.addItem('Threshold')
        self.method_combo.addItem('Edge Detection')

        # Assemble slider layout
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

        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.loaded_image_label)
        left_layout.addLayout(slider_layout)
        left_layout.addWidget(self.save_button)

        # Right panel: QGraphicsView for processed image and border overlay controls
        self.crop_view = QGraphicsView(self)
        self.crop_view.setMinimumSize(500, 500)
        self.scene = QGraphicsScene(self)
        self.crop_view.setScene(self.scene)

        # Overlay adjustment controls
        self.overlay_scale_label = QLabel('Overlay Scale', self)
        self.overlay_scale_slider = QSlider(Qt.Horizontal)
        self.overlay_scale_slider.setRange(50, 200)
        self.overlay_scale_slider.setValue(100)

        self.overlay_x_label = QLabel('Overlay X Offset', self)
        self.overlay_x_slider = QSlider(Qt.Horizontal)
        self.overlay_x_slider.setRange(-200, 200)
        self.overlay_x_slider.setValue(0)

        self.overlay_y_label = QLabel('Overlay Y Offset', self)
        self.overlay_y_slider = QSlider(Qt.Horizontal)
        self.overlay_y_slider.setRange(-200, 200)
        self.overlay_y_slider.setValue(0)

        self.cropper_label = QLabel('Crop Border (SVG)', self)
        self.cropper_combo = QComboBox(self)
        self.load_cropper_files()  # Load SVG files from the cropper folder

        self.load_overlay_button = QPushButton('Load Border Overlay', self)
        self.crop_button = QPushButton('Crop Image', self)

        right_layout.addWidget(self.crop_view)
        right_layout.addWidget(self.cropper_label)
        right_layout.addWidget(self.cropper_combo)
        right_layout.addWidget(self.load_overlay_button)
        right_layout.addWidget(self.overlay_scale_label)
        right_layout.addWidget(self.overlay_scale_slider)
        right_layout.addWidget(self.overlay_x_label)
        right_layout.addWidget(self.overlay_x_slider)
        right_layout.addWidget(self.overlay_y_label)
        right_layout.addWidget(self.overlay_y_slider)
        right_layout.addWidget(self.crop_button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect signals
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

        self.load_overlay_button.clicked.connect(self.load_overlay)
        self.overlay_scale_slider.valueChanged.connect(self.update_overlay_transform)
        self.overlay_x_slider.valueChanged.connect(self.update_overlay_transform)
        self.overlay_y_slider.valueChanged.connect(self.update_overlay_transform)
        self.crop_button.clicked.connect(self.crop_image)

    def load_cropper_files(self):
        cropper_folder = os.path.join(os.getcwd(), "cropper")
        if os.path.exists(cropper_folder):
            for file in os.listdir(cropper_folder):
                if file.lower().endswith(".svg"):
                    self.cropper_combo.addItem(file, os.path.join(cropper_folder, file))
        else:
            self.cropper_combo.addItem("No cropper files found")

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.png *.jpg *.bmp);;All Files (*)",
            options=options
        )
        if file_name:
            self.image = cv2.imread(file_name)
            if self.image is None:
                print("Error: Unable to load image!")
                return
            self.display_image(self.image, self.loaded_image_label, scale=0.7)
            self.update_image()

    def save_image(self):
        save_image = self.cropped_image if self.cropped_image is not None else self.processed_image
        if save_image is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image",
                "",
                "PNG Files (*.png);;JPG Files (*.jpg);;BMP Files (*.bmp)",
                options=options
            )
            if file_name:
                cv2.imwrite(file_name, save_image)

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
            self.refreshCropView()
            self.cropped_image = None

    def refreshCropView(self):
        if self.processed_image is None:
            return
        # Convert processed_image to QImage
        if len(self.processed_image.shape) == 2:
            qformat = QImage.Format_Grayscale8
            image_data = self.processed_image
            bytes_per_line = self.processed_image.shape[1]
        else:
            rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            qformat = QImage.Format_RGB888
            image_data = rgb_image
            bytes_per_line = self.processed_image.shape[1] * 3

        height, width = self.processed_image.shape[:2]
        qimg = QImage(image_data.data, width, height, bytes_per_line, qformat)
        pixmap = QPixmap.fromImage(qimg)

        # If pixmap_item doesn't exist, create and add it; otherwise update its pixmap.
        if self.pixmap_item is None:
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.pixmap_item.setZValue(0)
            self.scene.addItem(self.pixmap_item)
        else:
            self.pixmap_item.setPixmap(pixmap)

        # Ensure the overlay stays on top.
        if self.overlay_item is not None:
            if self.overlay_item.scene() is None:
                self.scene.addItem(self.overlay_item)
            self.overlay_item.setZValue(100)

        self.scene.setSceneRect(0, 0, width, height)
        self.crop_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def load_overlay(self):
        index = self.cropper_combo.currentIndex()
        svg_path = self.cropper_combo.itemData(index)
        if svg_path is None or not os.path.exists(svg_path):
            print("Invalid SVG file selected.")
            return
        self.current_svg_path = svg_path
        self.overlay_item = QGraphicsSvgItem(svg_path)
        if self.processed_image is not None:
            self.overlay_item.setPos(0, 0)
            self.overlay_item.setTransform(QTransform())
        self.overlay_item.setZValue(100)
        effect = QGraphicsColorizeEffect()
        effect.setColor(Qt.red)
        self.overlay_item.setGraphicsEffect(effect)
        # Instead of clearing the scene, simply add the overlay if not already present.
        if self.overlay_item.scene() is None:
            self.scene.addItem(self.overlay_item)
        self.overlay_scale_slider.setValue(100)
        self.overlay_x_slider.setValue(0)
        self.overlay_y_slider.setValue(0)

    def update_overlay_transform(self):
        if self.overlay_item is None:
            return
        scale_val = self.overlay_scale_slider.value() / 100.0
        x_offset = self.overlay_x_slider.value()
        y_offset = self.overlay_y_slider.value()
        transform = QTransform()
        transform.translate(x_offset, y_offset)
        transform.scale(scale_val, scale_val)
        self.overlay_item.setTransform(transform)

    def crop_image(self):
        if self.overlay_item is None or self.processed_image is None:
            print("No overlay loaded or no processed image available.")
            return
        overlay_rect_scene = self.overlay_item.mapRectToScene(self.overlay_item.boundingRect())
        print("Overlay scene rect:", overlay_rect_scene)
        scene_rect = self.scene.sceneRect()
        img_height, img_width = self.processed_image.shape[:2]
        scale_x = img_width / scene_rect.width()
        scale_y = img_height / scene_rect.height()
        crop_x = int(overlay_rect_scene.x() * scale_x)
        crop_y = int(overlay_rect_scene.y() * scale_y)
        crop_w = int(overlay_rect_scene.width() * scale_x)
        crop_h = int(overlay_rect_scene.height() * scale_y)
        if crop_x < 0: crop_x = 0
        if crop_y < 0: crop_y = 0
        if crop_x + crop_w > img_width: crop_w = img_width - crop_x
        if crop_y + crop_h > img_height: crop_h = img_height - crop_y
        print("Cropping image at:", crop_x, crop_y, crop_w, crop_h)
        self.cropped_image = self.processed_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()

        if len(self.cropped_image.shape) == 2:
            qformat = QImage.Format_Grayscale8
            bytes_per_line = self.cropped_image.shape[1]
        else:
            rgb_crop = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB)
            qformat = QImage.Format_RGB888
            bytes_per_line = self.cropped_image.shape[1] * 3

        qimg = QImage(
            self.cropped_image.data if len(self.cropped_image.shape)==2 else rgb_crop.data,
            self.cropped_image.shape[1],
            self.cropped_image.shape[0],
            bytes_per_line,
            qformat
        )
        crop_pixmap = QPixmap.fromImage(qimg)
        crop_scene = QGraphicsScene(self)
        crop_scene.addItem(QGraphicsPixmapItem(crop_pixmap))
        self.crop_view.setScene(crop_scene)
        print("Cropping complete.")

    def display_image(self, image, label, is_gray=False, scale=1.0):
        height, width = image.shape[:2]
        new_size = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        if is_gray or len(resized_image.shape) == 2:
            qformat = QImage.Format_Grayscale8
            image_data = resized_image
            bytes_per_line = resized_image.shape[1]
        else:
            if resized_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                qformat = QImage.Format_RGB888
                image_data = rgb_image
                bytes_per_line = resized_image.shape[1] * 3
            else:
                image_data = resized_image
                qformat = QImage.Format_RGB888
                bytes_per_line = resized_image.shape[1] * 3
        out_image = QImage(image_data.data, image_data.shape[1], image_data.shape[0], bytes_per_line, qformat)
        label.setPixmap(QPixmap.fromImage(out_image))
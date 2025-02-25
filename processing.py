import cv2
import numpy as np
from PIL import Image, ImageEnhance

def apply_enhancements(image, params):
    """
    Applies enhancements such as brightness, contrast, sharpness, and blur.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(params['brightness'] / 50.0)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(params['contrast'] / 50.0)
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(params['sharpness'] / 50.0)
    enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    blur_value = params['blur']
    if blur_value > 0:
        enhanced_image = cv2.GaussianBlur(enhanced_image, (blur_value * 2 + 1, blur_value * 2 + 1), 0)
    return enhanced_image

def process_with_ai_model(image, params):
    """
    Processes the image using thresholding or edge detection.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if params['method'] == 'Threshold':
        _, binary_image = cv2.threshold(gray_image, params['threshold'], 255, cv2.THRESH_BINARY)
    else:  # Edge Detection
        edge_sensitivity = params['edge_sensitivity']
        edges = cv2.Canny(gray_image, edge_sensitivity, edge_sensitivity * 2)
        thickness = params['line_thickness']
        kernel = np.ones((thickness, thickness), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        _, binary_image = cv2.threshold(thick_edges, params['threshold'], 255, cv2.THRESH_BINARY_INV)
    return binary_image
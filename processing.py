import cv2
import numpy as np
from PIL import Image, ImageEnhance
from svgpathtools import parse_path
from svgpathtools.paths2svg import disvg

def apply_enhancements(image, params):
    """
    Applies image enhancements such as brightness, contrast, sharpness, and blur.
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
    Processes the image using either thresholding or edge detection.
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

def crop_with_svg(image, svg_file_path):
    """
    Crops the given image using a cropping vector defined in an SVG file.
    For demonstration:
      - If the SVG file name contains "circle", a circular mask is applied.
      - If it contains "rectangle", the image is returned as is.
      - For any other shape (e.g. heart), the original image is returned.
    This function can be extended to perform precise cropping based on SVG paths.
    """
    if "circle" in svg_file_path.lower():
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(center[0], center[1], w - center[0], h - center[1])
        cv2.circle(mask, center, radius, 255, -1)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
    # For rectangle, return the image as is (or implement cropping coordinates as needed)
    if "rectangle" in svg_file_path.lower():
        return image
    # For heart or other shapes, placeholder: return the original image
    return image

def save_as_svg(file_name, image):
    """
    Converts a binary image to SVG by extracting contours.
    """
    height, width = image.shape[:2]
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        if len(contour) >= 2:
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
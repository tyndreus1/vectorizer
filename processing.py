import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import xml.etree.ElementTree as ET
import re

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

def create_mask_from_svg(svg_path, width, height, scale=1.0, offset_x=0, offset_y=0):
    """
    Create a binary mask from an SVG file.
    """
    try:
        # Create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Parse SVG to extract path data
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Get SVG viewBox if it exists
        viewbox = root.get('viewBox')
        if viewbox:
            vb_parts = [float(x) for x in viewbox.split()]
            svg_width, svg_height = vb_parts[2], vb_parts[3]
        else:
            svg_width = float(root.get('width', '100').strip('px'))
            svg_height = float(root.get('height', '100').strip('px'))
        
        # Scale factors to fit SVG to our mask dimensions
        scale_x = (width * scale) / svg_width
        scale_y = (height * scale) / svg_height
        
        # Center offset
        center_x = width//2 + offset_x
        center_y = height//2 + offset_y
        
        # Extract path elements and draw them on the mask
        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        for path in paths:
            d = path.get('d', '')
            if d:
                points = parse_svg_path(d)
                if points and len(points) > 1:
                    transformed_points = []
                    for x, y in points:
                        tx = int(x * scale_x + center_x - (svg_width * scale_x)/2)
                        ty = int(y * scale_y + center_y - (svg_height * scale_y)/2)
                        transformed_points.append((tx, ty))
                    
                    pts = np.array([transformed_points], dtype=np.int32)
                    cv2.fillPoly(mask, pts, 255)
        
        return mask
    except Exception as e:
        print(f"Error creating mask from SVG: {e}")
        return np.ones((height, width), dtype=np.uint8) * 255

def parse_svg_path(d):
    """
    Parse SVG path data to extract points.
    This is a simplified parser that works for basic paths.
    """
    # Extract all numeric values from the path data
    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
    
    # Convert to floats and pair them as (x,y) coordinates
    points = []
    for i in range(0, len(numbers) - 1, 2):
        try:
            x = float(numbers[i])
            y = float(numbers[i+1])
            points.append((x, y))
        except (IndexError, ValueError):
            pass
    
    return points

def crop_with_mask(image, mask):
    """
    Crop image using a binary mask.
    Returns a binary (black and white) result based on the mask.
    """
    # Ensure mask and image dimensions match
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)
    return result
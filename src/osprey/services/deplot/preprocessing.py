"""OpenCV preprocessing for chart images before DePlot extraction.

Provides chart region detection, contrast enhancement, and optional
trace isolation to improve DePlot accuracy on screenshots from control
room displays (Phoebus, StripTool, Grafana).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("osprey.services.deplot.preprocessing")


def detect_chart_region(image: np.ndarray) -> np.ndarray:
    """Detect and crop the primary chart region from a screenshot.

    Uses contour detection to find the largest rectangular region, which
    is typically the chart plot area. Falls back to the full image if
    no suitable contour is found.

    Args:
        image: BGR image as numpy array (OpenCV format).

    Returns:
        Cropped BGR image containing the chart region.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges to close gaps in chart borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.debug("No contours found, returning full image")
        return image

    # Find the largest contour by area
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.1  # At least 10% of the image

    best_contour = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area > best_area:
            best_contour = contour
            best_area = area

    if best_contour is None:
        logger.debug("No contour meets minimum area threshold, returning full image")
        return image

    x, y, w, h = cv2.boundingRect(best_contour)

    # Add small padding (2% of each dimension)
    pad_x = max(int(w * 0.02), 2)
    pad_y = max(int(h * 0.02), 2)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(image.shape[1] - x, w + 2 * pad_x)
    h = min(image.shape[0] - y, h + 2 * pad_y)

    logger.debug("Chart region detected: x=%d, y=%d, w=%d, h=%d", x, y, w, h)
    return image[y : y + h, x : x + w]


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Improves readability of chart lines and text, especially for screenshots
    with poor contrast or uneven lighting.

    Args:
        image: BGR image as numpy array.

    Returns:
        Contrast-enhanced BGR image.
    """
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def isolate_trace(image: np.ndarray, hue_range: tuple[int, int] = (0, 180)) -> np.ndarray:
    """Isolate a specific color trace from a chart using HSV thresholding.

    Useful for extracting a single data series from a multi-trace chart.
    Default hue range captures all colors (no filtering).

    Args:
        image: BGR image as numpy array.
        hue_range: (low, high) hue range in OpenCV scale (0-180).

    Returns:
        Masked BGR image with only the selected trace colors.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([hue_range[0], 50, 50])
    upper = np.array([hue_range[1], 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Create white background with trace overlay
    result = np.full_like(image, 255)
    result[mask > 0] = image[mask > 0]

    return result


def preprocess_chart(image: np.ndarray, detect_region: bool = True) -> np.ndarray:
    """Full preprocessing pipeline for chart images.

    Combines chart region detection and contrast enhancement.

    Args:
        image: BGR image as numpy array.
        detect_region: Whether to attempt chart region detection.

    Returns:
        Preprocessed BGR image ready for DePlot extraction.
    """
    if detect_region:
        image = detect_chart_region(image)
    image = enhance_contrast(image)
    return image

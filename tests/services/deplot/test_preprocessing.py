"""Tests for DePlot preprocessing (OpenCV chart processing).

These tests require opencv-python-headless (``uv sync --extra graph``).
They are skipped automatically when OpenCV is not installed.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed (install with: uv sync --extra graph)")


def _make_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a simple BGR test image with a white rectangle on gray background."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)  # Gray background
    # Draw a white rectangle (simulating a chart area)
    x1, y1 = width // 4, height // 4
    x2, y2 = 3 * width // 4, 3 * height // 4
    img[y1:y2, x1:x2] = 255
    return img


def _make_blank_image(width: int = 100, height: int = 100) -> np.ndarray:
    """Create a uniform gray image with no contours."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


class TestDetectChartRegion:
    """Tests for detect_chart_region()."""

    def test_returns_ndarray(self):
        from osprey.services.deplot.preprocessing import detect_chart_region

        img = _make_test_image()
        result = detect_chart_region(img)
        assert isinstance(result, np.ndarray)

    def test_crops_to_chart_region(self):
        from osprey.services.deplot.preprocessing import detect_chart_region

        img = _make_test_image(640, 480)
        result = detect_chart_region(img)
        # Cropped region should be smaller than original
        assert result.shape[0] <= img.shape[0]
        assert result.shape[1] <= img.shape[1]

    def test_returns_full_image_when_no_contours(self):
        from osprey.services.deplot.preprocessing import detect_chart_region

        img = _make_blank_image()
        result = detect_chart_region(img)
        # Should return the full image when no significant contours found
        assert result.shape == img.shape


class TestEnhanceContrast:
    """Tests for enhance_contrast()."""

    def test_returns_same_shape(self):
        from osprey.services.deplot.preprocessing import enhance_contrast

        img = _make_test_image()
        result = enhance_contrast(img)
        assert result.shape == img.shape

    def test_returns_uint8(self):
        from osprey.services.deplot.preprocessing import enhance_contrast

        img = _make_test_image()
        result = enhance_contrast(img)
        assert result.dtype == np.uint8


class TestIsolateTrace:
    """Tests for isolate_trace()."""

    def test_returns_same_shape(self):
        from osprey.services.deplot.preprocessing import isolate_trace

        img = _make_test_image()
        result = isolate_trace(img)
        assert result.shape == img.shape

    def test_with_narrow_hue_range(self):
        from osprey.services.deplot.preprocessing import isolate_trace

        img = _make_test_image()
        result = isolate_trace(img, hue_range=(0, 30))
        assert result.shape == img.shape
        # With narrow hue range on a gray image, most pixels should be white (background)
        white_ratio = np.mean(result == 255)
        assert white_ratio > 0.5


class TestPreprocessChart:
    """Tests for the full preprocess_chart pipeline."""

    def test_full_pipeline(self):
        from osprey.services.deplot.preprocessing import preprocess_chart

        img = _make_test_image()
        result = preprocess_chart(img, detect_region=True)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3  # Still a color image

    def test_skip_region_detection(self):
        from osprey.services.deplot.preprocessing import preprocess_chart

        img = _make_test_image()
        result = preprocess_chart(img, detect_region=False)
        # Without region detection, width/height should match (contrast only)
        assert result.shape[0] == img.shape[0]
        assert result.shape[1] == img.shape[1]

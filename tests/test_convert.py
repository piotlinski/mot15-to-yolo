import pytest

from src.convert import convert_to_yolo


@pytest.mark.parametrize(
    "bbox, width, height, expected",
    [
        ([10, 15, 3, 4], 20, 20, [0.575, 0.85, 0.15, 0.2]),
        ([0, 0, 20, 20], 20, 20, [0.5, 0.5, 1, 1]),
        ([32, 76, 12, 4], 100, 100, [0.38, 0.78, 0.12, 0.04]),
    ],
)
def test_convert_to_yolo(bbox, width, height, expected):
    """Test converting x1y1wh bbox to yolo format."""
    assert convert_to_yolo(bbox, width, height) == expected

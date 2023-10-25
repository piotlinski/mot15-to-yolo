import pytest

from src.convert import convert_parsed_to_yolo, convert_to_yolo


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


def test_convert_parsed_to_yolo():
    """Test converting parsed annotations to yolo format."""
    parsed = [
        {"bboxes": [[10, 15, 3, 4], [0, 0, 20, 20]], "classes": [0, 0], "ids": [0, 1]},
        {"bboxes": [], "classes": [], "ids": []},
        {"bboxes": [[32, 76, 12, 4]], "classes": [0], "ids": [1]},
    ]

    converted = list(convert_parsed_to_yolo(parsed, 100, 100))

    assert converted[0]["bboxes"] == [[0.115, 0.17, 0.03, 0.04], [0.1, 0.1, 0.2, 0.2]]
    assert converted[1]["bboxes"] == []
    assert converted[2]["bboxes"] == [[0.38, 0.78, 0.12, 0.04]]
    for old, new in zip(parsed, converted, strict=True):
        assert old["classes"] == new["classes"]
        assert old["ids"] == new["ids"]

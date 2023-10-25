from collections.abc import Iterable
from typing import Any


def convert_to_yolo(bbox: list[int], width: int, height: int) -> list[float]:
    """Convert x1y1wh bbox to yolo format."""
    x1, y1, w, h = bbox
    x_center = (x1 + w / 2) / width
    y_center = (y1 + h / 2) / height
    return [x_center, y_center, w / width, h / height]


def convert_parsed_to_yolo(
    annotations: Iterable[dict[str, list[Any]]], width: int, height: int
) -> Iterable[dict[str, list[Any]]]:
    """Convert parsed annotations to yolo format."""
    for annotation in annotations:
        annotation["bboxes"] = [
            convert_to_yolo(bbox, width, height) for bbox in annotation["bboxes"]
        ]
        yield annotation

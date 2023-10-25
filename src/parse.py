from collections.abc import Iterable
from configparser import ConfigParser
from pathlib import Path
from typing import Any


def parse_mot15_ini(in_file: Path) -> dict[str, Any]:
    """Parse the MOT15 seqinfo.ini file and return a dict with the relevant data."""
    info = ConfigParser()
    info.read(in_file)

    parsed = info["Sequence"]
    return {
        "name": parsed["name"],
        "directory": in_file.parent / parsed["imDir"],
        "framerate": float(parsed["frameRate"]),
        "length": int(parsed["seqLength"]),
        "width": int(parsed["imWidth"]),
        "height": int(parsed["imHeight"]),
        "extension": parsed["imExt"],
    }


def parse_mot15_gt(in_file: Path, length: int) -> Iterable[dict[str, list[Any]]]:
    """Parse MOT15 gt.txt file to a list of dicts with bboxes, classes and ids.

    .. note: the default bbox format is [x1, y1, w, h]
        we assume that the class is always 0
    """
    current_frame = 0
    annotation: dict[str, list[Any]] = {"bboxes": [], "classes": [], "ids": []}
    with open(in_file, "r") as fp:
        for line in fp:
            frame, obj, x1, y1, w, h, _, _, _, _ = line.split(",")
            frame_id = int(frame) - 1
            obj_id = int(obj) - 1
            if int(frame_id) > current_frame:
                for _ in range(frame_id - current_frame):
                    yield annotation
                    current_frame += 1
                    annotation = {"bboxes": [], "classes": [], "ids": []}
            annotation["bboxes"].append([float(x1), float(y1), float(w), float(h)])
            annotation["classes"].append(0)
            annotation["ids"].append(obj_id)

    while current_frame < length:
        yield annotation
        annotation = {"bboxes": [], "classes": [], "ids": []}
        current_frame += 1

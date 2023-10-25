from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.parse import parse_mot15_gt, parse_mot15_ini


@pytest.fixture
def seq_info():
    return """[Sequence]
name=Venice-1
imDir=img1
frameRate=30
seqLength=450
imWidth=1920
imHeight=1080
imExt=.jpg
"""


@pytest.fixture
def mot15_gt():
    return """1,1,415,449,129,269,1,-1,-1,-1
3,1,581,443,97,269,1,-1,-1,-1
3,2,1347,415,153,377,1,-1,-1,-1
3,3,1429,423,161,339,1,-1,-1,-1
6,1,413,449,131,269,1,-1,-1,-1
6,2,579,443,97,269,1,-1,-1,-1
6,3,1353,413,155,379,1,-1,-1,-1
6,4,1437,421,157,341,1,-1,-1,-1
"""


def test_parse_mot_15_ini(seq_info):
    """Check if MOT15 seqinfo.ini is parsed properly."""
    test_path = Path("dummy/path")
    with patch("builtins.open", mock_open(read_data=seq_info)):
        parsed = parse_mot15_ini(test_path)

    assert parsed["name"] == "Venice-1"
    assert parsed["directory"] == test_path.parent / "img1"
    assert parsed["framerate"] == 30.0
    assert parsed["length"] == 450
    assert parsed["width"] == 1920
    assert parsed["height"] == 1080
    assert parsed["extension"] == ".jpg"


@pytest.mark.parametrize("length", [7, 9, 10])
def test_parse_mot15_gt(length, mot15_gt):
    """Test if MOT15 ground truth is parsed properly."""
    with patch("builtins.open", mock_open(read_data=mot15_gt)):
        parsed = list(parse_mot15_gt(Path("dummy/path"), length))

    assert len(parsed) == length

    assert parsed[0]["bboxes"] == [[415, 449, 129, 269]]
    assert parsed[0]["classes"] == [0]
    assert parsed[0]["ids"] == [0]

    assert parsed[1]["bboxes"] == parsed[1]["classes"] == parsed[1]["ids"] == []

    assert parsed[2]["bboxes"] == [
        [581, 443, 97, 269],
        [1347, 415, 153, 377],
        [1429, 423, 161, 339],
    ]
    assert parsed[2]["classes"] == [0, 0, 0]
    assert parsed[2]["ids"] == [0, 1, 2]

    assert parsed[3]["bboxes"] == parsed[3]["classes"] == parsed[3]["ids"] == []
    assert parsed[4]["bboxes"] == parsed[4]["classes"] == parsed[4]["ids"] == []

    assert parsed[5]["bboxes"] == [
        [413, 449, 131, 269],
        [579, 443, 97, 269],
        [1353, 413, 155, 379],
        [1437, 421, 157, 341],
    ]
    assert parsed[5]["classes"] == [0, 0, 0, 0]
    assert parsed[5]["ids"] == [0, 1, 2, 3]

    for i in range(6, length):
        assert parsed[i]["bboxes"] == parsed[i]["classes"] == parsed[i]["ids"] == []

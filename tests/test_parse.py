from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.parse import parse_mot15_ini


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

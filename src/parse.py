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

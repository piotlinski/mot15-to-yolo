import argparse
import shutil
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm, trange

from src.config import VAL_SEQUENCES
from src.convert import convert_parsed_to_yolo
from src.parse import parse_mot15_gt, parse_mot15_ini


def prepare_obj_data(
    train: str | Path, valid: str | Path, names: str | Path, n_classes: int
) -> str:
    return (
        f"classes = {n_classes}\n"
        f"train = {train}\n"
        f"valid = {valid}\n"
        f"names = {names}\n"
        f"backup = backup/\n"
    )


def prepare_obj_names(classes: list[str]) -> str:
    return "\n".join(classes)


def prepare_txt(annotation: dict[str, list[Any]]) -> str:
    """Prepare TXT yolo annotation."""
    lines = [
        f"{cls} {' '.join(str(coord) for coord in bbox)}"
        for bbox, cls in zip(annotation["bboxes"], annotation["classes"], strict=True)
    ]
    return "\n".join(lines)


def process_directory(
    directory: Path,
) -> tuple[dict[str, Any], list[dict[str, list[Any]]]]:
    """Parse data from the given directory."""
    ini = parse_mot15_ini(directory / "seqinfo.ini")
    gt = parse_mot15_gt(directory / "gt" / "gt.txt", ini["length"])
    yolo_gt = convert_parsed_to_yolo(gt, ini["width"], ini["height"])
    return ini, list(yolo_gt)


def process_split(split_directory: Path, sequences: list[Path]):
    split = []
    split_directory.mkdir(exist_ok=True, parents=True)

    for seq in tqdm(
        sequences, desc=f"Processing {split_directory.name} split", leave=False
    ):
        seq_dir = split_directory / seq.name
        seq_dir.mkdir(exist_ok=True, parents=True)

        ini, yolo_gt = process_directory(seq)
        for idx in trange(ini["length"], desc=f"Processing {seq.name}", leave=False):
            filename = f"{idx + 1:06d}{ini['extension']}"
            in_filename = ini["directory"] / filename
            out_filename = seq_dir / filename
            shutil.copy(in_filename, out_filename)
            split.append(out_filename)

            out_txt = out_filename.with_suffix(".txt")
            with open(out_txt, "w") as fp:
                fp.write(prepare_txt(yolo_gt[idx]))
    return split


def prepare_dataset(directory: Path, output_directory: Path):
    sequences = [seq for seq in sorted(directory.iterdir()) if seq.is_dir()]
    train_sequences = [seq for seq in sequences if seq.name not in VAL_SEQUENCES]
    val_sequences = [seq for seq in sequences if seq.name in VAL_SEQUENCES]

    # prepare train dataset
    train = process_split(output_directory / "train", train_sequences)
    train_txt = output_directory / "train.txt"
    with open(train_txt, "w") as fp:
        fp.write("\n".join(str(path) for path in train))

    # prepare val dataset
    val = process_split(output_directory / "test", val_sequences)
    val_txt = output_directory / "test.txt"  # mismatch required by YOLRO
    with open(val_txt, "w") as fp:
        fp.write("\n".join(str(path) for path in val))

    # prepare obj.names
    obj_names = prepare_obj_names(["person"])
    obj_names_txt = output_directory / "obj.names"
    with open(obj_names_txt, "w") as fp:
        fp.write(obj_names)

    # prepare obj.data
    obj_data = prepare_obj_data(train_txt, val_txt, obj_names_txt, n_classes=1)
    obj_data_txt = output_directory / "obj.data"
    with open(obj_data_txt, "w") as fp:
        fp.write(obj_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    prepare_dataset(args.input_directory, args.output_directory)

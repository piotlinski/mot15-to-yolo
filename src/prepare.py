import argparse
import json
import shutil
from itertools import islice
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
        f"backup = data/\n"
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


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter.
    (source: itertools recipes)"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def process_split(
    split_directory: Path, sequences: list[Path], length: int | None = None
):
    split = []
    split_directory.mkdir(exist_ok=True, parents=True)

    for seq in tqdm(
        sequences, desc=f"Processing {split_directory.name} split", leave=False
    ):
        ini, yolo_gt = process_directory(seq)
        batch_length = length or ini["length"]

        for idx, subseq in enumerate(
            batched(range(ini["length"]), batch_length), start=1
        ):
            if len(subseq) < batch_length:
                print(
                    f"Skipping {seq.name} subsequence of length {len(subseq)} "
                    f"since it's shorter than {batch_length}"
                )
                continue

            seq_name = f"{seq.name}_{idx}" if length is None else seq.name
            seq_dir = split_directory / seq_name
            seq_dir.mkdir(exist_ok=True, parents=True)
            annotations = []

            for subidx in trange(
                batch_length, desc=f"Processing {seq.name} - subseq {idx}", leave=False
            ):
                filename = f"{subidx + 1:06d}{ini['extension']}"
                in_filename = ini["directory"] / filename
                out_filename = seq_dir / filename

                shutil.copy(in_filename, out_filename)
                split.append(out_filename)
                gt = yolo_gt[batch_length * (idx - 1) + subidx]
                annotations.append(gt)

                out_txt = out_filename.with_suffix(".txt")
                with open(out_txt, "w") as fp:
                    fp.write(prepare_txt(gt))

            with seq_dir.joinpath("annotations.json").open("w") as fp:
                json.dump(annotations, fp, indent=4)

    return split


def prepare_dataset(directory: Path, output_directory: Path, length: int | None = None):
    sequences = [seq for seq in sorted(directory.iterdir()) if seq.is_dir()]
    train_sequences = [seq for seq in sequences if seq.name not in VAL_SEQUENCES]
    val_sequences = [seq for seq in sequences if seq.name in VAL_SEQUENCES]

    # prepare train dataset
    train = process_split(output_directory / "train", train_sequences, length)
    train_txt = output_directory / "train.txt"
    with open(train_txt, "w") as fp:
        fp.write("\n".join(str(path) for path in train))

    # prepare val dataset
    val = process_split(output_directory / "test", val_sequences, length)
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
    parser.add_argument("--length", type=int, default=None)
    args = parser.parse_args()

    prepare_dataset(args.input_directory, args.output_directory, length=args.length)

"""
Read CSV labels and copy audio segments to their corresponding label folder.
No label (empty categoria) means vocalizacion.

CSV format:
  path     -> ../labeling_files/raw_specs/<species>/<recording>_<idx>.png
  categoria -> label string (empty = vocalizacion)

Source audio:  audio_files/raw_segm_audios/<species>/<recording>_<idx>.ogg
Destination:   audio_files/dataset/<split>/<species>/<recording>/<label>/<recording>_<idx>.ogg

The split (train/val/test) is determined by scanning the dataset folder,
since it was assigned randomly by generate_empty_folders.py.
"""

import os
import csv
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------- Configuration --------------------
BASE_DIR        = Path(__file__).resolve().parents[1]
CSV_PATH        = BASE_DIR / "labeling_files" / "paths_imgs_spec_FULL.csv"
SEGM_AUDIO_DIR  = BASE_DIR / "audio_files" / "raw_segm_audios"
DATASET_DIR     = BASE_DIR / "audio_files" / "dataset"
DEFAULT_LABEL   = "vocalizacion"
SPLITS          = ["train", "val", "test"]
NUM_THREADS     = 8    # number of threads used for parallel file copying
# -------------------------------------------------------


def build_split_index(dataset_dir: Path):
    """
    Return a dict  "species/recording" -> "train"|"val"|"test"
    by scanning the already-created folder structure.
    """
    index: dict[str, str] = {}
    for split in SPLITS:
        split_path = dataset_dir / split
        if not split_path.exists():
            continue
        for species_dir in split_path.iterdir():
            if not species_dir.is_dir():
                continue
            for rec_dir in species_dir.iterdir():
                if not rec_dir.is_dir():
                    continue
                key = f"{species_dir.name}/{rec_dir.name}"
                index[key] = split
    return index


def parse_row(path_str: str):
    """
    Extract (species, recording, segment_stem) from a CSV path like:
      ../labeling_files/raw_specs/amakin1/XC113758_0.png
    Returns ('amakin1', 'XC113758', 'XC113758_0').
    """
    p = Path(path_str)
    species = p.parent.name                     # e.g. amakin1
    stem = p.stem                               # e.g. XC113758_0
    # Split off the last _<index> to get the recording id
    parts = stem.rsplit("_", 1)
    recording = parts[0] if len(parts) == 2 else stem
    return species, recording, stem


def _copy_one(src: Path, dest_dir: Path):
    """
    Copy a single file to dest_dir. Returns 'copied' or 'skipped'.
    mkdir is safe to call concurrently thanks to exist_ok=True.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        return "skipped"
    shutil.copy2(src, dest)
    return "copied"


def copy_segments(csv_path: Path, segm_audio_dir: Path,
                  dataset_dir: Path, default_label: str,
                  num_threads: int = NUM_THREADS):

    split_index = build_split_index(dataset_dir)
    print(f"Loaded split index with {len(split_index)} recording entries.")

    # --- build task list sequentially (CSV is a single stream) ---
    tasks: list[tuple[Path, Path]] = []   # (src, dest_dir)
    missing_split = missing_src = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_str = row.get("path", "").strip()
            label    = row.get("categoria", "").strip() or default_label

            if not path_str:
                continue

            species, recording, stem = parse_row(path_str)
            key = f"{species}/{recording}"

            split = split_index.get(key)
            if split is None:
                print(f"[WARN] No split folder found for: {key}")
                missing_split += 1
                continue

            src = segm_audio_dir / species / f"{stem}.ogg"
            if not src.exists():
                print(f"[WARN] Source audio not found: {src}")
                missing_src += 1
                continue

            dest_dir = dataset_dir / split / species / recording / label
            tasks.append((src, dest_dir))

    print(f"Dispatching {len(tasks)} copy tasks across {num_threads} threads...")

    # --- parallel copy ---
    copied = skipped = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(_copy_one, src, dest_dir): (src, dest_dir)
                   for src, dest_dir in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result == "copied":
                copied += 1
            else:
                skipped += 1

    print(f"\nDone. Copied: {copied} | Already existed: {skipped} | "
          f"Missing split: {missing_split} | Missing source: {missing_src}")


if __name__ == "__main__":
    copy_segments(CSV_PATH, SEGM_AUDIO_DIR, DATASET_DIR, DEFAULT_LABEL)

## Running
## located in Clef-V2 folder
## run: python3 Files_and_Folders_Management/move_audios.py

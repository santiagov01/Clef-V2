import os
import shutil
import random

# ---------------- Configuration ----------------
SOURCE_DIR = R"D:\birdclef\birdclef-2025\train_audio"  # folder with species directories
OUTPUT_DIR = R"D:\birdclef\dataset"        # folder where train/test/val will be created

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

# Subfolders to create under each recording
LABEL_FOLDERS = ["voz", "silence", "vocalizacion", "trash"]

# ------------------------------------------------

def create_dir(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def split_list(lst, ratios):
    """Split a list into train, val, and test based on ratios."""
    random.shuffle(lst)
    n = len(lst)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    train = lst[:n_train]
    val = lst[n_train:n_train + n_val]
    test = lst[n_train + n_val:]
    return train, val, test

def create_structure():
    # Get species list
    species_list = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    for specie in species_list:
        specie_path = os.path.join(SOURCE_DIR, specie)
        recordings = [f for f in os.listdir(specie_path) if f.endswith(".ogg")]

        # Split the recordings
        train_set, val_set, test_set = split_list(recordings, SPLIT_RATIOS)

        # Create directory structure
        for split_name, split_data in zip(["train", "val", "test"], [train_set, val_set, test_set]):
            for recording in split_data:
                recording_name = os.path.splitext(recording)[0]
                base_path = os.path.join(OUTPUT_DIR, split_name, specie, recording_name)
                for label_folder in LABEL_FOLDERS:
                    create_dir(os.path.join(base_path, label_folder))

    print("Folder structure created successfully!")

if __name__ == "__main__":
    create_structure()

"""
train_binary_species.py
-----------------------
Trains one binary classifier (species vs. rest) per bird species found in the
dataset folder.  Only the 'vocalizacion' sub-folders are used.

Expected folder layout:
    <split>/
        <species>/
            <recording_id>/
                vocalizacion/   <- Only this folder is used
                voz/
                silence/
                trash/

Files inside 'vocalizacion' can be:
  - Images (.png, .jpg, .jpeg)  -> loaded directly
  - Audio  (.wav, .mp3, .flac, .ogg) -> mel-spectrogram generated on-the-fly

Usage examples:
    python train_binary_species.py --data_dir data/labeled --output outputs_binary
    python train_binary_species.py --data_dir data/labeled --output outputs_binary \
        --model efficientnet_b0 --trials 20 --epochs_max 30 --workers 4
"""

import os
import sys
import json
import logging
import argparse
import datetime
import time
import random
import platform
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for multiprocessing)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated image files

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress Optuna verbosity

import librosa
import librosa.display
import soundfile as sf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCALIZACION_FOLDER = "vocalizacion"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SPLITS = ("train", "validation", "test")
SEED = 42


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def build_logger(output_dir: str, species: str) -> logging.Logger:
    """
    Create a per-species file + console logger.
    Returns a standard Python Logger instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(species)
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers if function is called twice in the same process
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file handler
    log_path = os.path.join(output_dir, "train.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def log_hardware(logger: logging.Logger) -> None:
    """Log basic hardware/environment information."""
    logger.info("Platform: %s %s", platform.system(), platform.release())
    logger.info("Processor: %s", platform.processor())
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))


# ---------------------------------------------------------------------------
# Spectrogram generation
# ---------------------------------------------------------------------------

def audio_to_spectrogram_image(audio_path: str,
                                sr: int = 32000,
                                n_mels: int = 128,
                                hop_length: int = 512,
                                fmax: int = 16000) -> Image.Image:
    """
    Load an audio file and return a mel-spectrogram as a PIL RGB Image.
    The image is normalised to [0, 255].
    """
    y, actual_sr = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=actual_sr, n_mels=n_mels,
                                         hop_length=hop_length, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # normalise to [0, 255]
    mel_norm = ((mel_db - mel_db.min()) /
                (mel_db.max() - mel_db.min() + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(mel_norm).convert("RGB")
    return img


def load_image(path: str) -> Image.Image:
    """Load a file as a PIL RGB image, generating spectrogram if it is audio."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return Image.open(path).convert("RGB")
    elif ext in AUDIO_EXTENSIONS:
        return audio_to_spectrogram_image(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def collect_vocalizacion_files(split_dir: str, species: str) -> list[str]:
    """
    Walk <split_dir>/<species>/<recording>/<VOCALIZACION_FOLDER>/ and return
    a list of all valid image/audio file paths.

    Args:
        - split_dir: path to the split folder (e.g. "data/labeled/train")
        - species: name of the species (e.g. "Carduelis carduelis")
    Returns:
        - list of file paths (strings)
    """
    files = []
    species_dir = os.path.join(split_dir, species)
    if not os.path.isdir(species_dir):
        return files
    for recording in sorted(os.listdir(species_dir)):
        # TODO: Check if we only use the VOCALIZACION_FOLDER and ignore the rest (voz, silence, trash)
        #Maybe in negative label we can use the rest of the folders to increase negative samples
        voc_dir = os.path.join(species_dir, recording, VOCALIZACION_FOLDER) 
        if not os.path.isdir(voc_dir):
            continue
        for fname in sorted(os.listdir(voc_dir)):
            ext = Path(fname).suffix.lower()
            if ext in IMAGE_EXTENSIONS | AUDIO_EXTENSIONS:
                files.append(os.path.join(voc_dir, fname))
    return files


def discover_species(data_dir: str) -> list[str]:
    """Return sorted list of species found in the 'train' split folder."""
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    return sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )


def build_file_lists(data_dir: str, target_species: str, all_species: list[str],
                     split: str, neg_ratio: int) -> tuple[list[str], list[int]]:
    """
    Build a list of (file_path, binary_label) for one split.
    label = 1  -> target species
    label = 0  -> any other species (negatives are sub-sampled to neg_ratio * len(positives))
    """
    split_dir = os.path.join(data_dir, split)

    positives = collect_vocalizacion_files(split_dir, target_species)
    negatives = []
    # ITERATE OVER ALL THE OTHER SPECIES TO COLLECT NEGATIVE SAMPLES
    for sp in all_species:
        if sp != target_species:
    # TODO: Consider using the 'voz', 'silence', 'trash' folders from this and other species as additional negatives to increase negative samples, instead of only using 'vocalizacion' from other species. This could help balance the dataset if there are few positives.
            negatives.extend(collect_vocalizacion_files(split_dir, sp))

    # sub-sample negatives so that negative : positive <= neg_ratio
    max_neg = neg_ratio * max(len(positives), 1)
    if len(negatives) > max_neg:
        rng = random.Random(SEED)
        negatives = rng.sample(negatives, max_neg)

    paths = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    return paths, labels


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BinarySpeciesDataset(Dataset):
    """
    Binary dataset for a single species vs. rest.
    Supports both image files and audio files (mel-spectrogram on-the-fly).
    """

    def __init__(self, paths: list[str], labels: list[int],
                 transform=None):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = load_image(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transforms(image_size: int = 224, augment: bool = True):
    """Return train/val torchvision transform pipelines."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ]) if augment else transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to balance positive/negative classes.
     Asigna pesos a cada muestra, permitiendo que las clases minoritarias se
     muestren con mayor frecuencia. Se configura con pesos, número total de muestras
     y muestreo con reemplazo
    """
    counts = np.bincount(labels)
    weights_per_class = 1.0 / (counts + 1e-8)
    sample_weights = [weights_per_class[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = [
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnet_b4",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "resnet50",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
]


def get_model(model_name: str, num_classes: int = 2,
              pretrained: bool = True) -> nn.Module:
    """
    Build a pretrained CNN model with the classifier head replaced for
    binary classification (num_classes=2).
    """
    weights_arg = "DEFAULT" if pretrained else None

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "regnet_y_400mf":
        model = models.regnet_y_400mf(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "regnet_y_800mf":
        model = models.regnet_y_800mf(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "regnet_y_1_6gf":
        model = models.regnet_y_1_6gf(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from: {SUPPORTED_MODELS}")
    return model


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).long()
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device
             ) -> tuple[float, list, list]:
    """Evaluate model, return (avg_loss, y_true, y_pred)."""
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).long()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds)
    return total_loss / max(len(loader), 1), y_true, y_pred


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute binary classification metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial,
                     train_paths: list[str], train_labels: list[int],
                     val_paths: list[str], val_labels: list[int],
                     device: torch.device,
                     args) -> float:
    """
    Optuna objective: train a model with suggested hyperparameters and
    return validation F1 score (maximise).
    """
    # TODO. FIXME: Which hyperparameters to suggest?  We can start with a few and then expand
    # --- suggest hyperparameters ---
    model_name = trial.suggest_categorical("model_name", args.architectures)
    lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    epochs     = trial.suggest_int("epochs", args.epochs_min, args.epochs_max)
    image_size = trial.suggest_categorical("image_size", [128, 224])

    train_tf, val_tf = build_transforms(image_size=image_size, augment=True)

    train_ds = BinarySpeciesDataset(train_paths, train_labels, train_tf)
    val_ds   = BinarySpeciesDataset(val_paths, val_labels, val_tf)
    # TODO: FIXME: Is it necessary to use this weighted sampler for Data Imbalance, 
    # even tough it's suppose to be balanced when labeling positive and negative segments
    sampler  = make_weighted_sampler(train_labels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = get_model(model_name, num_classes=2, pretrained=True).to(device)

    # TODO: Check if it's necessary to add class_weghts to deal with data imabalance
    # class weights to handle remaining imbalance
    counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        [1.0 / (c + 1e-8) for c in counts], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_dl, optimizer, criterion, device)
        scheduler.step()
        _, y_true, y_pred = evaluate(model, val_dl, criterion, device)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    # TODO Which metric use to find the best model? 
        if f1 > best_f1:
            best_f1 = f1
        # pruning
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_f1


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_loss_plot(train_losses: list, val_losses: list,
                   val_epochs: list, out_path: str) -> None:
    """Save a train/val loss curve plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             label="Train Loss", marker="o", linewidth=1.5)
    plt.plot(val_epochs, val_losses,
             label="Val Loss", marker="x", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_confusion_matrix(y_true: list, y_pred: list,
                          out_path: str, title: str = "Confusion Matrix") -> None:
    """Save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Other", "Target"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Final model training (with best hyperparameters)
# ---------------------------------------------------------------------------

def train_final_model(species: str,
                      train_paths, train_labels,
                      val_paths, val_labels,
                      test_paths, test_labels,
                      best_params: dict,
                      out_dir: str,
                      device: torch.device,
                      args,
                      logger: logging.Logger) -> dict:
    """
    Train the final model with the best hyperparameters found by Optuna.
    Logs and saves all artifacts.  Returns a metrics dict.
    """
    model_name   = best_params["model_name"]
    lr           = best_params["lr"]
    batch_size   = int(best_params["batch_size"])
    weight_decay = best_params["weight_decay"]
    epochs       = int(best_params["epochs"])
    image_size   = int(best_params["image_size"])

    logger.info("Final training | model=%s lr=%.2e bs=%d wd=%.2e "
                "epochs=%d img=%d", model_name, lr, batch_size,
                weight_decay, epochs, image_size)

    train_tf, val_tf = build_transforms(image_size=image_size, augment=True)

    train_ds = BinarySpeciesDataset(train_paths, train_labels, train_tf)
    val_ds   = BinarySpeciesDataset(val_paths, val_labels, val_tf)
    test_ds  = BinarySpeciesDataset(test_paths, test_labels, val_tf)

    sampler  = make_weighted_sampler(train_labels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = get_model(model_name, num_classes=2, pretrained=True).to(device)
    # TODO: Check if it's necessary to add class_weghts to deal with data imabalance
    counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        [1.0 / (c + 1e-8) for c in counts], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses, val_epochs = [], [], []
    best_val_f1   = -1.0
    best_ckpt     = os.path.join(out_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        t_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        train_losses.append(t_loss)
        scheduler.step()

        # validate every epoch
        v_loss, y_true_v, y_pred_v = evaluate(model, val_dl, criterion, device)
        val_losses.append(v_loss)
        val_epochs.append(epoch)
        val_f1 = f1_score(y_true_v, y_pred_v, zero_division=0)
        val_acc = accuracy_score(y_true_v, y_pred_v)

        logger.info("Epoch %3d/%d | Train Loss %.4f | Val Loss %.4f | "
                    "Val Acc %.4f | Val F1 %.4f",
                    epoch, epochs, t_loss, v_loss, val_acc, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)

    # reload best checkpoint for test evaluation
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # test evaluation
    _, y_true_test, y_pred_test = evaluate(model, test_dl, criterion, device)
    test_metrics = compute_metrics(y_true_test, y_pred_test)
    val_metrics  = compute_metrics(y_true_v, y_pred_v)

    logger.info("Test results | Acc %.4f | P %.4f | R %.4f | F1 %.4f",
                test_metrics["accuracy"], test_metrics["precision"],
                test_metrics["recall"], test_metrics["f1"])
    logger.info("\n%s",
                classification_report(y_true_test, y_pred_test,
                                      target_names=["Other", "Target"],
                                      zero_division=0))

    # --- save artifacts ---
    save_loss_plot(train_losses, val_losses, val_epochs,
                   os.path.join(out_dir, "loss_curve.png"))
    save_confusion_matrix(y_true_test, y_pred_test,
                          os.path.join(out_dir, "confusion_test.png"),
                          title=f"Test Confusion Matrix – {species}")

    # metrics JSON
    metrics_out = {
        "species":         species,
        "best_params":     best_params,
        "val_metrics":     val_metrics,
        "test_metrics":    test_metrics,
        "train_samples":   len(train_labels),
        "val_samples":     len(val_labels),
        "test_samples":    len(test_labels),
        "positives_train": int(sum(train_labels)),
        "positives_test":  int(sum(test_labels)),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    return metrics_out


# ---------------------------------------------------------------------------
# Per-species pipeline
# ---------------------------------------------------------------------------

def run_species(species: str, all_species: list[str], args) -> dict | None:
    """
    Full pipeline for a single species:
      1. Discover files
      2. Optuna hyperparameter search
      3. Final model training and evaluation
    Returns the metrics dict or None on failure.
    """
    set_seeds(SEED)

    # determine output directory for this species
    out_dir = os.path.join(args.output, species)
    os.makedirs(out_dir, exist_ok=True)

    logger = build_logger(out_dir, species)
    logger.info("=== Starting pipeline for species: %s ===", species)
    log_hardware(logger)

    t_start = time.time()

    # --- data discovery ---
    train_paths, train_labels = build_file_lists(
        args.data_dir, species, all_species, "train", args.neg_ratio)
    
    val_paths, val_labels = build_file_lists(
        args.data_dir, species, all_species, "validation", args.neg_ratio)
    test_paths, test_labels = build_file_lists(
        args.data_dir, species, all_species, "test", args.neg_ratio)

    logger.info("Train: %d samples (%d positive)",
                len(train_paths), sum(train_labels))
    logger.info("Val:   %d samples (%d positive)",
                len(val_paths), sum(val_labels))
    logger.info("Test:  %d samples (%d positive)",
                len(test_paths), sum(test_labels))

    if sum(train_labels) < args.min_samples:
        logger.warning("Skipping species '%s': only %d positive samples "
                       "(min required: %d).",
                       species, sum(train_labels), args.min_samples)
        return None

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Optuna search ---
    logger.info("Starting Optuna study with %d trials...", args.trials)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study   = optuna.create_study(direction="maximize",
                                  pruner=pruner,
                                  sampler=sampler)
    study.optimize(
        lambda trial: optuna_objective(
            trial, train_paths, train_labels,
            val_paths, val_labels, device, args),
        n_trials=args.trials,
        show_progress_bar=False,
    )

    logger.info("Best trial: F1=%.4f | params=%s",
                study.best_value, study.best_trial.params)

    # save Optuna results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)

    # --- final training ---
    metrics = train_final_model(
        species,
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        best_params=study.best_trial.params,
        out_dir=out_dir,
        device=device,
        args=args,
        logger=logger,
    )

    elapsed = time.time() - t_start
    metrics["elapsed_seconds"] = round(elapsed, 2)
    logger.info("Pipeline complete for '%s' in %.1f s", species, elapsed)

    # update metrics JSON with elapsed time
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Parallel execution wrapper
# ---------------------------------------------------------------------------

def run_species_safe(species: str, all_species: list[str], args) -> dict | None:
    """
    Wrapper around run_species that catches exceptions so a failure in one
    species does not abort the entire run.
    """
    try:
        return run_species(species, all_species, args)
    except Exception as exc:
        # write to a minimal error file so the failure is recorded
        err_dir = os.path.join(args.output, species)
        os.makedirs(err_dir, exist_ok=True)
        with open(os.path.join(err_dir, "error.log"), "w") as f:
            import traceback
            f.write(traceback.format_exc())
        print(f"[ERROR] Species '{species}' failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def save_summary(all_metrics: list[dict], output_dir: str) -> None:
    """Aggregate metrics for all species and save a summary CSV."""
    rows = []
    for m in all_metrics:
        if m is None:
            continue
        row = {
            "species":          m["species"],
            "train_positives":  m["positives_train"],
            "test_positives":   m["positives_test"],
            "val_accuracy":     m["val_metrics"]["accuracy"],
            "val_precision":    m["val_metrics"]["precision"],
            "val_recall":       m["val_metrics"]["recall"],
            "val_f1":           m["val_metrics"]["f1"],
            "test_accuracy":    m["test_metrics"]["accuracy"],
            "test_precision":   m["test_metrics"]["precision"],
            "test_recall":      m["test_metrics"]["recall"],
            "test_f1":          m["test_metrics"]["f1"],
            "elapsed_s":        m.get("elapsed_seconds", -1),
            "model_name":       m["best_params"].get("model_name", ""),
        }
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows).sort_values("species")
        path = os.path.join(output_dir, "summary.csv")
        df.to_csv(path, index=False)
        print(f"Summary saved to: {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one binary CNN classifier per bird species "
                    "using 'vocalizacion' spectrograms."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing 'train', 'validation', 'test' splits."
    )
    parser.add_argument(
        "--output", type=str, default="outputs_binary",
        help="Root output directory (default: outputs_binary)."
    )
    parser.add_argument(
        "--architectures", type=str, nargs="+",
        default=["efficientnet_b0", "efficientnet_b3",
                 "regnet_y_400mf", "regnet_y_800mf",
                 "mobilenet_v3_small"],
        help="CNN architectures to include in the Optuna search space."
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of Optuna trials per species (default: 20)."
    )
    parser.add_argument(
        "--epochs_min", type=int, default=5,
        help="Minimum epochs in Optuna search (default: 5)."
    )
    parser.add_argument(
        "--epochs_max", type=int, default=30,
        help="Maximum epochs in Optuna search (default: 30)."
    )
    parser.add_argument(
        "--neg_ratio", type=int, default=3,
        help="Maximum ratio of negatives to positives (default: 3)."
    )
    parser.add_argument(
        "--min_samples", type=int, default=10,
        help="Minimum positive training samples to train a species "
             "(default: 10)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader worker processes (default: 2)."
    )
    parser.add_argument(
        "--parallel_species", type=int, default=1,
        help="Number of species to train in parallel. "
             "Use 1 for sequential (recommended when using GPU). "
             "Each parallel process initialises its own CUDA context."
    )
    parser.add_argument(
        "--species_filter", type=str, nargs="*", default=None,
        help="Train only these species (default: all discovered species)."
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Global random seed (default: 42)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seeds(args.seed)

    # global logger for the run coordinator
    os.makedirs(args.output, exist_ok=True)
    run_logger = build_logger(args.output, "run_coordinator")
    run_logger.info("Run started at %s",
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    run_logger.info("Data dir: %s", args.data_dir)
    run_logger.info("Output dir: %s", args.output)

    all_species = discover_species(args.data_dir)
    run_logger.info("Discovered %d species: %s", len(all_species), all_species)

    # filter if requested
    if args.species_filter:
        unknown = set(args.species_filter) - set(all_species)
        if unknown:
            run_logger.warning("Unknown species in filter: %s", unknown)
        all_species_to_run = [s for s in all_species
                              if s in set(args.species_filter)]
    else:
        all_species_to_run = all_species

    run_logger.info("Species to train: %d", len(all_species_to_run))

    if args.parallel_species > 1 and torch.cuda.is_available():
        run_logger.warning(
            "parallel_species=%d with CUDA: each process will compete for "
            "the same GPU. Consider parallel_species=1 for GPU training.",
            args.parallel_species
        )

    all_metrics = []

    if args.parallel_species <= 1:
        # --- sequential ---
        for species in all_species_to_run:
            result = run_species_safe(species, all_species, args)
            all_metrics.append(result)
    else:
        # --- parallel (ProcessPoolExecutor) ---
        # Note: each spawned process loads its own model and data independently.
        futures_map = {}
        with ProcessPoolExecutor(max_workers=args.parallel_species) as pool:
            for species in all_species_to_run:
                future = pool.submit(run_species_safe, species, all_species, args)
                futures_map[future] = species

            for future in as_completed(futures_map):
                sp = futures_map[future]
                try:
                    result = future.result()
                    all_metrics.append(result)
                    run_logger.info("Finished: %s", sp)
                except Exception as exc:
                    run_logger.error("Species '%s' raised: %s", sp, exc)
                    all_metrics.append(None)

    # --- save summary ---
    save_summary([m for m in all_metrics if m is not None], args.output)
    run_logger.info("All species finished. Summary written to %s", args.output)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Usage examples
# ---------------------------------------------------------------------------
# Sequential (GPU recommended):
#   python train_binary_species.py \
#       --data_dir data/labeled \
#       --output outputs_binary \
#       --architectures efficientnet_b0 efficientnet_b3 regnet_y_400mf \
#       --trials 20 --epochs_min 5 --epochs_max 30
#
# Parallel (CPU-only, 4 species at once):
#   python train_binary_species.py \
#       --data_dir data/labeled \
#       --output outputs_binary \
#       --parallel_species 4 \
#       --trials 10 --epochs_max 15
#
# Train only specific species:
#   python train_binary_species.py \
#       --data_dir data/labeled \
#       --output outputs_binary \
#       --species_filter species_A species_B \
#       --trials 15

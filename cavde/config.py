from dataclasses import dataclass
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class Config:

    # =========================
    # PROJECT
    # =========================

    project_name: str = "CAVDE"

    version: str = "2.0.0-alpha"

    # =========================
    # DIRECTORY
    # =========================

    dataset_dir: Path = ROOT / "datasets"

    checkpoint_dir: Path = ROOT / "checkpoints"

    log_dir: Path = ROOT / "logs"

    export_dir: Path = ROOT / "exports"

    cache_dir: Path = ROOT / "cache"

    # =========================
    # MODEL
    # =========================

    d_model: int = 256

    num_heads: int = 8

    num_layers: int = 6

    feed_forward: int = 1024

    dropout: float = 0.1

    max_sequence_length: int = 256

    # =========================
    # TRAINING
    # =========================

    batch_size: int = 32

    learning_rate: float = 3e-4

    epochs: int = 1000

    weight_decay: float = 0.01

    gradient_clip: float = 1.0

    # =========================
    # TOKENIZER
    # =========================

    min_word_frequency: int = 2

    vocab_size: int = 50000

    # =========================
    # DEVICE
    # =========================

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =========================
    # RANDOM
    # =========================

    seed: int = 42

    # =========================
    # SAVE
    # =========================

    save_every_epoch: bool = True

    save_best_only: bool = True

    # =========================
    # SCHEDULER
    # =========================

    scheduler = "step"

    step_size = 5

    gamma = 0.5

    t_max = 10


CONFIG = Config()

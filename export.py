from __future__ import annotations

from pathlib import Path
import shutil


CHECKPOINT = Path("checkpoints/latest.pt")

EXPORT_DIR = Path("models")

EXPORT_NAME = "cavde-base.pt"


def export():

    if not CHECKPOINT.exists():

        print("No checkpoint found.")

        return

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = EXPORT_DIR / EXPORT_NAME

    shutil.copy2(
        CHECKPOINT,
        destination,
    )

    print()

    print("=" * 50)
    print("          CAVDE EXPORT")
    print("=" * 50)

    print()

    print("Source      :", CHECKPOINT)

    print("Destination :", destination)

    print()

    print("Export complete.")
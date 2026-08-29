import logging

from pathlib import Path


def create_logger(

    log_directory: Path,

    name: str = "CAVDE",

):

    log_directory.mkdir(

        parents=True,

        exist_ok=True,

    )

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(

        "[%(asctime)s] %(levelname)s : %(message)s"

    )

    file_handler = logging.FileHandler(

        log_directory / "train.log",

        encoding="utf-8",

    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
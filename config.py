import datetime
import logging
from pathlib import Path

LOGS_FOLDER_PATH = Path("logs")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(
    filename=f"{LOGS_FOLDER_PATH / datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H-%M-%S")}.log",
    mode="w",
)

stream_handler_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
file_handler_formatter = logging.Formatter(
    fmt="%(asctime)s | %(filename)s | %(levelname)s | %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)

stream_handler.setFormatter(stream_handler_formatter)
file_handler.setFormatter(file_handler_formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

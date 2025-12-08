from pathlib import Path
import logging

class Logger(logging.Logger):
    def __init__(
            self,
            name: str,
            logpath: Path = Path('src/logging/logfile.log')
        ) -> None:
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.propagate = False

        console_handler = ConsoleHandler()
        self.addHandler(console_handler)

        file_handler = CustomFileHandler(logpath=logpath)
        self.addHandler(file_handler)

class ConsoleHandler(logging.StreamHandler):
    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s",
            datefmt=r"%m/%d/%Y %H:%M:%S",
        )
        self.setFormatter(formatter)
        self.setLevel(level)

class CustomFileHandler(logging.FileHandler):
    def __init__(
            self,
            logpath: Path
        ):
        super().__init__(logpath, encoding="UTF-8")
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s",
            datefmt=r"%m/%d/%Y %H:%M:%S",
        )
        self.setFormatter(formatter)
        self.setLevel(logging.INFO)

logging.setLoggerClass(Logger)

logger = logging.getLogger(__name__)
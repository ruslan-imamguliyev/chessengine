from pathlib import Path
import torch


PATH_TO_CONFIG_FILE = Path('src/config/config.yaml')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIBTORCH_PATH = r"C:\libs\libtorch\lib"
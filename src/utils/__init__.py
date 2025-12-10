from yaml import safe_load
from pathlib import Path
from typing import Union
from box.exceptions import BoxValueError
from src.logging import logger

def read_yaml(
        path_to_yaml: Union[str, Path]
    ) -> dict:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = safe_load(yaml_file)
            return content
    except BoxValueError as be:
        logger.exception(be)
        raise ValueError("yaml file is empty")
    except Exception as e:
        logger.exception(e)
        raise e
from yaml import safe_load
from pathlib import Path
from typing import Union, List
from box.exceptions import BoxValueError
import numpy as np
from src.logging import logger
import os


def fen_to_tensor(
        fen: str
    ) -> np.array:

    """
    Convert a FEN string into a numpy array of shape (18, 8, 8).
    
    :param fen: FEN notation of a position
    :type fen: str
    :return: Numpy array
    :rtype: Any
    """
    x = np.zeros((18, 8, 8), dtype=np.uint8)

    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }

    board, side, castling, ep, _, _ = fen.split()

    rows = board.split('/')
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                x[piece_to_plane[ch], r, c] = 1
                c += 1

    if side == 'w':
        x[12, :, :] = 1
    if 'K' in castling:
        x[13, :, :] = 1
    if 'Q' in castling:
        x[14, :, :] = 1
    if 'k' in castling:
        x[15, :, :] = 1
    if 'q' in castling:
        x[16, :, :] = 1

    if ep != '-':
        file = ord(ep[0]) - ord('a')
        rank = 8 - int(ep[1])
        x[17, rank, file] = 1

    return x


def create_paths(
        path: Union[Union[str, Path], Union[List[str], List[Path]]]
    ) -> None:
    """
    Makes paths if they don't exist.
    
    :param path: Either a string of the path or the Path object. May be a list of those.
    :type path: Union[Union[str, Path], Union[List[str], List[Path]]]
    """
    if type(path) in [str, Path]:
        path = [path]
    
    for p in path:
        if not os.path.exists(p):
            logger.info("Making path for the " + str(p))
            try:
                os.makedirs(p)
            except Exception as e:
                logger.exception("Couldn't make path: " + str(e))
                raise e
    
    


def read_yaml(
        path_to_yaml: Union[str, Path]
    ) -> dict:
    """
    Reads .yaml file and returns python dict.
    
    :param path_to_yaml: Path to the .yaml file
    :type path_to_yaml: Union[str, Path]
    :return: Python dictionary
    :rtype: dict
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = safe_load(yaml_file)
            return content
    except BoxValueError as be:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
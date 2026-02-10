from yaml import safe_load
from pathlib import Path
from typing import Union, List
from box.exceptions import BoxValueError
import numpy as np


def fen_to_tensor(
        fen: str
    ) -> np.array:

    """
    Convert a batch of FEN strings into a numpy array of shape (18, 8, 8).

    Args:
        fen (str): FEN notaion as a string

    Returns:
        np.array: numpy array of shape (18, 8, 8)
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




def read_yaml(
        path_to_yaml: Union[str, Path]
    ) -> dict:
    """
    Reads .yaml file and returns python dict.

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        dict: python dict
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = safe_load(yaml_file)
            return content
    except BoxValueError as be:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
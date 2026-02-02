from yaml import safe_load
from pathlib import Path
from typing import Union
from box.exceptions import BoxValueError
import torch


def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Convert a FEN string into a PyTorch tensor of shape (18, 8, 8).

    Args:
        fen (str): FEN notaion as a string

    Returns:
        torch.Tensor: PyTorch tensor of shape (18, 8, 8)
    """

    board, side, castling, ep, _, _ = fen.split()

    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)

    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }


    rows = board.split('/')
    for rank_idx, row in enumerate(rows):
        file_idx = 0
        for char in row:
            if char.isdigit():
                file_idx += int(char)
            else:
                plane = piece_to_plane[char]
                tensor[plane, rank_idx, file_idx] = 1.0
                file_idx += 1

    if side == 'w':
        tensor[12, :, :] = 1.0  # white to move

    if 'K' in castling:
        tensor[13, :, :] = 1.0
    if 'Q' in castling:
        tensor[14, :, :] = 1.0
    if 'k' in castling:
        tensor[15, :, :] = 1.0
    if 'q' in castling:
        tensor[16, :, :] = 1.0

    # en passant
    if ep != '-':
        file = ord(ep[0]) - ord('a')
        rank = 8 - int(ep[1])
        tensor[17, rank, file] = 1.0

    return tensor



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
from yaml import safe_load
from pathlib import Path
from typing import Union, List
from box.exceptions import BoxValueError
import torch

def batch_fen_to_tensor(
        fens: List[str]
    ) -> torch.Tensor:

    """
    Convert a batch of FEN strings into a PyTorch tensor of shape (batch_size, 18, 8, 8).

    Args:
        fen (str): FEN notaion as a string

    Returns:
        torch.Tensor: PyTorch tensor of shape (batch_size, 18, 8, 8)
    """

    batch_size = len(fens)
    x = torch.zeros((batch_size, 18, 8, 8), dtype=torch.float32)

    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }

    for i, fen in enumerate(fens):
        board, side, castling, ep, _, _ = fen.split()

        rows = board.split('/')
        for r, row in enumerate(rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    x[i, piece_to_plane[ch], r, c] = 1.0
                    c += 1

        if side == 'w':
            x[i, 12].fill_(1.0)

        if 'K' in castling:
            x[i, 13].fill_(1.0)
        if 'Q' in castling:
            x[i, 14].fill_(1.0)
        if 'k' in castling:
            x[i, 15].fill_(1.0)
        if 'q' in castling:
            x[i, 16].fill_(1.0)

        if ep != '-':
            file = ord(ep[0]) - ord('a')
            rank = 8 - int(ep[1])
            x[i, 17, rank, file] = 1.0

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
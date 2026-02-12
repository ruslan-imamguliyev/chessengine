from yaml import safe_load
from pathlib import Path
from typing import Union, List
from box.exceptions import BoxValueError
import numpy as np
from src.logging import logger
import matplotlib.pyplot as plt
import os
import torch


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in the model.
    
    :param model: pytorch model
    :type model: torch.nn.Module
    :return: num of parameters
    :rtype: int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def plot_planes(planes: np.array) -> None:
    """
    Plots tensor planes
    
    :param tensor: numpy array of shape (18, 8, 8)
    :type tensor: np.array
    """

    plane_names = [
        "white pawn",
        "white knight",
        "white bishop",
        "white rook",
        "white queen",
        "white king",
        "black pawn",
        "black knight",
        "black bishop",
        "black rook",
        "black queen",
        "black king",
        "side to move (white=1, black=0)",
        "white king-side castling",
        "white queen-side castling",
        "black king-side castling",
        "black queen-side castling",
        "en passant square"
    ]

    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(18):
        ax = axes[i]
        ax.imshow(planes[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(plane_names[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def tensor_to_fen(
        tensor: np.array
    ) -> str:

    """
    Convert a numpy array of shape (18, 8, 8) back into a FEN string.
    
    :param tensor: numpy array of shape (18, 8, 8)
    :type tensor: np.array
    :return: fen string
    :rtype: str
    """

    piece_map = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k',
    }

    board_rows = []

    for r in range(8):
        row_str = ""
        empty_count = 0

        for c in range(8):
            piece_char = None

            # check piece planes
            for plane in range(12):
                if tensor[plane, r, c] > 0:
                    piece_char = piece_map[plane]
                    break

            if piece_char:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += piece_char
            else:
                empty_count += 1

        if empty_count > 0:
            row_str += str(empty_count)

        board_rows.append(row_str)

    board_part = "/".join(board_rows)

    # side to move
    side = "w" if tensor[12].mean() > 0.5 else "b"

    # castling rights
    castling = ""
    if tensor[13].mean() > 0.5:
        castling += "K"
    if tensor[14].mean() > 0.5:
        castling += "Q"
    if tensor[15].mean() > 0.5:
        castling += "k"
    if tensor[16].mean() > 0.5:
        castling += "q"

    if castling == "":
        castling = "-"

    # en passant
    ep_square = "-"
    ep_plane = tensor[17]

    positions = np.argwhere(ep_plane > 0)
    if len(positions) == 1:
        r, c = positions[0]
        file = chr(ord('a') + c)
        rank = str(8 - r)
        ep_square = file + rank

    # default halfmove/fullmove (unknown)
    halfmove = "0"
    fullmove = "1"

    return f"{board_part} {side} {castling} {ep_square} {halfmove} {fullmove}"


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
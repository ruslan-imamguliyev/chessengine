import torch
import chess
import numpy as np
from typing import List
from src.utils import board_to_tensor
from src.constants import DEVICE


class ModelWrapper:
    def __init__(self, model, device=DEVICE):
        """
        Initialize evaluator with trained model.
        
        Args:
            model: Your trained ChessResNet model
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
        
    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a single position.
        
        Returns: Score from white's perspective in [-1, 1]
        """
        bitplanes = board_to_tensor(board)
        tensor = torch.from_numpy(bitplanes).float().unsqueeze(0).to(self.device)
        score = self.model(tensor).item()
        
        # Return from current player's perspective
        if board.turn == chess.BLACK:
            score = -score
        
        return score
    
    @torch.no_grad()
    def evaluate_batch(self, boards: List[chess.Board]) -> List[float]:
        """
        Evaluate multiple positions efficiently.
        
        Returns: List of scores from white's perspective
        """
        if not boards:
            return []
        
        # Convert all boards to bitplanes
        bitplanes_list = [board_to_tensor(board) for board in boards]
        bitplanes_array = np.stack(bitplanes_list)
        
        # Batch inference
        tensor = torch.from_numpy(bitplanes_array).float().to(self.device)
        scores = self.model(tensor).cpu().numpy().flatten()
        
        # Adjust for side to move
        for i, board in enumerate(boards):
            if board.turn == chess.BLACK:
                scores[i] = -scores[i]
        
        return scores.tolist()
    @torch.no_grad()
    def evaluate_child_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[float]:
        """
        Evaluate a list of legal moves from the same root board using batched inference.

        Returns scores from the current side-to-move perspective *after* each move.
        """
        if not moves:
            return []

        child_boards = []
        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            child_boards.append(child)

        return self.evaluate_batch(child_boards)

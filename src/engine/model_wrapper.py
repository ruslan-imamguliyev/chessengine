import torch
import chess
import numpy as np
from typing import List
from src.utils import fen_to_tensor
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
        bitplanes = fen_to_tensor(board.fen())
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
        bitplanes_list = [fen_to_tensor(board.fen()) for board in boards]
        bitplanes_array = np.stack(bitplanes_list)
        
        # Batch inference
        tensor = torch.from_numpy(bitplanes_array).float().to(self.device)
        scores = self.model(tensor).cpu().numpy().flatten()
        
        # Adjust for side to move
        for i, board in enumerate(boards):
            if board.turn == chess.BLACK:
                scores[i] = -scores[i]
        
        return scores.tolist()
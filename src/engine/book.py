import chess
import chess.polyglot
import random

class OpeningBook:
    def __init__(self, path):
        self.reader = chess.polyglot.open_reader(path)

    def get_move(self, board: chess.Board):
        try:
            entries = list(self.reader.find_all(board))
            if not entries:
                return None
            
            # weighted random selection
            moves = [e.move for e in entries]
            weights = [e.weight for e in entries]
            
            return random.choices(moves, weights=weights)[0]
        
        except IndexError:
            return None
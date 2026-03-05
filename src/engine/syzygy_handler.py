import chess
import chess.syzygy

class SyzygyHandler:
    def __init__(self, path):
        self.tablebase = chess.syzygy.open_tablebase(path)

    def probe_wdl(self, board: chess.Board):
        try:
            return self.tablebase.probe_wdl(board)
        except:
            return None

    def probe_dtz(self, board: chess.Board):
        try:
            return self.tablebase.probe_dtz(board)
        except:
            return None
    
    def available(self, board: chess.Board):
        return board.occupied.bit_count() <= 5
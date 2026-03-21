import chess
from src.config.manager import EngineConfig
from src.engine.manager import Engine
from src.constants import LIBTORCH_PATH, DEVICE
import os

os.add_dll_directory(LIBTORCH_PATH)

import cpp_engine.build.Release.chessengine_cpp as chessengine_cpp

__all__ = ["chessengine_cpp"]

class AlphaBetaEngineCPP(Engine):
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        evaltr = chessengine_cpp.TorchEvaluator(self.config.compiled_model_path, DEVICE)
        ab_cfg = chessengine_cpp.ABConfig()
        ab_cfg.depth = self.config.depth
        ab_cfg.time_limit_sec = self.config.time_limit
        self.ab = chessengine_cpp.AlphaBetaEngine(ab_cfg, evaltr)
    
    def get_best_move(self, board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(self.ab.get_best_move_uci(board.fen()))
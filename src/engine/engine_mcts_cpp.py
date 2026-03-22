import chess
from src.config.manager import EngineConfig
from src.engine.manager import Engine
from src.constants import LIBTORCH_PATH, DEVICE
import os

os.add_dll_directory(LIBTORCH_PATH)

import cpp_engine.build.Release.chessengine_cpp as chessengine_cpp

__all__ = ["chessengine_cpp"]

class MCTSEngineCPP(Engine):
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        evaltr = chessengine_cpp.TorchEvaluator(self.config.compiled_model_path, DEVICE)
        mcts_cfg = chessengine_cpp.MCTSConfig()
        mcts_cfg.num_simulations = self.config.num_simulations
        mcts_cfg.leaf_parallelism = self.config.leaf_parallelism
        self.mcts = chessengine_cpp.MCTSEngine(mcts_cfg, evaltr)
    
    def get_best_move(self, board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(self.mcts.get_best_move_uci(board.fen()))
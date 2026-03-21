from abc import ABC, abstractmethod
import chess
from src.engine.book import OpeningBook
from src.engine.syzygy_handler import SyzygyHandler
from src.config.manager import EngineConfig, ConfigurationManager
from src.logging import logger


class Engine(ABC):
    def __init__(self, config: EngineConfig):
        super().__init__()
        self.config = config
        self.book = OpeningBook(self.config.book_path)
        self.syzygy = SyzygyHandler(self.config.syzygy_path)

    @abstractmethod
    def get_best_move(self, board: chess.Board) -> chess.Move:
        pass
    
    def play(self, board: chess.Board) -> chess.Move:
        logger.info(f"Searching position: {board.fen()}")

        book_move = self.book.get_move(board)

        if book_move is not None:
            logger.info(f"Book move found: {book_move.uci()}")
            return book_move
        
        if self.syzygy and self.syzygy.available(board):
            best_move = None
            best_dtz = float("inf")
            
            for move in board.legal_moves:
                board.push(move)
                dtz = -self.syzygy.probe_dtz(board)
                wdl = self.syzygy.probe_wdl(board)
                board.pop()
                if wdl < 0 and dtz < best_dtz:
                    best_dtz = dtz
                    best_move = move
                
            
            if not best_move:
                return next(iter(board.legal_moves))
            
            return best_move
        
        return self.get_best_move(board)


class EngineManager:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_engine_config()
    
    def get_engine(self) -> Engine:
        match self.config.type:
            case "alphabeta":
                if self.config.use_cpp_backend:
                    from src.engine.engine_ab_cpp import AlphaBetaEngineCPP
                    return AlphaBetaEngineCPP(self.config)
                else:
                    from src.engine.engine_ab import AlphaBetaEngine
                    return AlphaBetaEngine(self.config)
            case "mcts":
                if self.config.use_cpp_backend:
                    from src.engine.engine_mcts_cpp import MCTSEngineCPP
                    return MCTSEngineCPP(self.config)
                else:
                    from src.engine.engine_mcts import MCTSEngine
                    return MCTSEngine(self.config)
            case _:
                raise ValueError(f"Unknown engine type: {self.config.type}")
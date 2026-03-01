from typing import Optional, List, Tuple
import time
import chess
import chess.polyglot
from src.engine.tt import TranspositionTable
from src.engine.model_wrapper import ModelWrapper
from src.config.manager import ConfigurationManager
from src.models import ModelManager
from src.logging import logger


class ChessEngine:
    """
    Chess engine using Alpha-Beta search with neural network evaluation.
    """
    
    # Piece values for move ordering (centipawns)
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_engine_config()

        self.evaluator = ModelWrapper(ModelManager(config=config).load_model())
        self.tt = TranspositionTable(self.config.tt_size_mb)
        self.nodes_searched = 0
        self.tt_hits = 0
        self.stop_search = False
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move], 
                    tt_best_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """
        Order moves for better alpha-beta pruning.
        
        Good move ordering dramatically improves pruning efficiency.
        """
        def move_score(move):
            score = 0
            
            # TT best move first
            if tt_best_move and move == tt_best_move:
                return 1000000
            
            # Captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
                    score += 10 * self.PIECE_VALUES.get(victim.piece_type, 0)
                    score -= self.PIECE_VALUES.get(attacker.piece_type, 0)
            
            # Promotions
            if move.promotion:
                score += self.PIECE_VALUES.get(move.promotion, 0)
            
            # Checks
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, 
                         max_depth: int = 4) -> float:
        """
        Quiescence search to avoid horizon effect.
        Only searches captures and checks to reach quiet positions.
        """
        if max_depth == 0:
            return self.evaluator.evaluate(board)
        
        # Stand pat evaluation
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Generate only tactical moves (captures, checks)
        moves = list(board.legal_moves)
        tactical_moves = [m for m in moves if board.is_capture(m) or board.gives_check(m)]
        
        if not tactical_moves:
            return stand_pat
        
        for move in self.order_moves(board, tactical_moves):
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, max_depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float,
                   root_call: bool = False) -> Tuple[float, Optional[chess.Move]]:
        """
        Alpha-Beta search with transposition table and move ordering.
        
        Returns: (score, best_move)
        """
        self.nodes_searched += 1
        
        if self.stop_search:
            return 0.0, None
        
        # Check transposition table
        zobrist = chess.polyglot.zobrist_hash(board)
        tt_score, tt_move = self.tt.probe(zobrist, depth, alpha, beta)
        if tt_score is not None and not root_call:
            self.tt_hits += 1
            return tt_score, tt_move
        
        # Terminal nodes
        if board.is_checkmate():
            return -1.0, None
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0, None
        if board.can_claim_draw():
            return 0.0, None
        
        # Leaf nodes - use quiescence search
        if depth == 0:
            score = self.quiescence_search(board, alpha, beta)
            return score, None
        
        # Generate and order moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        ordered_moves = self.order_moves(board, legal_moves, tt_move)
        
        best_move = ordered_moves[0]
        best_score = -float('inf')
        flag = TranspositionTable.UPPERBOUND
        
        for move in ordered_moves:
            board.push(move)
            score, _ = self.alpha_beta(board, depth - 1, -beta, -alpha)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            if score > alpha:
                alpha = score
                flag = TranspositionTable.EXACT
            
            if alpha >= beta:
                flag = TranspositionTable.LOWERBOUND
                break
        
        # Store in transposition table
        self.tt.store(zobrist, depth, best_score, best_move, flag)
        
        return best_score, best_move
    
    def iterative_deepening(self, board: chess.Board, max_depth: int = 6,
                           time_limit: Optional[float] = None) -> Tuple[chess.Move, float, int]:
        """
        Iterative deepening search.
        Searches depth 1, 2, 3, ... up to max_depth.
        Can be stopped early if time runs out.
        
        Returns: (best_move, score, depth_reached)
        """
        start_time = time.time()
        best_move = None
        best_score = 0.0
        depth_reached = 0
        
        self.nodes_searched = 0
        self.tt_hits = 0
        self.stop_search = False
        
        for depth in range(1, max_depth + 1):
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            score, move = self.alpha_beta(board, depth, -float('inf'), float('inf'), root_call=True)
            
            if move is not None:
                best_move = move
                best_score = score
                depth_reached = depth
                
                elapsed = time.time() - start_time
                nps = self.nodes_searched / elapsed if elapsed > 0 else 0
                
                logger.info(f"Depth {depth}: {move.uci()} | "
                      f"Score: {score:.4f} | "
                      f"Nodes: {self.nodes_searched:,} | "
                      f"NPS: {nps:,.0f} | "
                      f"TT hits: {self.tt_hits:,}")
        
        return best_move, best_score, depth_reached
    
    def get_best_move(self, board: chess.Board) -> chess.Move:
        """
        Get the best move for the current position.
        
        Args:
            board: Current position
            depth: Maximum search depth
            time_limit: Time limit in seconds (optional)
        
        Returns: Best move
        """
        
        logger.info(f"Searching position: {board.fen()}")
        
        start_time = time.time()
        best_move, score, depth_reached = self.iterative_deepening(
            board, max_depth=self.config.depth, time_limit=self.config.time_limit
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Best move: {best_move.uci() if best_move else 'None'}")
        logger.info(f"Score: {score:.4f}")
        logger.info(f"Depth reached: {depth_reached}")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"Nodes: {self.nodes_searched:,}")
        logger.info(f"NPS: {self.nodes_searched/elapsed:,.0f}" if elapsed > 0 else "NPS: N/A")
        
        return best_move
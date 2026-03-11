from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import os
import time

import chess
import chess.polyglot

from src.engine.book import OpeningBook
from src.engine.model_wrapper import ModelWrapper
from src.engine.syzygy_handler import SyzygyHandler
from src.engine.tt import TranspositionTable
from src.config.manager import ConfigurationManager
from src.models import ModelManager
from src.logging import logger


class AlphaBetaEngine:
    """Chess engine using alpha-beta + NN evaluation + root parallel search."""

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    MATE_SCORE = 100000.0

    def __init__(self, config: ConfigurationManager):
        self.config = config.get_engine_config()
        self.book = OpeningBook(self.config.book_path)
        self.evaluator = ModelWrapper(
            ModelManager(config=config).load_model(strategy=self.config.model_strategy)
        )
        self.tt = TranspositionTable(self.config.tt_size_mb)
        self.syzygy = SyzygyHandler(self.config.syzygy_path)

        cpu_count = os.cpu_count() or 1
        configured_workers = max(1, int(self.config.num_workers))
        self.num_workers = min(configured_workers, cpu_count)

        self.nodes_searched = 0
        self.tt_hits = 0
        self.stop_search = False
        self.killer_moves: Dict[int, List[chess.Move]] = {}
        self.history_heuristic: Dict[Tuple[int, int], int] = {}

    def _terminal_score(self, board: chess.Board, ply: int) -> Optional[float]:
        if board.is_checkmate():
            return -self.MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        return None

    def _move_priority(self, board: chess.Board, move: chess.Move, depth: int, tt_best_move: Optional[chess.Move] = None) -> int:
        if tt_best_move and move == tt_best_move:
            return 10_000_000

        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                score += 10 * self.PIECE_VALUES.get(victim.piece_type, 0)
                score -= self.PIECE_VALUES.get(attacker.piece_type, 0)

        if move.promotion:
            score += self.PIECE_VALUES.get(move.promotion, 0)

        if move in self.killer_moves.get(depth, []):
            score += 80_000

        piece = board.piece_at(move.from_square)
        if piece:
            score += self.history_heuristic.get((piece.piece_type, move.to_square), 0)

        return score

    def order_moves(self, board: chess.Board, moves: List[chess.Move], depth: int, tt_best_move: Optional[chess.Move] = None) -> List[chess.Move]:
        return sorted(
            moves,
            key=lambda m: self._move_priority(board, m, depth, tt_best_move),
            reverse=True,
        )

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, max_depth: int = 6) -> float:
        self.nodes_searched += 1
        terminal = self._terminal_score(board, ply=0)
        if terminal is not None:
            return terminal

        stand_pat = self.evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if max_depth <= 0:
            return stand_pat

        tactical_moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
        if not tactical_moves:
            return stand_pat

        ordered = self.order_moves(board, tactical_moves, depth=0)
        for move in ordered:
            if self.stop_search:
                break
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, max_depth=max_depth - 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0, root_call: bool = False) -> Tuple[float, Optional[chess.Move]]:
        self.nodes_searched += 1
        if self.stop_search:
            return 0.0, None

        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal, None

        alpha_original = alpha
        zobrist = chess.polyglot.zobrist_hash(board)
        tt_score, tt_move = self.tt.probe(zobrist, depth, alpha, beta)
        if tt_score is not None and not root_call:
            self.tt_hits += 1
            return tt_score, tt_move

        if depth == 0:
            return self.quiescence_search(board, alpha, beta), None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None

        ordered_moves = self.order_moves(board, legal_moves, depth=depth, tt_best_move=tt_move)
        best_move = ordered_moves[0]
        best_score = -float("inf")

        for idx, move in enumerate(ordered_moves):
            board.push(move)
            if idx == 0:
                score, _ = self.alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
                score = -score
            else:
                score, _ = self.alpha_beta(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                score = -score
                if alpha < score < beta:
                    score, _ = self.alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
                    score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                killers = self.killer_moves.setdefault(depth, [])
                if move not in killers:
                    killers.insert(0, move)
                    del killers[2:]

                piece = board.piece_at(move.from_square)
                if piece:
                    key = (piece.piece_type, move.to_square)
                    self.history_heuristic[key] = self.history_heuristic.get(key, 0) + depth * depth
                break

        if best_score <= alpha_original:
            flag = TranspositionTable.UPPERBOUND
        elif best_score >= beta:
            flag = TranspositionTable.LOWERBOUND
        else:
            flag = TranspositionTable.EXACT

        self.tt.store(zobrist, depth, best_score, best_move, flag)
        return best_score, best_move

    def _root_parallel_search(self, board: chess.Board, depth: int, time_limit: Optional[float], start_time: float) -> Tuple[float, Optional[chess.Move]]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None

        priors = self.evaluator.evaluate_child_moves(board, legal_moves)
        ordered_pairs = sorted(zip(legal_moves, priors), key=lambda x: x[1], reverse=True)
        ordered_moves = [mv for mv, _ in ordered_pairs]

        if depth < self.config.root_parallel_min_depth or len(ordered_moves) < 2 or self.num_workers == 1:
            best_score, best_move = -float("inf"), ordered_moves[0]
            for move in ordered_moves:
                board.push(move)
                score, _ = self.alpha_beta(board, depth - 1, -float("inf"), float("inf"), ply=1)
                score = -score
                board.pop()
                if score > best_score:
                    best_score, best_move = score, move
            return best_score, best_move

        best_score = -float("inf")
        best_move = ordered_moves[0]

        def search_move(move: chess.Move) -> Tuple[chess.Move, float]:
            local_board = board.copy(stack=False)
            local_board.push(move)
            local_score, _ = self.alpha_beta(local_board, depth - 1, -float("inf"), float("inf"), ply=1)
            return move, -local_score

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(search_move, move): move for move in ordered_moves}
            for future in as_completed(futures):
                if self.stop_search:
                    break
                if time_limit and (time.time() - start_time) > time_limit:
                    self.stop_search = True
                    break
                move, score = future.result()
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_score, best_move

    def iterative_deepening(self, board: chess.Board, max_depth: int = 4, time_limit: Optional[float] = None) -> Tuple[chess.Move, float, int]:
        start_time = time.time()
        best_move = None
        best_score = 0.0
        depth_reached = 0

        self.nodes_searched = 0
        self.tt_hits = 0
        self.stop_search = False
        self.killer_moves.clear()
        self.history_heuristic.clear()

        for depth in range(1, max_depth + 1):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            score, move = self._root_parallel_search(board, depth, time_limit=time_limit, start_time=start_time)
            if self.stop_search:
                break

            if move is not None:
                best_move = move
                best_score = score
                depth_reached = depth

                elapsed = time.time() - start_time
                nps = self.nodes_searched / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Depth {depth}: {move.uci()} | Score: {score:.4f} | "
                    f"Nodes: {self.nodes_searched:,} | NPS: {nps:,.0f} | "
                    f"TT hits: {self.tt_hits:,}"
                )

        return best_move, best_score, depth_reached

    def get_best_move(self, board: chess.Board) -> chess.Move:
        logger.info(f"Searching position: {board.fen()}")
        start_time = time.time()

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
            if best_move:
                return best_move

        best_move, score, depth_reached = self.iterative_deepening(
            board,
            max_depth=self.config.depth,
            time_limit=self.config.time_limit,
        )

        if best_move is None:
            best_move = next(iter(board.legal_moves))

        elapsed = time.time() - start_time
        logger.info(f"Best move: {best_move.uci()}")
        logger.info(f"Score: {score:.4f}")
        logger.info(f"Depth reached: {depth_reached}")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"Nodes: {self.nodes_searched:,}")
        logger.info(f"NPS: {self.nodes_searched / elapsed:,.0f}" if elapsed > 0 else "NPS: N/A")

        return best_move

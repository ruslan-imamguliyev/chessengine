from dataclasses import dataclass, field
from math import exp, sqrt
from typing import Dict, List, Optional, Tuple
import time

import chess

from src.config.manager import ConfigurationManager
from src.engine.book import OpeningBook
from src.engine.model_wrapper import ModelWrapper
from src.engine.syzygy_handler import SyzygyHandler
from src.logging import logger
from src.models import ModelManager


@dataclass
class MCTSNode:
    board: chess.Board
    parent: Optional["MCTSNode"] = None
    move: Optional[chess.Move] = None
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[chess.Move, "MCTSNode"] = field(default_factory=dict)

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class AlphaBetaEngine:
    """
    Parallel neural search engine using batched PUCT MCTS.

    Kept class name for backward compatibility with existing imports.
    """

    def __init__(self, config: ConfigurationManager):
        self.config = config.get_engine_config()
        self.book = OpeningBook(self.config.book_path)
        self.evaluator = ModelWrapper(
            ModelManager(config=config).load_model(strategy=self.config.model_strategy)
        )
        self.syzygy = SyzygyHandler(self.config.syzygy_path)

        self.nodes_searched = 0
        self.tt_hits = 0

    @staticmethod
    def _softmax(values: List[float], temperature: float = 1.0) -> List[float]:
        if not values:
            return []

        t = max(1e-3, temperature)
        max_v = max(values)
        exps = [exp((v - max_v) / t) for v in values]
        total = sum(exps)
        if total <= 0:
            return [1.0 / len(values)] * len(values)
        return [x / total for x in exps]

    def _terminal_value(self, board: chess.Board) -> Optional[float]:
        if board.is_checkmate():
            return -1.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        return None

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        sqrt_parent = sqrt(max(1, node.visits))
        c = self.config.puct_c

        best_score = -float("inf")
        best_child = None

        for child in node.children.values():
            q = -child.q_value()
            u = c * child.prior * sqrt_parent / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _select_leaf(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.is_expanded():
            node = self._select_child(node)
            if node is None:
                break

        if node is None:
            return root
        return node

    def _expand_batch(self, leaves: List[MCTSNode]) -> List[Tuple[MCTSNode, float]]:
        """
        Expand all leaves in one NN batch and return (leaf, value_from_leaf_perspective).
        """
        to_eval_boards: List[chess.Board] = []
        to_eval_meta: List[Tuple[MCTSNode, List[chess.Move]]] = []
        outcomes: List[Tuple[MCTSNode, float]] = []

        for leaf in leaves:
            terminal = self._terminal_value(leaf.board)
            if terminal is not None:
                outcomes.append((leaf, terminal))
                continue

            legal_moves = list(leaf.board.legal_moves)
            if not legal_moves:
                outcomes.append((leaf, 0.0))
                continue

            child_boards = []
            for move in legal_moves:
                child = leaf.board.copy(stack=False)
                child.push(move)
                child_boards.append(child)

            to_eval_boards.extend(child_boards)
            to_eval_meta.append((leaf, legal_moves))

        # Nothing to evaluate (all terminal)
        if not to_eval_boards:
            return outcomes

        child_scores = self.evaluator.evaluate_batch(to_eval_boards)

        idx = 0
        for leaf, moves in to_eval_meta:
            move_scores = child_scores[idx: idx + len(moves)]
            idx += len(moves)

            # Scores are from child perspective; parent (leaf) perspective is negated.
            parent_scores = [-s for s in move_scores]
            priors = self._softmax(parent_scores, temperature=self.config.prior_temperature)

            for move, prior in zip(moves, priors):
                child_board = leaf.board.copy(stack=False)
                child_board.push(move)
                leaf.children[move] = MCTSNode(
                    board=child_board,
                    parent=leaf,
                    move=move,
                    prior=prior,
                )

            leaf_value = max(parent_scores) if parent_scores else 0.0
            outcomes.append((leaf, leaf_value))

        return outcomes

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value, flipping perspective at each ply."""
        current = node
        v = value
        while current is not None:
            current.visits += 1
            current.value_sum += v
            current = current.parent
            v = -v

    def _run_mcts(self, board: chess.Board, time_limit: Optional[float]) -> chess.Move:
        root = MCTSNode(board=board.copy(stack=False), prior=1.0)

        start = time.time()
        simulation_budget = max(1, self.config.num_simulations)

        for sim in range(simulation_budget):
            if time_limit and (time.time() - start) > time_limit:
                break

            leaves = []
            batch_size = max(1, self.config.leaf_parallelism)
            for _ in range(batch_size):
                leaf = self._select_leaf(root)
                leaves.append(leaf)

            outcomes = self._expand_batch(leaves)
            self.nodes_searched += len(outcomes)

            for leaf, value in outcomes:
                self._backup(leaf, value)

            if (sim + 1) % 50 == 0:
                elapsed = time.time() - start
                nps = self.nodes_searched / elapsed if elapsed > 0 else 0
                logger.info(
                    f"MCTS sim {sim + 1}/{simulation_budget} | "
                    f"Root children: {len(root.children)} | Nodes: {self.nodes_searched:,} | NPS: {nps:,.0f}"
                )

        if not root.children:
            return next(iter(board.legal_moves))

        best_child = max(root.children.values(), key=lambda c: c.visits)
        return best_child.move

    def get_best_move(self, board: chess.Board) -> chess.Move:
        logger.info(f"Searching position: {board.fen()}")
        self.nodes_searched = 0

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

        start = time.time()
        best_move = self._run_mcts(board, time_limit=self.config.time_limit)
        elapsed = time.time() - start
        logger.info(f"Best move: {best_move.uci()}")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"Nodes: {self.nodes_searched:,}")
        logger.info(f"NPS: {self.nodes_searched / elapsed:,.0f}" if elapsed > 0 else "NPS: N/A")
        return best_move

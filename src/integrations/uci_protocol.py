from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, TextIO
import sys

import chess

if TYPE_CHECKING:
    from src.engine.manager import Engine


@dataclass
class UCIIdentity:
    name: str = "chessengine"
    author: str = "ruslan_imamguliyev"


class UCIProtocol:
    class EngineLike(Protocol):
        def play(self, board: chess.Board) -> chess.Move:
            ...

    def __init__(
        self,
        engine_factory: Callable[[], "UCIProtocol.EngineLike"] | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        identity: UCIIdentity | None = None,
    ) -> None:
        self._engine_factory = engine_factory or self._default_engine_factory
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._identity = identity or UCIIdentity()

        self._board = chess.Board()
        self._engine: UCIProtocol.EngineLike | None = None
    
    def run(self) -> None:
        for raw_line in self._stdin:
            line = raw_line.strip()
            if not line:
                continue

            if self._handle_command(line):
                break

    def _handle_command(self, line: str) -> bool:
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            self._send(f"id name {self._identity.name}")
            self._send(f"id author {self._identity.author}")
            self._send("uciok")
            return False

        if command == "isready":
            self._send("readyok")
            return False

        if command == "ucinewgame":
            self._board = chess.Board()
            self._engine = None
            return False

        if command == "position":
            self._set_position(tokens[1:])
            return False

        if command == "go":
            self._go(tokens[1:])
            return False

        if command == "stop":
            return False

        if command == "setoption":
            # Options are accepted for compatibility. Runtime options are not implemented yet.
            return False

        if command == "quit":
            return True

        return False

    def _go(self, args: list[str]) -> None:
        if self._board.is_game_over():
            self._send("bestmove 0000")
            return

        # The engine currently uses fixed internal limits. We parse known UCI go fields
        # to remain protocol-compatible and forward-compatible.
        _ = self._parse_go_limits(args)

        engine = self._get_engine()
        best_move = engine.play(self._board.copy(stack=True))
        self._send(f"bestmove {best_move.uci()}")

    def _parse_go_limits(self, args: list[str]) -> dict[str, int | None]:
        limits: dict[str, int | None] = {
            "wtime": None,
            "btime": None,
            "winc": None,
            "binc": None,
            "movetime": None,
            "depth": None,
            "nodes": None,
            "movestogo": None,
        }

        i = 0
        while i < len(args):
            key = args[i]
            if key in limits and i + 1 < len(args):
                try:
                    limits[key] = int(args[i + 1])
                except ValueError:
                    limits[key] = None
                i += 2
                continue

            i += 1

        return limits

    def _set_position(self, args: list[str]) -> None:
        if not args:
            return

        if args[0] == "startpos":
            board = chess.Board()
            move_start = 1
        elif args[0] == "fen":
            try:
                move_index = args.index("moves")
                fen_tokens = args[1:move_index]
                move_start = move_index
            except ValueError:
                fen_tokens = args[1:]
                move_start = len(args)

            fen = " ".join(fen_tokens)
            board = chess.Board(fen)
        else:
            return

        if move_start < len(args) and args[move_start] == "moves":
            for move_uci in args[move_start + 1 :]:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    raise ValueError(f"Illegal move in position command: {move_uci}")
                board.push(move)

        self._board = board

    def _get_engine(self) -> "UCIProtocol.EngineLike":
        if self._engine is None:
            self._engine = self._engine_factory()

        return self._engine

    @staticmethod
    def _default_engine_factory() -> "Engine":
        from src.config.manager import ConfigurationManager
        from src.engine.manager import EngineManager

        config = ConfigurationManager()
        manager = EngineManager(config)
        return manager.get_engine()

    def _send(self, message: str) -> None:
        self._stdout.write(f"{message}\n")
        self._stdout.flush()


def main() -> None:
    protocol = UCIProtocol()
    protocol.run()


if __name__ == "__main__":
    main()

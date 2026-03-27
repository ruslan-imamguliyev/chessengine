from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, TextIO
import sys
import time

import chess

if TYPE_CHECKING:
    from src.engine.manager import Engine


@dataclass
class UCIIdentity:
    name: str = "chessengine"
    author: str = "chessengine"


@dataclass(frozen=True)
class UCIOption:
    name: str
    option_type: str
    default: str
    min_value: int | None = None
    max_value: int | None = None


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
        self._options = [
            UCIOption(name="Hash", option_type="spin", default="128", min_value=1, max_value=65536),
            UCIOption(name="Threads", option_type="spin", default="1", min_value=1, max_value=1024),
        ]
        self._option_values = {option.name: option.default for option in self._options}

    def run(self) -> None:
        while True:
            raw_line = self._stdin.readline()
            if raw_line == "":
                # EOF: GUI process closed stdin or process is shutting down.
                break

            line = raw_line.strip()
            if not line:
                # Keep waiting for the next command from the GUI.
                time.sleep(0.001)
                continue

            try:
                if self._handle_command(line):
                    break
            except Exception as exc:  # pragma: no cover - defensive behavior for GUI compatibility
                self._send(f"info string error: {exc}")

    def _handle_command(self, line: str) -> bool:
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            self._send(f"id name {self._identity.name}")
            self._send(f"id author {self._identity.author}")
            for option in self._options:
                self._send(self._format_option(option))
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
            self._set_option(tokens[1:])
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

    def _set_option(self, args: list[str]) -> None:
        if not args:
            return

        if args[0] != "name":
            return

        name_tokens: list[str] = []
        value_tokens: list[str] = []
        parsing_value = False

        for token in args[1:]:
            if token == "value" and not parsing_value:
                parsing_value = True
                continue

            if parsing_value:
                value_tokens.append(token)
            else:
                name_tokens.append(token)

        option_name = " ".join(name_tokens)
        if not option_name or option_name not in self._option_values:
            return

        option = self._get_option_by_name(option_name)
        if option is None:
            return

        value = " ".join(value_tokens) if value_tokens else option.default
        if option.option_type == "spin":
            try:
                spin_value = int(value)
            except ValueError:
                return

            if option.min_value is not None and spin_value < option.min_value:
                spin_value = option.min_value
            if option.max_value is not None and spin_value > option.max_value:
                spin_value = option.max_value

            value = str(spin_value)

        self._option_values[option_name] = value

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

    def _get_option_by_name(self, name: str) -> UCIOption | None:
        for option in self._options:
            if option.name == name:
                return option

        return None

    @staticmethod
    def _format_option(option: UCIOption) -> str:
        message = f"option name {option.name} type {option.option_type} default {option.default}"
        if option.min_value is not None:
            message += f" min {option.min_value}"
        if option.max_value is not None:
            message += f" max {option.max_value}"
        return message

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

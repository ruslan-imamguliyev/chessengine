import io

import chess

from src.integrations.uci_protocol import UCIIdentity
from src.integrations.uci_protocol import UCIProtocol

if __name__ == "__main__":
    commands = "\n".join(
                [
                    "position startpos moves e2e4 e7e5 g1f3",
                    "go depth 4",
                    "quit",
                ]
            )

    stdin = io.StringIO(commands + "\n")
    stdout = io.StringIO()
    protocol = UCIProtocol(stdin=stdin, stdout=stdout)

    protocol.run()

    lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
    print(lines)
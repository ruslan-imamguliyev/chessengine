# C++ Chess Engines (Alpha-Beta + MCTS) with TorchScript and pybind11

This folder contains a C++ reimplementation of the existing Python engines:

- `AlphaBetaEngine` (iterative deepening + quiescence + transposition table)
- `MCTSEngine` (PUCT MCTS with batched leaf expansion)

The implementation uses:

- [Disservin/chess-library](https://github.com/Disservin/chess-library) for legal move generation/state checks.
- LibTorch (TorchScript runtime) for neural evaluation.
- pybind11 for Python bindings.

## Folder layout

- `include/chessengine/torch_evaluator.hpp`: TorchScript evaluator wrapper.
- `include/chessengine/alpha_beta_engine.hpp`: Alpha-Beta engine API.
- `include/chessengine/mcts_engine.hpp`: MCTS engine API.
- `src/*.cpp`: C++ implementations.
- `python/bindings.cpp`: pybind11 module (`chessengine_cpp`).

---

## 1) Compiler and toolchain setup

## Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3-dev
```

Minimum recommended versions:

- GCC 12+ (or Clang 15+)
- CMake 3.22+
- Python 3.10+ for bindings

## Windows

Install:

1. **Visual Studio 2022 Build Tools** (Desktop development with C++).
2. **CMake 3.22+** (from cmake.org installer).
3. **Python 3.10+**.
4. Optional: **Ninja** for faster builds.

Then use **x64 Native Tools Command Prompt for VS 2022**.

---

## 2) Install LibTorch + pybind11

### LibTorch

Download the LibTorch C++ distribution matching your PyTorch runtime (CPU or CUDA), then unzip:

- Linux example: `/opt/libtorch`
- Windows example: `C:\libs\libtorch`

Set environment variable `Torch_DIR` to `<libtorch>/share/cmake/Torch`.

### pybind11

If pybind11 is not already discoverable by CMake:

```bash
python -m pip install pybind11
```

Pass `-Dpybind11_DIR=$(python -m pybind11 --cmakedir)` to CMake.

---

## 3) Build steps (Linux/macOS)

From repository root:

```bash
cmake -S cpp_engine -B cpp_engine/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=/opt/libtorch/share/cmake/Torch \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

cmake --build cpp_engine/build --config Release
```

If you only need native C++ library and not Python module:

```bash
cmake -S cpp_engine -B cpp_engine/build -DCHESSENGINE_BUILD_PYTHON=OFF
cmake --build cpp_engine/build --config Release
```

---

## 4) Build steps (Windows, MSVC)

In **x64 Native Tools Command Prompt for VS 2022**:

```bat
cmake -S cpp_engine -B cpp_engine\build ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DTorch_DIR=C:\libs\libtorch\share\cmake\Torch ^
  -Dpybind11_DIR=%PYBIND11_CMAKE_DIR%

cmake --build cpp_engine\build --config Release
```

Set `%PYBIND11_CMAKE_DIR%` with:

```bat
for /f "delims=" %i in ('python -m pybind11 --cmakedir') do set PYBIND11_CMAKE_DIR=%i
```

### Windows runtime DLL note

When importing the Python extension, ensure LibTorch DLLs are on `%PATH%` (e.g. `C:\libs\libtorch\lib`).

---

## 5) Python integration example

After build, locate produced extension (`chessengine_cpp*.so` / `.pyd`) and add it to `PYTHONPATH`.

```python
import chess
import chessengine_cpp

MODEL_PATH = "models/your_model.ts"
FEN = chess.STARTING_FEN

evalr = chessengine_cpp.TorchEvaluator(MODEL_PATH)

ab_cfg = chessengine_cpp.ABConfig()
ab_cfg.depth = 4
ab_cfg.time_limit_sec = 1.5
ab = chessengine_cpp.AlphaBetaEngine(ab_cfg, evalr)
print("AB:", ab.get_best_move_uci(FEN))

mcts_cfg = chessengine_cpp.MCTSConfig()
mcts_cfg.num_simulations = 800
mcts_cfg.leaf_parallelism = 16
mcts = chessengine_cpp.MCTSEngine(mcts_cfg, evalr)
print("MCTS:", mcts.get_best_move_uci(FEN))
```

---

## 6) Integrate into this project

Recommended approach:

1. Export your trained PyTorch model to TorchScript (`.pt`/`.ts`).
2. Build `cpp_engine` with pybind11 enabled.
3. In Python, instantiate `TorchEvaluator` + engine class and call `get_best_move_uci(fen)`.
4. Convert UCI string back to `chess.Move` if needed.

Example adapter sketch:

```python
import chess
import chessengine_cpp

class CppAlphaBetaAdapter:
    def __init__(self, model_path: str):
        self.eval = chessengine_cpp.TorchEvaluator(model_path)
        cfg = chessengine_cpp.ABConfig()
        cfg.depth = 4
        self.engine = chessengine_cpp.AlphaBetaEngine(cfg, self.eval)

    def get_best_move(self, board: chess.Board) -> chess.Move:
        uci = self.engine.get_best_move_uci(board.fen())
        return chess.Move.from_uci(uci)
```

---

## 7) Design parity notes with Python engines

- Alpha-Beta keeps:
  - iterative deepening
  - TT probing/storing
  - move ordering by TT move, MVV-LVA captures, promotions, checks
  - quiescence search on captures/checks
- MCTS keeps:
  - PUCT (`q + u`)
  - batched child evaluation
  - softmax priors from model scores
  - sign flipping during backup

The evaluator uses the same 18-plane board encoding currently used in Python.

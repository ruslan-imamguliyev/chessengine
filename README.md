# chessengine
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/43567298-13e2-41e5-8654-461e644f5599" />

A research-oriented chess engine project that combines:
- **Neural position evaluation** with PyTorch,
- **Classical search** via **Alpha-Beta**,
- **Neural tree search** via **PUCT MCTS**,
- Optional **C++ backends** (via pybind11 + TorchScript) for faster inference/search.

The repository includes end-to-end tooling for:
1. ingesting and preparing a chess evaluation dataset,
2. training a neural evaluator,
3. compiling the model to TorchScript,
4. serving moves through a UCI protocol loop.
---

## Table of Contents

- [Project Goals](#project-goals)
- [Core Features](#core-features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Model Compilation (TorchScript)](#model-compilation-torchscript)
- [Running the Engine (UCI)](#running-the-engine-uci)
- [Choosing Engine Types and Backends](#choosing-engine-types-and-backends)
- [C++ Engine Build](#c-engine-build)
- [Artifacts Produced](#artifacts-produced)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Goals

This project is designed to be a practical experimentation framework for chess-engine development:

- train and compare neural evaluators,
- benchmark Alpha-Beta versus MCTS search on top of the same model,
- switch between Python and C++ implementations with a config toggle,
- integrate with UCI-compatible chess GUIs.

---

## Core Features

### Engine options

- **Alpha-Beta engine** with:
  - iterative deepening,
  - quiescence search,
  - transposition table,
  - move ordering (TT move, captures, promotions, checks).
- **MCTS engine** with:
  - PUCT child selection,
  - batched leaf expansion,
  - softmax priors from neural scores,
  - configurable simulation budget.

### Evaluation model

- Configurable **ResNet family** variants in `src/config/config.yaml`.
- 18-plane chess position encoding (`18 x 8 x 8`) from FEN/board states.
- Training loop supports mixed precision and MLflow logging.

### Integrations

- **UCI protocol loop** for engine/GUI communication.
- **Opening book** support (`polyglot` format).
- **Syzygy tablebase** probing for applicable endgames.
- Optional **C++ engine backend** for search/eval acceleration.

---

## Repository Structure

```text
.
├── src/
│   ├── config/                # dataclass config manager + YAML config
│   ├── constants/             # global constants (device, config path)
│   ├── data/                  # ingestion, preprocessing, transformation, dataset entity
│   ├── engine/                # AB, MCTS, TT, wrappers, engine manager
│   ├── integrations/          # UCI protocol + MLflow integration
│   ├── logging/               # custom logger setup
│   ├── models/                # model definitions, loading, training, compilation
│   ├── pipelines/             # runnable scripts for each stage
│   └── utils/                 # tensor/fen conversion + utility helpers
├── cpp_engine/                # C++ implementation + bindings
├── research/                  # notebooks and experiments
├── example.py                 # starts UCI protocol loop
├── pyproject.toml             # dependencies and project metadata
└── README.md
```

---

## Requirements

- **Python** `>=3.13,<3.15`
- Recommended: CUDA-capable GPU for training/inference speed.
- Optional for dataset download: Kaggle credentials configured for `kagglehub`.
- Optional for C++ backend: CMake toolchain + LibTorch + pybind11.

---

## Installation

### 1) Clone

```bash
git clone https://github.com/ruslan-imamguliyev/chessengine.git
cd chessengine
```

### 2) Install dependencies with Poetry

```bash
poetry install
```

If you want research extras:

```bash
poetry install --with research,dev
```

### 3) Enter shell

```bash
poetry shell
```

---

## Configuration

All runtime behavior is configured in:

- `src/config/config.yaml`

Key groups:

- `data_ingestion`: dataset source/method (currently `kagglehub`), output path.
- `data_preprocessing`: preprocessing output and mate-score normalization settings.
- `data_transformation`: train/validation split and tensor export path.
- `dataset_entity`: DataLoader settings (`batch_size`, `num_workers`, etc.).
- `models`: active architecture and model hyperparameters.
- `model_trainer`: optimizer/loss/scheduler/training checkpoint settings.
- `engine`: search algorithm, C++ toggle, time/depth/simulations/book/tablebase paths.
- `model_compiler`: TorchScript output path.

---

## Data Pipeline

The expected flow is:

1. **Ingestion** → download/install raw data
2. **Preprocessing** → merge CSVs and normalize evaluation targets
3. **Transformation** → convert FEN rows into tensors and save `.pth` train/val files

### 1) Ingest raw dataset

```bash
python -m src.pipelines.data_ingestion
```

### 2) Preprocess dataset

```bash
python -m src.pipelines.data_preprocessing
```

### 3) Transform to tensors

```bash
python -m src.pipelines.data_transformation
```

Notes:

- Preprocessing expects raw CSV files in the configured `data_preprocessing.input_path`.
- Transformation writes tensor bundles (`train.pth`, `val.pth`) to `data_transformation.output_path`.

---

## Model Training

Run training pipeline:

```bash
python -m src.pipelines.model_training
```

Training behavior includes:

- `torch.compile(...)` model optimization,
- SmoothL1 loss (`beta` configurable),
- AdamW + cosine annealing schedule,
- checkpointing (`latest_checkpoint.pth`, `best_checkpoint.pth`),
- MLflow metrics/artifact logging.

Make sure your MLflow tracking URI is configured (see `src/integrations/mlflow.py`) before running long training jobs.

---

## Model Compilation (TorchScript)

Compile a trained checkpoint-backed model into TorchScript (used by C++ and optional Python C++ wrappers):

```bash
python -m src.models.compiler
```

This creates a compiled model file at `model_compiler.output_path` (default `model.ts`).

---

## Running the Engine (UCI)

### Quick run

```bash
python example.py
```

or directly:

```bash
python -m src.integrations.uci_protocol
```

The process listens on stdin/stdout with UCI commands (`uci`, `isready`, `position`, `go`, etc.).

### Minimal manual UCI smoke test

```text
uci
isready
position startpos
go depth 4
quit
```

You should receive `uciok`, `readyok`, and a `bestmove ...` response.

---

## Choosing Engine Types and Backends

Set these in `src/config/config.yaml` under `engine`:

- `type: alphabeta` or `type: mcts`
- `use_cpp_backend: true` or `false`

### Example A: Python Alpha-Beta

```yaml
engine:
  type: alphabeta
  use_cpp_backend: false
  depth: 4
  time_limit: 2
```

### Example B: Python MCTS

```yaml
engine:
  type: mcts
  use_cpp_backend: false
  num_simulations: 400
  leaf_parallelism: 8
  time_limit: 2
```

### Example C: C++ backend

```yaml
engine:
  type: alphabeta
  use_cpp_backend: true
  compiled_model_path: model.ts
```

If `use_cpp_backend: true`, ensure the C++ extension is built and importable.

---

## C++ Engine Build

A full C++ implementation and build guide lives in:

- [`cpp_engine/README.md`](cpp_engine/README.md)

It documents:

- Linux/Windows compiler setup,
- CMake + Ninja/MSVC builds,
- LibTorch and pybind11 wiring,
- Python binding usage examples.

---

## Artifacts Produced

Common outputs during workflow:

- **Raw dataset**: `dataset/raw/` (configurable)
- **Preprocessed CSV**: `dataset/preprocessed/preprocessed.csv`
- **Transformed tensors**: `dataset/transformed/train.pth`, `val.pth`
- **Checkpoints**: `checkpoints/latest_checkpoint.pth`, `best_checkpoint.pth`
- **Compiled model**: `model.ts`
- **Logs**: `src/logging/logfile.log`

---

## Troubleshooting

### `FileNotFoundError` for checkpoints

The engine model loader expects a checkpoint in:

- `<checkpoint_dir>/<strategy>_checkpoint.pth`

Default strategy is `latest`, so ensure `checkpoints/latest_checkpoint.pth` exists.

### UCI returns no useful move

Verify:

- model checkpoint exists and matches selected architecture,
- engine `type` is valid (`alphabeta` or `mcts`),
- opening book/tablebase paths are correct or intentionally disabled.

### Dataset download fails

- Check internet access.
- Validate Kaggle credentials/environment for `kagglehub`.
- Re-run ingestion (the ingestor includes a retry path on connection errors).

### C++ backend import errors

- Confirm extension module is built and on `PYTHONPATH`.
- Confirm LibTorch runtime libraries are discoverable.
- Confirm compiled model (`.ts`) path is valid.

---

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE).

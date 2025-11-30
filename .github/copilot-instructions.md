# Copilot instructions for ParallelLBFGS

Purpose: quickly orient an AI coding agent to be productive in this repository.

1) Big picture
- The codebase implements a CUDA-based hyper-heuristic optimization framework. The compiled entrypoint is the CUDA executable built from `main.cu` and dozens of `core/*.cuh` headers.
- Major components:
  - core/optimizer: hyper-levels (`core/optimizer/hyper/`) and base levels (`core/optimizer/base/`).
  - core/problem: benchmark problem implementations (Rosenbrock, Rastrigin, StyblinskiTang, etc.).
  - core/common: utility code (IO, Metrics, Random, OptimizerContext, JSON helpers).
  - hhanalysis/: Python experiment orchestration and logging (experiments are driven from `hhanalysis/experiment/run.py`).
  - libs/: prebuilt native libraries (nomad, cmaes, sgtelib) that must be on `LD_LIBRARY_PATH`.

2) How to build & run (developer workflows)
- Primary convenience script: run `pip3 install -r requirements.txt` then execute `./startexp.sh` to run the Python experiment harness (it sets LD_LIBRARY_PATH and calls `hhanalysis/experiment/run.py`).
- Makefile targets show common nvcc invocations. Example to build & run the default optimizer:

```bash
export LD_LIBRARY_PATH=$(pwd)/libs
export CPATH=$(pwd)/libs/include/nomad:$(pwd)/libs/include/eigen3:$(pwd)/libs/include/cmaes
/usr/local/cuda-11.4/bin/nvcc main.cu -g -G -L$(pwd)/libs -lcmaes -lsgtelib -lnomadAlgos -lnomadEval -lnomadUtils -o opt -arch=sm_60
./opt
```

- The repository also includes CMakeLists.txt for CMake-based builds; `cmake` will add the same `main.cu` translation unit and many `core/*.cuh` sources as listed in CMakeLists.txt.

3) Important runtime / compile-time conventions
- Most solver/problem selection is done at compile-time through preprocessor defines passed to `nvcc` (see the Makefile test-* targets). Typical defines:
  - `-DPROBLEM_ROSENBROCK`, `-DPROBLEM_RASTRIGIN`, etc. to select problem.
  - `-DHH_TRIALS`, `-DITERATION_COUNT`, `-DPOPULATION_SIZE`, `-DX_DIM`, `-DLOGS_PATH` to control run parameters.
  - `HH_METHOD` / `-DHH_METHOD="SA"` selects the hyper-heuristic; default in source is `SA` (see `main.cu`).

- Logging: experiments write JSON logs under hhanalysis/logs; many Python orchestration scripts expect these files.

4) Project-specific patterns an agent should follow
- Heavy use of `.cuh` headers with implementation code: modifications to algorithm logic often happen inside `core/optimizer/hyper/*` or `core/optimizer/operators/*`.
- Classes are instantiated in `main.cu` via string-to-class selection (see the HH_METHOD checks). To add a new hyper-level, implement it in `core/optimizer/hyper/` and add selection mapping in `main.cu`.
- Configuration is often static/compile-time. For parameter sweeps prefer editing the Python orchestration in `hhanalysis/experiment/run.py` rather than changing C++ source.

5) Integration points & external deps
- Native libraries live in `libs/` and must be referenced at compile and runtime (`-L... -l...` and `LD_LIBRARY_PATH`). Examples: `libcmaes`, `libnomadAlgos`, `libsgtelib`.
- CUDA toolchain: repo assumes nvcc at `/usr/local/cuda-11.4/bin/nvcc` and CUDA compute capability `sm_60` in Makefile/CMakeLists.

6) Quick examples (copyable)
- Run the packaged experiment harness:

```bash
pip3 install -r requirements.txt
./startexp.sh
```

- Build & run Rosenbrock test (from Makefile target):

```bash
/usr/local/cuda-11.4/bin/nvcc main.cu -g -G -DSAFE -DPROBLEM_ROSENBROCK -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE=30 -DITERATION_COUNT=1000 -DPOPULATION_SIZE=30 -DX_DIM=3 -DHH_TRIALS=10 -DLOGS_PATH='"hhanalysis/logs/rosenbrock.json"' -o gd -arch=sm_60
./gd
```

7) When editing code, pay attention to
- Macro-driven configuration: changing a macro can affect many translation units.
- Binary size & compilation time: `main.cu` compiles many headers; prefer small, iterative changes and local builds when possible.
- GPU debugging: use `-g -G` and `-Xptxas -v` as in Makefile targets for verbose compilation and device debugging.

8) Useful files to inspect for context
- [README.md](../README.md) — basic requirements and startexp usage
- [Makefile](../Makefile) — common `nvcc` invocation examples and test targets
- [CMakeLists.txt](../CMakeLists.txt) — full list of compilation units
- [main.cu](../main.cu) — entrypoint, HH_METHOD mapping and lifecycle
- [hhanalysis/experiment/run.py](../hhanalysis/experiment/run.py) — Python orchestration and experiment configurations

If anything important is missing or unclear (specific local toolchain, CI, or expected dev workflow), tell me which area you'd like expanded and I'll iterate.

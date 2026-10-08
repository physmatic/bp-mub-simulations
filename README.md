# bp-mub-simulations

## Layout

```
src/        importable package (circuits, Hamiltonians, ansatz, benchmark, experiments registry)
scripts/    command-line entry points
tests/      pytest suite
outputs/    generated JSON / PDF results
```

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## Experiments

Named parameter sets live in `src/experiments.py`, each with a comment describing it
(including **GOOD PARAMS**, the all-to-all DM run). Run them with:

```
python scripts/run_experiments.py           # all experiments
python scripts/run_experiments.py tfim      # only the named ones
python scripts/run_experiments.py --list    # list available experiments
```

Each experiment writes `outputs/<name>.json` and `outputs/<name>.pdf`.
To keep a new parameter set, add an entry to `EXPERIMENTS`.

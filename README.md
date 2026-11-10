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

Staged runs: get small-n figures fast, then extend and merge (each n is seeded independently):

```
python scripts/run_experiments.py --qubits 2-12 --jobs 4 --notify
python scripts/run_experiments.py --qubits 13-16 --merge --jobs 4 --notify
```

`--notify` pops up a macOS notification and opens each finished PDF in Preview.
First-round (Ry+CZ, L=2n) experiments: `--legacy`, outputs in `outputs/old/`.

## Figures

```
python scripts/draw_mub_circuits.py          # all 2^n + 1 MUB circuits U(j), n = 1..5
python scripts/draw_mub_circuits.py 3 --open # one n, open in Preview
```

Writes `outputs/figures/mub_circuits_n{n}.pdf` (vector) and `.png`. Gates are recorded from
`src/mub_circuit.py`, so the figures always show the circuits used in the simulations.

# bp-mub-simulations

## Experiments

Named parameter sets live in `experiments.py`, each with a comment describing it
(including **GOOD PARAMS**, the all-to-all DM run). Run them with:

```
python run_experiments.py           # all experiments
python run_experiments.py tfim      # only the named ones
python run_experiments.py --list    # list available experiments
```

Each experiment writes `outputs/<name>.json` and `outputs/<name>.pdf`.
To keep a new parameter set, add an entry to `EXPERIMENTS`.

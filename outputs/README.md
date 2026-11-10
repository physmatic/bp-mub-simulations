# Outputs

Generated results: one `<name>.json` (all numbers, per-n details) and `<name>.pdf` (figure)
per experiment in `src/experiments.py`.

Current experiments (RyRz+CZ hardware-efficient ansatz, depth L = 6n):

| file | Hamiltonian | metric |
|---|---|---|
| `tfim_ryrz` | 1D TFIM | mean over parameters of Var[∂_k C] |
| `dm_ryrz` | all-to-all DM | mean over parameters of Var[∂_k C] |
| `tfim_ryrz_theta1` | 1D TFIM | Var[∂C/∂θ₁] |
| `dm_ryrz_theta1` | all-to-all DM | Var[∂C/∂θ₁] |

`old/` holds the first-round results, made with the Ry+CZ ansatz at L = 2n. That ansatz has a
polynomial (n(n+1)-dimensional) Lie algebra, so its random-init baseline shows no barren
plateau; those results are kept for reference only. `old/gradient_variance_benchmark.*` is an
earlier one-off run from `scripts/run_benchmark.py`.

`old/L5n/` holds the same four experiments at L = 5n for n = 2..12. They are correct, but the MUB
curves zigzag between even and odd n because the CZ layers cancel only for even L (see the
comment in `src/experiments.py`).

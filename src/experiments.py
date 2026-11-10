"""Registry of named benchmark experiments.

Each experiment is a set of keyword arguments for `run_variance_benchmark`.
Add a new entry to EXPERIMENTS to keep a parameter set, with a comment on what it is.

EXPERIMENTS        -> written to outputs/<name>.json / .pdf
LEGACY_EXPERIMENTS -> written to outputs/old/ (run with `--legacy`); kept only for
                      reproducibility of the first results, which used the
                      non-expressive Ry+CZ ansatz (see src/ansatz.py).
"""

# Why L = 6n entangling layers?
# -----------------------------
# Two requirements: (1) L >= 5n so the random-init baseline has converged (below), and
# (2) L even. At theta = 0 every rotation is the identity and the CZ chain C satisfies
# C^2 = I, so the ansatz collapses to C^L: the identity for even L, one CZ chain for odd L.
# With L = 5n the parity follows n, which makes the MUB curves zigzag between even and
# odd n (e.g. exact TFIM mean variance 0.2626 / 0.2857 / 0.2605 at n = 10 / 11 / 12).
# With L = 6n (always even) the curves are smooth: 0.2627 / 0.2615 / 0.2606.
# 6n costs ~20% more than 5n. The 5n results are archived in outputs/old/L5n/.
#
# Convergence of the random-init baseline:
# The random-parameter baseline is only a fair "barren plateau" reference once the
# ensemble of circuits has converged to (approximately) a 2-design; beyond that
# depth the gradient variance stops changing. We measured the mean per-parameter
# gradient variance (TFIM, RyRz+CZ, 200 samples) vs depth at fixed n:
#
#        n    L=2n      L=5n      L=10n
#        8    5.5e-2    2.5e-2    2.6e-2
#       10    3.0e-2    8.1e-3    8.3e-3
#       12    1.7e-2    2.5e-3    2.5e-3
#
# L=2n is NOT converged (7x too large at n=12, and the gap grows with n), while
# L=5n and L=10n agree within statistical error. This matches the physics of a 1D
# nearest-neighbour chain: the light cone grows by one site per layer, so ~n layers
# are needed just to connect the two ends, plus a constant factor to scramble.
# 5n is the cheapest depth that is converged in our data (10n costs 2x for no change),
# and 6n is the next even multiple, safely past convergence.
# Caveat: verified up to n=12; a spot check at n=16 against L=10n is cheap insurance.
COMMON = dict(
    qubits_list=list(range(2, 17)),
    num_samples=1_000,
    layers_factor=6,  # L = 6n entangling layers (RyRz -> CZ chain), plus a final RyRz layer
    gate_set="ryrz",
    metric="mean_param_var",
    diff_method="adjoint",
    device_name="lightning.qubit",
    param_idx="mid",
    seed=42,
)

EXPERIMENTS = {
    # 1D open-boundary TFIM (J=1.0, h=1.0): H = -J sum Z_i Z_{i+1} - h sum X_i.
    # Metric: mean over all parameters of Var[dC/dtheta_k].
    "tfim_ryrz": dict(COMMON, hamiltonian_type="tfim"),
    # All-to-all DM model (Jx=1.0, Jy=0.5, D=0.8, h=1.0, 1/n normalization).
    "dm_ryrz": dict(COMMON, hamiltonian_type="all_to_all_dm"),
    # Traditional barren-plateau metric: Var[dC/dtheta_1] of the first parameter
    # (param_idx=0: first-layer Ry on qubit 0), instead of the mean over all parameters.
    "tfim_ryrz_theta1": dict(COMMON, hamiltonian_type="tfim", metric="single_param", param_idx=0),
    "dm_ryrz_theta1": dict(COMMON, hamiltonian_type="all_to_all_dm", metric="single_param", param_idx=0),
}

# First-round results: Ry+CZ ansatz (polynomial Lie algebra, no barren plateau) at L = 2n.
LEGACY_COMMON = dict(COMMON, layers_factor=2, gate_set="ry")
LEGACY_EXPERIMENTS = {
    # GOOD PARAMS: all-to-all DM model (Jx=1.0, Jy=0.5, D=0.8, h=1.0, 1/n); good MUB-vs-Haar run.
    "good_params_dm": dict(LEGACY_COMMON, hamiltonian_type="all_to_all_dm"),
    "tfim": dict(LEGACY_COMMON, hamiltonian_type="tfim"),
    "good_params_dm_theta1": dict(LEGACY_COMMON, hamiltonian_type="all_to_all_dm", metric="single_param", param_idx=0),
    "tfim_theta1": dict(LEGACY_COMMON, hamiltonian_type="tfim", metric="single_param", param_idx=0),
    # Classical 1D Ising (ZZ + Z). MUB gradients are exactly zero here: all weight sits on the
    # computational basis, whose states are eigenstates of H.
    "ising": dict(LEGACY_COMMON, hamiltonian_type="ising"),
    "ising_theta1": dict(LEGACY_COMMON, hamiltonian_type="ising", metric="single_param", param_idx=0),
}

"""Registry of named benchmark experiments.

Each experiment is a set of keyword arguments for `run_variance_benchmark`.
Add a new entry to EXPERIMENTS to keep a parameter set, with a comment on what it is.
"""

COMMON = dict(
    qubits_list=list(range(2, 17)),
    num_samples=1_000,
    layers_factor=2,  # L = 2n entangling layers (Ry -> CZ chain), plus a final Ry layer
    metric="mean_param_var",
    diff_method="adjoint",
    device_name="lightning.qubit",
    param_idx="mid",
    seed=42,
)

EXPERIMENTS = {
    # GOOD PARAMS: all-to-all DM model (Jx=1.0, Jy=0.5, D=0.8, h=1.0, 1/n); good MUB-vs-Haar run.
    "good_params_dm": dict(COMMON, hamiltonian_type="all_to_all_dm"),
    # 1D open-boundary TFIM (J=1.0, h=1.0): H = -J sum Z_i Z_{i+1} - h sum X_i.
    "tfim": dict(COMMON, hamiltonian_type="tfim"),
    # Traditional barren-plateau metric: Var[dC/dtheta_1] of the first parameter
    # (param_idx=0: first-layer Ry on qubit 0), instead of the mean over all parameters.
    "good_params_dm_theta1": dict(COMMON, hamiltonian_type="all_to_all_dm", metric="single_param", param_idx=0),
    "tfim_theta1": dict(COMMON, hamiltonian_type="tfim", metric="single_param", param_idx=0),
    # Classical (computational-basis) 1D open-boundary Ising (J=1.0, h=1.0): H = -J sum Z_i Z_{i+1} - h sum Z_i.
    "ising": dict(COMMON, hamiltonian_type="ising"),
    "ising_theta1": dict(COMMON, hamiltonian_type="ising", metric="single_param", param_idx=0),
}

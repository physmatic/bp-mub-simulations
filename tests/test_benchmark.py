"""Unit tests for HEA ansatz and variance benchmark engine."""

from pathlib import Path
import numpy as np
import pennylane as qml


from src.ansatz import hardware_efficient_ansatz, get_num_hea_params
from src.hamiltonians import build_tfim_hamiltonian
from src.benchmark import (
    evaluate_single_param_shift,
    run_variance_benchmark,
    plot_variance_benchmark,
)


def test_hea_param_count():
    """Verify parameter count is (layers + 1) * n."""
    for n in [2, 3, 4, 5]:
        for L in [1, 2, n]:
            expected = (L + 1) * n
            assert get_num_hea_params(n, L) == expected


def test_hea_ansatz_execution():
    """Verify HEA executes cleanly inside a PennyLane QNode and preserves state norm."""
    n = 3
    layers = 2
    num_p = get_num_hea_params(n, layers)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(params):
        hardware_efficient_ansatz(params, wires=range(n), layers=layers)
        return qml.state()

    params = np.random.uniform(0, 2 * np.pi, num_p)
    state = circuit(params)
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)


def test_single_param_shift_exactness():
    """Verify evaluate_single_param_shift matches PennyLane's analytical qml.grad."""
    n = 2
    layers = 2
    num_p = get_num_hea_params(n, layers)
    H = build_tfim_hamiltonian(n, J=1.0, h=1.0)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        hardware_efficient_ansatz(params, wires=range(n), layers=layers)
        return qml.expval(H)

    pnp_params = qml.numpy.array(np.random.uniform(0, 2 * np.pi, num_p), requires_grad=True)

    # Analytical gradient via PennyLane
    qml_grads = qml.grad(circuit)(pnp_params)

    # Parameter shift for each parameter
    for k in range(num_p):
        ps_grad = evaluate_single_param_shift(circuit, np.array(pnp_params), param_idx=k)
        assert np.isclose(ps_grad, qml_grads[k], atol=1e-6), (
            f"Param {k}: expected {qml_grads[k]}, got {ps_grad}"
        )


def test_mini_benchmark_and_plot(tmp_path=None):
    """Run a small benchmark sweep on n in [2, 3] and verify results structure and plot generation."""
    test_out = Path("/tmp/test_grad_var_benchmark.png")
    results = run_variance_benchmark(
        qubits_list=[2, 3],
        num_samples=15,
        param_idx=0,
        seed=123,
        verbose=False
    )

    assert results["qubits"] == [2, 3]
    assert len(results["haar_var"]) == 2
    assert len(results["mub_var"]) == 2
    assert results["haar_var"][0] > 0.0
    assert results["mub_var"][0] > 0.0

    # Plot
    plot_variance_benchmark(results, output_path=test_out)
    assert test_out.exists(), "Benchmark plot file was not created"
    assert test_out.stat().st_size > 1000, "Benchmark plot file is unexpectedly small"
    test_out.unlink()  # Clean up temporary test file


def test_mini_benchmark_grad_norm_sq():
    """Run a small benchmark sweep on n in [2, 3] with metric='grad_norm_sq'."""
    results = run_variance_benchmark(
        qubits_list=[2, 3],
        num_samples=10,
        metric="grad_norm_sq",
        seed=42,
        verbose=False
    )
    assert results["metric"] == "grad_norm_sq"
    assert len(results["haar_var"]) == 2
    assert len(results["mub_var"]) == 2
    assert results["haar_var"][0] > 0.0
    assert results["mub_var"][0] > 0.0


def test_mini_benchmark_mean_param_var():
    """Run a small benchmark sweep on n in [2, 3] with metric='mean_param_var'."""
    results = run_variance_benchmark(
        qubits_list=[2, 3],
        num_samples=10,
        metric="mean_param_var",
        seed=42,
        verbose=False
    )
    assert results["metric"] == "mean_param_var"
    assert len(results["haar_var"]) == 2
    assert len(results["mub_var"]) == 2
    assert results["haar_var"][0] > 0.0
    assert results["mub_var"][0] > 0.0


def test_ryrz_param_count_and_gradient():
    """RyRz ansatz has 2 params per qubit per layer and its adjoint gradient matches parameter shift."""
    from src.ansatz import get_num_hea_params
    n, layers = 3, 2
    assert get_num_hea_params(n, layers, "ryrz") == 2 * (layers + 1) * n
    H = build_tfim_hamiltonian(n)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        hardware_efficient_ansatz(params, wires=range(n), layers=layers, gate_set="ryrz")
        return qml.expval(H)

    p = qml.numpy.array(np.random.default_rng(0).uniform(0, 2 * np.pi, get_num_hea_params(n, layers, "ryrz")),
                        requires_grad=True)
    g = qml.grad(circuit)(p)
    for k in range(len(p)):
        assert np.isclose(evaluate_single_param_shift(circuit, np.array(p), param_idx=k), g[k], atol=1e-6)


def test_per_n_seeding_allows_merging():
    """Results for a given n do not depend on the other n values in qubits_list, and merging matches."""
    import importlib.util, json
    spec = importlib.util.spec_from_file_location("run_experiments", Path(__file__).parent.parent / "scripts/run_experiments.py")
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)

    kw = dict(num_samples=8, hamiltonian_type="tfim", metric="single_param", param_idx=0, seed=7,
              layers_factor=1, device_name="default.qubit", verbose=False)
    rt = lambda r: json.loads(json.dumps(r))
    full = rt(run_variance_benchmark(qubits_list=[2, 3, 4], **kw))
    merged = rx.merge_results(rt(run_variance_benchmark(qubits_list=[2, 3], **kw)),
                              rt(run_variance_benchmark(qubits_list=[4], **kw)))
    assert merged["qubits"] == [2, 3, 4]
    for key in ("haar_var", "mub_var", "haar_var_err", "mub_var_err"):
        assert np.allclose(merged[key], full[key]), key
    assert merged["details"]["4"]["mub_vals"] == full["details"]["4"]["mub_vals"]


if __name__ == "__main__":
    test_hea_param_count()
    test_hea_ansatz_execution()
    test_single_param_shift_exactness()
    test_mini_benchmark_and_plot()
    test_mini_benchmark_grad_norm_sq()
    test_mini_benchmark_mean_param_var()
    print("All HEA ansatz and benchmark tests passed successfully!")

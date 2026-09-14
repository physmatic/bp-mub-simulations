"""Unit tests for HEA ansatz and variance benchmark engine."""

import sys
from pathlib import Path
import numpy as np
import pennylane as qml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "MUB_Circuits_generator"))

from src.MUB_Circuits_generator.ansatz import hardware_efficient_ansatz, get_num_hea_params
from src.MUB_Circuits_generator.hamiltonians import build_tfim_hamiltonian
from src.MUB_Circuits_generator.benchmark import (
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


if __name__ == "__main__":
    test_hea_param_count()
    test_hea_ansatz_execution()
    test_single_param_shift_exactness()
    test_mini_benchmark_and_plot()
    test_mini_benchmark_grad_norm_sq()
    test_mini_benchmark_mean_param_var()
    print("All HEA ansatz and benchmark tests passed successfully!")

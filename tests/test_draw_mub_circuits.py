"""Tests for the MUB circuit figure script (scripts/draw_mub_circuits.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pennylane as qml

from src.mub_circuit import mub_unitary

_spec = importlib.util.spec_from_file_location(
    "draw_mub_circuits", Path(__file__).parent.parent / "scripts" / "draw_mub_circuits.py")
dmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dmc)

GATES = {"Hadamard": qml.Hadamard, "S": qml.S, "CZ": qml.CZ, "PauliZ": qml.PauliZ}


def test_drawn_gates_reproduce_simulated_unitary():
    """The gates that get drawn, in drawn column order, multiply to the simulated U(j)."""
    for n in [1, 2, 3, 4]:
        for j in range(2 ** n):
            placed, _ = dmc.layout(dmc.record_ops(n, j))
            ordered = sorted(placed, key=lambda g: g[2])  # by column

            def circ():
                for name, wires, _ in ordered:
                    GATES[name](wires=wires)

            U = qml.matrix(circ, wire_order=list(range(n)))()
            assert np.allclose(U, mub_unitary(n, j)), (n, j)


def test_cz_columns_do_not_overlap():
    for n in [3, 4, 5]:
        for j in range(2 ** n):
            placed, _ = dmc.layout(dmc.record_ops(n, j))
            spans = {}
            for name, wires, col in placed:
                if name == "CZ":
                    spans.setdefault(col, []).append((min(wires), max(wires)))
            for col_spans in spans.values():
                col_spans.sort()
                assert all(a[1] < b[0] for a, b in zip(col_spans, col_spans[1:]))


def test_draw_figure_writes_pdf(tmp_path):
    pdf = dmc.draw_figure(2, out_dir=tmp_path)
    assert pdf.exists() and pdf.stat().st_size > 1000

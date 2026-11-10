"""Draw all 2^n + 1 MUB circuits U(j) for given qubit counts, one publication-ready figure per n.

The gates are recorded from `src.mub_circuit.mub_circuit` itself (the exact circuits used in the
simulations), then drawn with plain matplotlib so the output is vector PDF with identical gate
sizes in every panel. Stages are aligned across panels: H layer | S layer | CZ layers, with
commuting CZ gates packed into as few columns as possible.

Usage:
    python scripts/draw_mub_circuits.py              # n = 1..5
    python scripts/draw_mub_circuits.py 3 4          # selected n
    python scripts/draw_mub_circuits.py --open       # also open the PDFs in Preview (macOS)

Output: outputs/figures/mub_circuits_n{n}.pdf (+ .png preview)
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pennylane as qml

from src.mub_circuit import mub_circuit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "figures"

# Panel grid (columns) per n; rows follow from the 2^n + 1 panels.
GRID_COLS = {1: 3, 2: 5, 3: 3, 4: 5, 5: 5}

# Colorblind- and grayscale-safe: gates differ by fill lightness as well as hue, and by label.
STYLE = {
    "Hadamard": dict(label=r"$H$", face="#DCE9F5", edge="#2B5C8A"),
    "S": dict(label=r"$S$", face="#FBE3CF", edge="#A6561C"),
    "Adjoint(S)": dict(label=r"$S^{\dagger}$", face="#FBE3CF", edge="#A6561C"),
    "PauliZ": dict(label=r"$Z$", face="#E4E4E4", edge="#444444"),
}
WIRE_COLOR = "#333333"
CZ_COLOR = "#1A1A1A"

BOX = 0.62        # gate box side (wire spacing = column spacing = 1)
PANEL_GAP_X = 1.3
PANEL_GAP_Y = 1.25
TITLE_SPACE = 0.85
LABEL_SPACE = 0.75  # room for q_i labels left of the wires


def record_ops(n: int, j):
    """Gate list [(name, wires)] of U(j); j = None is the computational basis (identity)."""
    if j is None:
        return []
    tape = qml.tape.make_qscript(mub_circuit)(n, j)
    return [(op.name, [int(w) for w in op.wires]) for op in tape.operations]


def layout(ops):
    """Assign columns: 0 = H, 1 = single-qubit phase gates, 2.. = CZs packed greedily by wire span."""
    placed, cz_cols = [], []  # cz_cols: list of occupied (lo, hi) spans per CZ column
    for name, wires in ops:
        if name == "Hadamard":
            placed.append((name, wires, 0))
        elif len(wires) == 1:
            placed.append((name, wires, 1))
        else:  # CZ: symmetric and all CZs commute, so any packing is valid
            lo, hi = min(wires), max(wires)
            for c, spans in enumerate(cz_cols):
                if all(hi < a or lo > b for a, b in spans):
                    spans.append((lo, hi))
                    break
            else:
                c = len(cz_cols)
                cz_cols.append([(lo, hi)])
            placed.append((name, wires, 2 + c))
    return placed, len(cz_cols)


def draw_panel(ax, x0, y0, n, ops, n_cols, title):
    """Draw one circuit with its top-left wire at (x0, y0); wire i at y0 - i."""
    x_end = x0 + n_cols - 0.35
    for i in range(n):
        y = y0 - i
        ax.plot([x0 - 0.55, x_end], [y, y], color=WIRE_COLOR, lw=0.9, zorder=1, solid_capstyle="butt")
        ax.text(x0 - 0.7, y, rf"$q_{{{i}}}$", ha="right", va="center", fontsize=9, color=WIRE_COLOR)
    ax.text((x0 - LABEL_SPACE + x_end) / 2, y0 + 0.75, title, ha="center", va="center", fontsize=10.5)

    for name, wires, col in ops:
        x = x0 + col
        if name == "CZ":
            ya, yb = y0 - wires[0], y0 - wires[1]
            ax.plot([x, x], [ya, yb], color=CZ_COLOR, lw=1.1, zorder=2)
            for y in (ya, yb):
                ax.add_patch(plt.Circle((x, y), 0.1, color=CZ_COLOR, zorder=3))
            continue
        st = STYLE.get(name, dict(label=name, face="white", edge="black"))
        y = y0 - wires[0]
        ax.add_patch(FancyBboxPatch((x - BOX / 2, y - BOX / 2), BOX, BOX,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=st["face"], ec=st["edge"], lw=0.9, zorder=3))
        ax.text(x, y, st["label"], ha="center", va="center", fontsize=10, color=st["edge"], zorder=4)


def draw_figure(n: int, out_dir: Path = OUT_DIR) -> Path:
    bases = [None] + list(range(2 ** n))
    circuits = [layout(record_ops(n, j)) for j in bases]
    n_cols = 2 + max(k for _, k in circuits)  # same width in every panel: H, S, CZ columns
    grid_cols = GRID_COLS.get(n, math.ceil(math.sqrt(len(bases))))
    grid_rows = math.ceil(len(bases) / grid_cols)

    panel_w = LABEL_SPACE + n_cols + PANEL_GAP_X
    panel_h = (n - 1) + TITLE_SPACE + PANEL_GAP_Y
    total_w = grid_cols * panel_w - PANEL_GAP_X
    total_h = grid_rows * panel_h - PANEL_GAP_Y

    unit = 0.36  # inches per grid unit (wire spacing)
    fig = plt.figure(figsize=(total_w * unit + 0.2, total_h * unit + 0.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")

    for idx, (j, (ops, _)) in enumerate(zip(bases, circuits)):
        r, c = divmod(idx, grid_cols)
        x0 = c * panel_w + LABEL_SPACE
        y0 = -r * panel_h - TITLE_SPACE
        title = r"$j=\infty$ (computational)" if j is None else rf"$j={j}$"
        draw_panel(ax, x0, y0, n, ops, n_cols, title)

    ax.set_xlim(-0.3, total_w + 0.1)
    ax.set_ylim(-total_h - BOX / 2 - 0.15, 0.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"mub_circuits_n{n}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw all MUB circuits U(j) for each n.")
    parser.add_argument("ns", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--open", action="store_true", help="Open the PDFs in Preview (macOS)")
    args = parser.parse_args()

    plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
    for n in args.ns:
        pdf = draw_figure(n)
        print(f"[OK] n={n}: {pdf}")
        if args.open and sys.platform == "darwin":
            subprocess.run(["open", "-g", "-a", "Preview", str(pdf)], check=False)


if __name__ == "__main__":
    main()

"""Q3 noise-resilient quantum-kernel classifier.

Local Aer simulation comparing ideal, noisy, readout-mitigated, and
simulator-noise-scaled ZNE kernels. Processor configurations are
provisional unless verified by the organizers.

ZNE method: fit a quadratic to P(00) estimates at noise scales 1, 3, 5,
then evaluate at scale 0. This is noise-parameter scaling, NOT gate folding.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import qiskit

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

SEED = 42
SHOTS = 4096
TRAIN_PER_CLASS = 6
ZNE_SCALES = (1.0, 3.0, 5.0)

for folder in (FIG, TAB):
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Processor configuration
# ---------------------------------------------------------------------

def load_config(filename: str) -> dict[str, Any]:
    path = ROOT / "processors" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing processor configuration: {path}\n"
            "Expected processors/processor_A.json and processor_B.json."
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))

    required = {"name", "num_qubits", "basis_gates", "coupling_map", "noise_model"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"{filename} is missing keys: {sorted(missing)}")

    if set(cfg["basis_gates"]) != {"rz", "sx", "x", "cx"}:
        raise ValueError(f"{filename}: basis_gates must be exactly rz, sx, x, cx per the guide.")

    for key in (
        "one_qubit_error_probability",
        "two_qubit_error_probability",
        "readout_error_probability",
    ):
        value = float(cfg["noise_model"][key])
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{filename}: {key} must be in [0, 1).")

    return cfg


# ---------------------------------------------------------------------
# Feature map and compute-uncompute kernel
# ---------------------------------------------------------------------

def feature_map(x: np.ndarray) -> QuantumCircuit:
    """Two-qubit data-encoding feature map."""
    qc = QuantumCircuit(2, name="U(x)")
    qc.ry(float(x[0]), 0)
    qc.ry(float(x[1]), 1)
    qc.cx(0, 1)
    qc.rz(float(np.pi * x[0] * x[1]), 1)
    qc.cx(0, 1)
    return qc


def kernel_circuit(x: np.ndarray, z: np.ndarray) -> QuantumCircuit:
    """Compute-uncompute circuit; P(00) estimates the kernel value."""
    qc = QuantumCircuit(2, 2)
    qc.compose(feature_map(x), [0, 1], inplace=True)
    qc.compose(feature_map(z).inverse(), [0, 1], inplace=True)
    qc.measure([0, 1], [0, 1])
    return qc


def compile_for(qc: QuantumCircuit, config: dict[str, Any]) -> QuantumCircuit:
    """Compile once per data pair using the supplied provisional topology."""
    return transpile(
        qc,
        basis_gates=config["basis_gates"],
        coupling_map=config["coupling_map"],
        optimization_level=1,
        seed_transpiler=SEED,
    )


# ---------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------

def scaled_probability(p: float, scale: float) -> float:
    """Probability that at least one error occurs under a simple repeated-error model."""
    if scale < 1:
        raise ValueError("Noise scale must be >= 1.")
    if not 0 <= p < 1:
        raise ValueError("Base error probability must be in [0, 1).")
    return float(1.0 - (1.0 - p) ** scale)


def make_noise(config: dict[str, Any], noise_scale: float = 1.0) -> NoiseModel:
    """Create an illustrative Aer noise model.

    Readout error is held fixed across ZNE scales. Gate errors are attached
    to recognized configured basis gates. This remains a simplified model:
    the config's probabilities are not hardware-calibrated unless verified.
    """
    n = config["noise_model"]
    p1 = scaled_probability(float(n["one_qubit_error_probability"]), noise_scale)
    p2 = scaled_probability(float(n["two_qubit_error_probability"]), noise_scale)
    pr = float(n["readout_error_probability"])

    basis = set(config["basis_gates"])
    model = NoiseModel()

    # Map configured basis operations to the simplified one-/two-qubit errors.
    one_qubit_candidates = {"id", "x", "y", "z", "h", "sx", "s", "sdg", "t", "tdg"}
    two_qubit_candidates = {"cx", "cz", "ecr", "iswap", "swap"}

    one_qubit_gates = sorted(basis & one_qubit_candidates)
    two_qubit_gates = sorted(basis & two_qubit_candidates)

    if p1 > 0 and one_qubit_gates:
        error_1q = depolarizing_error(p1, 1)
        model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)

    if p2 > 0 and two_qubit_gates:
        error_2q = depolarizing_error(p2, 2)
        model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    if pr > 0:
        model.add_all_qubit_readout_error(
            ReadoutError([[1.0 - pr, pr], [pr, 1.0 - pr]])
        )

    return model


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def circuit_metrics(
    qc: QuantumCircuit,
    processor: str,
    mode: str,
    available_physical_qubits: int,
) -> dict[str, Any]:
    ops = qc.count_ops()
    twoq_names = {"cx", "cz", "ecr", "iswap", "swap"}

    return {
        "processor": processor,
        "mode": mode,
        "available_physical_qubits": int(available_physical_qubits),
        "circuit_qubits": int(qc.num_qubits),
        "depth": int(qc.depth() or 0),
        "total_gates": int(sum(ops.values())),
        "two_qubit_gates": int(sum(int(v) for k, v in ops.items() if k in twoq_names)),
        "swap_count": int(ops.get("swap", 0)),
        "operation_counts": json.dumps(
            {str(k): int(v) for k, v in ops.items()}, sort_keys=True
        ),
    }


def kernel_diagnostics(K: np.ndarray) -> dict[str, float]:
    """Summarize matrix symmetry, diagonal, and PSD behavior."""
    if K.size == 0 or K.shape[0] != K.shape[1]:
        return {
            "symmetry_max_abs": float("nan"),
            "diagonal_mean": float("nan"),
            "min_eigenvalue_symmetric_part": float("nan"),
        }

    symmetric = (K + K.T) / 2.0
    eig_min = float(np.linalg.eigvalsh(symmetric).min())
    return {
        "symmetry_max_abs": float(np.max(np.abs(K - K.T))),
        "diagonal_mean": float(np.mean(np.diag(K))),
        "min_eigenvalue_symmetric_part": eig_min,
    }


# ---------------------------------------------------------------------
# Counts and readout mitigation
# ---------------------------------------------------------------------

def counts_to_p00(
    counts: dict[str, int],
    shots: int,
    readout_p: float = 0.0,
    mitigate: bool = False,
) -> tuple[float, float | None]:
    """Return (raw P(00), mitigated P(00), if requested).

    Qiskit count keys for a 2-qubit circuit with classical bits [0, 1]
    are strings such as '00', '01', '10', '11'. Leading spaces are removed.
    The tensor assignment matrix is consistent with this two-bit ordering
    for the symmetric independent-bit readout model used here.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")

    normalized = {str(k).replace(" ", ""): int(v) for k, v in counts.items()}
    probs = np.array(
        [normalized.get(k, 0) / shots for k in ("00", "01", "10", "11")],
        dtype=float,
    )
    raw = float(probs[0])

    if not mitigate or readout_p <= 0.0:
        return raw, None
    if readout_p >= 0.5:
        raise ValueError("Readout mitigation requires readout error < 0.5.")

    a = np.array([[1.0 - readout_p, readout_p],
                  [readout_p, 1.0 - readout_p]], dtype=float)
    assignment = np.kron(a, a)

    try:
        corrected = np.linalg.solve(assignment, probs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Readout assignment matrix is singular.") from exc

    # Constrained-probability approximation: clip and renormalize all outcomes.
    corrected = np.clip(corrected, 0.0, None)
    total = float(corrected.sum())
    if total > 0:
        corrected /= total
    mitigated_p00 = float(corrected[0])
    return raw, mitigated_p00


def run_counts(
    circuits: list[QuantumCircuit],
    config: dict[str, Any] | None,
    noise_scale: float = 1.0,
) -> list[dict[str, int]]:
    """Run a batch of circuits and return one counts dictionary per circuit."""
    if config is None:
        simulator = AerSimulator()
    else:
        simulator = AerSimulator(noise_model=make_noise(config, noise_scale))

    result = simulator.run(
        circuits,
        shots=SHOTS,
        seed_simulator=SEED,
    ).result()

    return [result.get_counts(i) for i in range(len(circuits))]


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def dataset():
    """Official Q3 dataset: two moons, 100 samples, noise=0.1, seed=42.

    A stratified hold-out split is used. To keep the compute-uncompute kernel
    simulation tractable, the training subset is capped at six examples per
    class (12 total); the held-out test set remains the full split.
    """
    X, y = make_moons(n_samples=100, noise=0.1, random_state=SEED)
    Xtr_full, Xte_raw, ytr_full, yte = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    rng = np.random.default_rng(SEED)
    keep = []
    for label in (0, 1):
        ids = np.flatnonzero(ytr_full == label)
        keep.extend(
            rng.choice(ids, size=min(TRAIN_PER_CLASS, len(ids)), replace=False).tolist()
        )
    keep = np.asarray(sorted(keep), dtype=int)
    Xtr_raw, ytr = Xtr_full[keep], ytr_full[keep]

    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    # Fit preprocessing on the full training partition only, avoiding test leakage.
    scaler.fit(Xtr_full)
    Xtr = scaler.transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)
    return Xtr, Xte, ytr, yte


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def topology_plot(config: dict[str, Any], letter: str) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(config["num_qubits"])))
    graph.add_edges_from(config["coupling_map"])
    pos = nx.spring_layout(graph, seed=SEED)

    plt.figure(figsize=(7, 4.5))
    nx.draw_networkx(graph, pos=pos, node_size=900, font_size=10, width=2)
    plt.title(f'{config["name"]}\nPROVISIONAL - NOT OFFICIAL HARDWARE DATA')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / f"processor_{letter}_connectivity.png", dpi=180)
    plt.close()


def plot_classification(scores: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    pivot = scores.pivot(index="processor", columns="condition", values=metric)
    ax = pivot.plot(kind="bar", ylim=(0, 1), figsize=(11, 5.5), rot=10)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Condition", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=180)
    plt.close()


def plot_circuit_resources(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(
        axes, ("depth_mean", "two_qubit_gates_mean", "swap_count_mean")
    ):
        summary.groupby("processor")[metric].mean().plot(kind="bar", ax=ax, rot=15)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Compiled circuit resources - mean over evaluated circuits")
    fig.tight_layout()
    fig.savefig(FIG / "circuit_resource_comparison.png", dpi=180)
    plt.close(fig)


def plot_zne(zne_df: pd.DataFrame) -> None:
    if zne_df.empty:
        return
    test_df = zne_df[zne_df["kernel_set"] == "test"]
    plt.figure(figsize=(10, 6))
    for (processor, method), group in test_df.groupby(["processor", "method"]):
        group = group.sort_values("noise_scale")
        plt.plot(
            group["noise_scale"], group["mean_kernel_probability"],
            marker="o", label=f"{processor} - {method}"
        )
    plt.xlabel("Gate-noise scale (readout error held fixed)")
    plt.ylabel("Mean kernel probability P(00)")
    plt.title("Quantum-kernel noise scaling and zero-noise extrapolation")
    plt.xticks([0, 1, 3, 5])
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "zne_noise_scaling.png", dpi=180)
    plt.close()


# ---------------------------------------------------------------------
# Kernel matrix builders
# ---------------------------------------------------------------------

def build_compiled_circuits(
    Xa: np.ndarray,
    Xb: np.ndarray,
    config: dict[str, Any],
    circuit_rows: list[dict[str, Any]],
    mode: str,
) -> list[QuantumCircuit]:
    circuits = []
    for x in Xa:
        for z in Xb:
            qc = compile_for(kernel_circuit(x, z), config)
            circuits.append(qc)
            circuit_rows.append(
                circuit_metrics(qc, config["name"], mode, config["num_qubits"])
            )
    return circuits


def kernel_matrix(
    Xa: np.ndarray,
    Xb: np.ndarray,
    config: dict[str, Any],
    condition: str,
    circuit_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Build a standard ideal/noisy kernel matrix."""
    is_ideal = condition == "ideal"
    mitigate = condition == "noisy_readout_mitigated"
    readout_p = 0.0 if is_ideal else float(
        config["noise_model"]["readout_error_probability"]
    )

    circuits = build_compiled_circuits(Xa, Xb, config, circuit_rows, condition)
    counts_list = run_counts(circuits, None if is_ideal else config)

    out = np.zeros((len(Xa), len(Xb)), dtype=float)
    idx = 0
    for i in range(len(Xa)):
        for j in range(len(Xb)):
            raw, corrected = counts_to_p00(
                counts_list[idx], SHOTS, readout_p, mitigate
            )
            out[i, j] = corrected if mitigate and corrected is not None else raw
            idx += 1
    return out


def kernel_matrix_zne(
    Xa: np.ndarray,
    Xb: np.ndarray,
    config: dict[str, Any],
    circuit_rows: list[dict[str, Any]],
    zne_rows: list[dict[str, Any]],
    readout_mitigate: bool = False,
    kernel_set: str = "train",
) -> np.ndarray:
    """Build a ZNE kernel matrix by extrapolating each P(00) element."""
    scales = np.asarray(ZNE_SCALES, dtype=float)
    readout_p = float(config["noise_model"]["readout_error_probability"])
    method = (
        "zne_plus_readout_mitigation"
        if readout_mitigate else "zne_without_readout_mitigation"
    )

    circuits = build_compiled_circuits(
        Xa, Xb, config, circuit_rows, f"zne_compiled_{method}"
    )

    # Store each scale's kernel estimates: shape (scales, rows, columns).
    measured = np.empty((len(scales), len(Xa), len(Xb)), dtype=float)

    for scale_idx, scale in enumerate(scales):
        counts_list = run_counts(circuits, config, noise_scale=float(scale))
        idx = 0
        for i in range(len(Xa)):
            for j in range(len(Xb)):
                raw, corrected = counts_to_p00(
                    counts_list[idx],
                    SHOTS,
                    readout_p=readout_p,
                    mitigate=readout_mitigate,
                )
                measured[scale_idx, i, j] = (
                    corrected if readout_mitigate and corrected is not None else raw
                )
                idx += 1

        scale_matrix = measured[scale_idx]
        zne_rows.append({
            "processor": config["name"],
            "kernel_set": kernel_set,
            "noise_scale": float(scale),
            "mean_kernel_probability": float(scale_matrix.mean()),
            "std_kernel_probability": float(scale_matrix.std()),
            "num_kernel_evaluations": int(scale_matrix.size),
            "shots_per_circuit": SHOTS,
            "method": method,
            "measurement_type": "simulated_sample_estimate",
            "extrapolated_raw_mean": np.nan,
            "clipped_fraction": np.nan,
        })

    # For three fixed scale factors, fit a quadratic for each matrix element.
    raw_extrapolated = np.empty((len(Xa), len(Xb)), dtype=float)
    for i in range(len(Xa)):
        for j in range(len(Xb)):
            coefficients = np.polyfit(scales, measured[:, i, j], deg=2)
            raw_extrapolated[i, j] = np.polyval(coefficients, 0.0)

    clipped_mask = (raw_extrapolated < 0.0) | (raw_extrapolated > 1.0)
    extrapolated = np.clip(raw_extrapolated, 0.0, 1.0)

    zne_rows.append({
        "processor": config["name"],
        "kernel_set": kernel_set,
        "noise_scale": 0.0,
        "mean_kernel_probability": float(extrapolated.mean()),
        "std_kernel_probability": float(extrapolated.std()),
        "num_kernel_evaluations": int(extrapolated.size),
        "shots_per_circuit": SHOTS,
        "method": method,
        "measurement_type": "quadratic_extrapolation_clipped",
        "extrapolated_raw_mean": float(raw_extrapolated.mean()),
        "clipped_fraction": float(clipped_mask.mean()),
    })

    return extrapolated


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()
    configs = [
        ("A", load_config("processor_A.json")),
        ("B", load_config("processor_B.json")),
    ]

    for letter, cfg in configs:
        topology_plot(cfg, letter)

    Xtr, Xte, ytr, yte = dataset()
    circuit_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    zne_rows: list[dict[str, Any]] = []
    kernel_error_rows: list[dict[str, Any]] = []

    # Save feature-map diagram if Matplotlib drawing dependencies are installed.
    try:
        feature_map(np.array([0.7, 1.2])).draw(
            output="mpl", filename=str(FIG / "feature_map_circuit.png")
        )
        plt.close("all")
    except Exception as exc:
        (FIG / "feature_map_circuit_draw_error.txt").write_text(
            repr(exc), encoding="utf-8"
        )
        print("Feature-map drawing failed:", exc)

    conditions = [
        "ideal",
        "noisy_raw",
        "noisy_readout_mitigated",
        "noisy_zne",
        "noisy_zne_readout_mitigated",
    ]

    for letter, cfg in configs:
        # The ideal kernels are retained as the reference for this processor.
        # Every noisy/mitigated kernel is compared against the same split/order.
        ideal_train_kernel: np.ndarray | None = None
        ideal_test_kernel: np.ndarray | None = None

        for condition in conditions:
            print(f"\nRunning Processor {letter}: {condition}", flush=True)
            t0 = time.perf_counter()

            if condition in ("noisy_zne", "noisy_zne_readout_mitigated"):
                use_readout_mitigation = condition == "noisy_zne_readout_mitigated"
                Ktr = kernel_matrix_zne(
                    Xtr, Xtr, cfg, circuit_rows, zne_rows,
                    readout_mitigate=use_readout_mitigation,
                    kernel_set="train",
                )
                Kte = kernel_matrix_zne(
                    Xte, Xtr, cfg, circuit_rows, zne_rows,
                    readout_mitigate=use_readout_mitigation,
                    kernel_set="test",
                )
            else:
                Ktr = kernel_matrix(Xtr, Xtr, cfg, condition, circuit_rows)
                Kte = kernel_matrix(Xte, Xtr, cfg, condition, circuit_rows)

            # Enforce symmetry only on the training kernel.
            Ktr = (Ktr + Ktr.T) / 2.0

            if condition == "ideal":
                ideal_train_kernel = Ktr.copy()
                ideal_test_kernel = Kte.copy()
                train_error = np.zeros_like(Ktr)
                test_error = np.zeros_like(Kte)
            else:
                if ideal_train_kernel is None or ideal_test_kernel is None:
                    raise RuntimeError(
                        f"Ideal kernel reference missing for Processor {letter}. "
                        "Ensure the 'ideal' condition runs first."
                    )
                train_error = Ktr - ideal_train_kernel
                test_error = Kte - ideal_test_kernel

            # Record errors separately for training and held-out test kernels.
            for kernel_set, difference in (("train", train_error), ("test", test_error)):
                kernel_error_rows.append({
                    "processor": cfg["name"],
                    "condition": condition,
                    "kernel_set": kernel_set,
                    "kernel_MAE": float(np.mean(np.abs(difference))),
                    "kernel_RMSE": float(np.sqrt(np.mean(np.square(difference)))),
                    "kernel_max_absolute_error": float(np.max(np.abs(difference))),
                    "num_kernel_elements": int(difference.size),
                    "reference": "same-processor ideal simulated kernel",
                })

            model = SVC(kernel="precomputed", C=1.0)
            model.fit(Ktr, ytr)
            pred = model.predict(Kte)

            diagnostics = kernel_diagnostics(Ktr)
            score_rows.append({
                "processor": cfg["name"],
                "condition": condition,
                "accuracy": float(accuracy_score(yte, pred)),
                "precision": float(precision_score(yte, pred, zero_division=0)),
                "recall": float(recall_score(yte, pred, zero_division=0)),
                "f1": float(f1_score(yte, pred, zero_division=0)),
                "train_kernel_symmetry_max_abs": diagnostics["symmetry_max_abs"],
                "train_kernel_diagonal_mean": diagnostics["diagonal_mean"],
                "train_kernel_min_eigenvalue": diagnostics["min_eigenvalue_symmetric_part"],
                "runtime_seconds": round(time.perf_counter() - t0, 3),
            })

    # Classification metrics
    scores = pd.DataFrame(score_rows)
    scores.to_csv(TAB / "classification_metrics.csv", index=False)

    # Kernel error metrics against the same processor's ideal simulated kernels.
    kernel_error_df = pd.DataFrame(kernel_error_rows)
    kernel_error_df.to_csv(TAB / "kernel_error_metrics.csv", index=False)

    # ZNE data
    zne_df = pd.DataFrame(zne_rows)
    zne_df.to_csv(TAB / "zne_noise_scaling.csv", index=False)
    plot_zne(zne_df)

    # Circuit resource summary
    raw = pd.DataFrame(circuit_rows)
    raw.to_csv(TAB / "circuit_metrics_per_circuit.csv", index=False)
    summary = raw.groupby(["processor", "mode"], as_index=False).agg(
        available_physical_qubits=("available_physical_qubits", "max"),
        circuit_qubits=("circuit_qubits", "max"),
        depth_mean=("depth", "mean"),
        depth_max=("depth", "max"),
        total_gates_mean=("total_gates", "mean"),
        two_qubit_gates_mean=("two_qubit_gates", "mean"),
        two_qubit_gates_max=("two_qubit_gates", "max"),
        swap_count_mean=("swap_count", "mean"),
        swap_count_max=("swap_count", "max"),
    )
    summary.to_csv(TAB / "circuit_metrics.csv", index=False)

    plot_classification(
        scores, "accuracy", "accuracy_comparison.png",
        "Test accuracy by processor and condition"
    )
    plot_classification(
        scores, "f1", "f1_comparison.png",
        "Test F1 by processor and condition"
    )
    plot_circuit_resources(summary)

    meta = {
        "project": "Qiskit Fall Fest 2026 Phase 1 - Q3 Noise-Resilient Classifier",
        "status": "Local Aer simulation using the supplied Phase 1 noise values; no physical hardware execution.",
        "seed": SEED,
        "shots_per_circuit": SHOTS,
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "train_samples": int(len(ytr)),
        "test_samples": int(len(yte)),
        "dataset": "sklearn.datasets.make_moons(n_samples=100, noise=0.1, random_state=42)",
        "features": ["moon feature 1", "moon feature 2"],
        "conditions": conditions,
        "zne_scales": list(ZNE_SCALES),
        "zne_method": (
            "Quadratic extrapolation of P(00) at simulator gate-noise scales 1, 3, 5, "
            "evaluated at scale 0. Readout error is held fixed. This is noise-parameter "
            "scaling, not gate folding. Raw extrapolated mean and clipped fraction are logged."
        ),
        "noise_scaling_rule": "p_scaled = 1 - (1 - p_base)**scale",
        "readout_mitigation": (
            "Symmetric independent-bit assignment-matrix inversion; negative corrected "
            "probabilities clipped to zero and the full vector renormalized."
        ),
        "limitations": [
            "Aer noise model is synthetic and does not represent measured hardware calibration.",
            "Noise-parameter scaling is not gate-folding ZNE and need not follow a linear noise trajectory.",
            "Quadratic extrapolation may be unstable and clipped estimates can hide overshoot; raw mean and clipped fraction are logged.",
            "Finite-shot kernel estimates can yield non-positive-semidefinite training matrices.",
            "No quantum advantage is claimed.",
            "Processor A uses a participant-defined chain; Processor B uses the guide-reference graph copied from participant-supplied instructions. Verify current official kit before submission.",
            "The two-qubit feature-map circuit can fit on one connected edge of either topology, so zero SWAP overhead is expected and does not establish a general topology advantage.",
        ],
        "runtime_seconds_total": round(time.perf_counter() - started, 3),
    }
    (TAB / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\n=== Q3 Noise-Resilient Classifier complete ===")
    print("\nClassification metrics:\n", scores.to_string(index=False))
    print("\nCircuit resource summary:\n", summary.to_string(index=False))
    print("\nKernel error metrics:\n", kernel_error_df.to_string(index=False))
    print("\nZNE noise-scaling results:\n", zne_df.to_string(index=False))
    print("\nOutputs:", ROOT / "results")
    print("NOTE: Aer synthetic simulation only; no physical hardware benchmark or quantum-advantage claim.")


if __name__ == "__main__":
    main()

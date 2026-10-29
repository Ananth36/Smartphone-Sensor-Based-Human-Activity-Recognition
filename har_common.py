"""
har_common.py  -  shared helpers for the ICT-4442 HAR project (UCI HAR dataset)
================================================================================

Put this file in the project root:  MyDrive/DL PROJ/har_common.py
Every model (MLP, CNN, LSTM/BiLSTM, Transformer) imports it so that all of us
load exactly the same processed data and compute exactly the same metrics.

    import sys; sys.path.insert(0, '/content/drive/MyDrive/DL PROJ')
    import har_common as hc

    data = hc.load_data('sequence')      # raw windows  (N, 128, 9)
    data = hc.load_data('tabular')       # 561 features (N, 561)
    ...train...
    hc.save_results('results.csv', model_name='cnn', ..., proba_test=P_test, ...)

The metric functions only need numpy arrays (labels + class probabilities), so
they work for PyTorch, TensorFlow/Keras and scikit-learn alike. The efficiency
helpers (latency, FLOPs, throughput) are PyTorch-specific; Keras users can pass
their own numbers to save_results(efficiency=...).
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants (canonical UCI order: label k in the files -> class index k-1)
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]
N_CLASSES = 6
WALKING_IDX = [0, 1, 2]          # dynamic activities
STATIC_IDX = [3, 4, 5]           # static activities
SIT, STAND = 3, 4
CHANNEL_NAMES = ["body_acc_x", "body_acc_y", "body_acc_z",
                 "body_gyro_x", "body_gyro_y", "body_gyro_z",
                 "total_acc_x", "total_acc_y", "total_acc_z"]

# Column order of the shared comparison table (see the metrics checklist).
RESULT_COLUMNS = [
    # identity
    "model_name", "family", "owner", "input_representation", "run_tag", "dataset_version",
    # test performance
    "test_accuracy", "test_macro_f1", "test_macro_precision", "test_macro_recall",
    "test_weighted_f1", "test_balanced_accuracy", "test_cohen_kappa", "test_mcc",
    "test_log_loss", "test_roc_auc_macro_ovr", "test_top2_accuracy",
    # per class
    *[f"f1_{c}" for c in CLASS_NAMES],
    # error analysis
    "sit_stand_confusion_rate", "walking_variants_confusion_rate", "static_dynamic_confusion_rate",
    # robustness
    "subject_acc_mean", "subject_acc_std", "subject_acc_min",
    # generalization
    "val_accuracy", "val_macro_f1", "train_macro_f1", "gap_train_test_f1",
    # efficiency
    "params_total", "model_size_mb", "flops_per_window", "latency_ms_cpu_b1",
    "latency_ms_gpu_b1", "throughput_gpu_wps", "peak_gpu_mem_mb",
    # training cost
    "train_time_s", "epochs_trained", "best_epoch", "time_per_epoch_s",
    # reproducibility
    "hardware", "framework", "seed", "notes",
]


# --------------------------------------------------------------------------- #
# Locating the project folder on Google Drive
# --------------------------------------------------------------------------- #
def find_project_root(start: str | os.PathLike = "/content/drive/MyDrive",
                      name: str = "dl proj") -> Path:
    """Return the 'DL PROJ' folder (matched case-insensitively, so 'DL Proj' works too).

    Set the environment variable HAR_PROJECT_ROOT to override the search.
    """
    env = os.environ.get("HAR_PROJECT_ROOT")
    if env:
        return Path(env)
    start = Path(start)
    if not start.exists():
        raise FileNotFoundError(
            f"{start} not found. In Colab run: from google.colab import drive; "
            f"drive.mount('/content/drive')")
    for p in [start, *start.iterdir()]:
        if p.is_dir() and p.name.lower().replace("_", " ") == name:
            return p
    # one level deeper (e.g. MyDrive/Sem7/DL PROJ)
    for p in start.iterdir():
        if p.is_dir():
            try:
                for q in p.iterdir():
                    if q.is_dir() and q.name.lower().replace("_", " ") == name:
                        return q
            except PermissionError:
                pass
    raise FileNotFoundError(f"Could not find a folder called '{name}' under {start}. "
                            f"Set os.environ['HAR_PROJECT_ROOT'] to its path.")


def default_processed_dir() -> Path:
    return find_project_root() / "processed"


# --------------------------------------------------------------------------- #
# Loading the processed dataset
# --------------------------------------------------------------------------- #
def load_meta(processed_dir: str | os.PathLike | None = None) -> dict:
    processed_dir = Path(processed_dir) if processed_dir else default_processed_dir()
    with open(processed_dir / "meta.json") as f:
        return json.load(f)


def load_data(representation: str = "sequence", augmented: bool = False,
              processed_dir: str | os.PathLike | None = None,
              channels_first: bool = False) -> dict:
    """Load the processed, model-ready dataset.

    representation : 'sequence' -> X of shape (N, 128, 9)  [raw_9ch_128, z-scored]
                     'tabular'  -> X of shape (N, 561)       [tabular_561, standardized]
    augmented      : sequence only. Replace X_train / y_train / subj_train with the
                     shared pre-augmented training set (original + 1 augmented copy).
    channels_first : sequence only. Return (N, 9, 128) - the layout PyTorch Conv1d wants.

    Returns a dict with X_train, y_train, subj_train, X_val, y_val, subj_val,
    X_test, y_test, subj_test (numpy arrays), plus 'meta' and 'class_names'.
    """
    processed_dir = Path(processed_dir) if processed_dir else default_processed_dir()
    if representation not in ("sequence", "tabular"):
        raise ValueError("representation must be 'sequence' or 'tabular'")
    with np.load(processed_dir / f"{representation}.npz") as z:
        d = {k: z[k] for k in z.files}
    if augmented:
        if representation != "sequence":
            raise ValueError("The pre-augmented set exists only for the sequence representation.")
        with np.load(processed_dir / "sequence_train_augmented.npz") as z:
            d["X_train"], d["y_train"], d["subj_train"] = z["X_train"], z["y_train"], z["subj_train"]
    if channels_first and representation == "sequence":
        for k in ("X_train", "X_val", "X_test"):
            d[k] = np.ascontiguousarray(d[k].transpose(0, 2, 1))
    d["meta"] = load_meta(processed_dir)
    d["class_names"] = list(CLASS_NAMES)
    return d


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _safe_proba(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """The three error-analysis rates defined in the metrics checklist."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss = np.isin(y_true, [SIT, STAND])
    sit_stand = np.mean(((y_true == SIT) & (y_pred == STAND)) | ((y_true == STAND) & (y_pred == SIT)),
                        where=ss) if ss.any() else np.nan
    wk = np.isin(y_true, WALKING_IDX)
    walk = np.mean(np.isin(y_pred, WALKING_IDX) & (y_pred != y_true), where=wk) if wk.any() else np.nan
    t_static, p_static = np.isin(y_true, STATIC_IDX), np.isin(y_pred, STATIC_IDX)
    static_dyn = np.mean(t_static != p_static)
    return {"sit_stand_confusion_rate": float(sit_stand),
            "walking_variants_confusion_rate": float(walk),
            "static_dynamic_confusion_rate": float(static_dyn)}


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, prefix: str = "") -> dict:
    """All performance metrics from labels + class probabilities (N, 6)."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 f1_score, log_loss, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score, top_k_accuracy_score)
    y_true = np.asarray(y_true).astype(int)
    p = _safe_proba(proba)
    y_pred = p.argmax(1)
    labels = list(range(N_CLASSES))
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, p, labels=labels),
        "roc_auc_macro_ovr": roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=labels),
        "top2_accuracy": top_k_accuracy_score(y_true, p, k=2, labels=labels),
    }
    m = {f"{prefix}{k}": float(v) for k, v in m.items()}
    f1s = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    for c, v in zip(CLASS_NAMES, f1s):
        m[f"{prefix}f1_{c}"] = float(v)
    return m


def subject_stats(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray) -> dict:
    accs = {int(s): float(np.mean(y_pred[subjects == s] == y_true[subjects == s]))
            for s in np.unique(subjects)}
    v = np.array(list(accs.values()))
    return {"subject_acc_mean": float(v.mean()), "subject_acc_std": float(v.std(ddof=0)),
            "subject_acc_min": float(v.min()), "per_subject": accs}


# --------------------------------------------------------------------------- #
# Efficiency (PyTorch)
# --------------------------------------------------------------------------- #
def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def file_size_mb(path: str | os.PathLike) -> float:
    return os.path.getsize(path) / 2 ** 20


def count_flops(model, input_shape=(1, 128, 9)) -> float | None:
    """FLOPs for one window with torch.utils.flop_counter (counts 2 FLOPs per multiply-add,
    matmuls / convolutions / attention; element-wise ops are ignored)."""
    import torch
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    dev = next(model.parameters()).device
    x = torch.zeros(*input_shape, device=dev)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            model(x)
    model.train(was_training)
    return float(fc.get_total_flops()) / input_shape[0]


def measure_latency(model, input_shape=(1, 128, 9), device: str = "cpu", n_warmup: int = 50,
                    n_runs: int = 300, num_threads: int | None = 1) -> float:
    """Mean inference time in ms for ONE batch of `input_shape`, after warm-up.

    CPU latency is measured on `num_threads` threads (default 1, so numbers are comparable
    across Colab sessions). The model is moved to `device` and NOT moved back.
    """
    import copy
    import torch
    m = copy.deepcopy(model).to(device).eval()
    x = torch.randn(*input_shape, device=device)
    old_threads = torch.get_num_threads()
    if device == "cpu" and num_threads:
        torch.set_num_threads(num_threads)
    try:
        with torch.inference_mode():
            for _ in range(n_warmup):
                m(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                m(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / n_runs
    finally:
        torch.set_num_threads(old_threads)
    return dt * 1000.0


def measure_throughput(model, window_shape=(128, 9), batch: int = 256, device: str = "cuda",
                       n_warmup: int = 10, n_runs: int = 50) -> float | None:
    """Windows per second at batch size `batch` (GPU)."""
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        return None
    ms = measure_latency(model, (batch, *window_shape), device=device, n_warmup=n_warmup,
                         n_runs=n_runs, num_threads=None)
    return batch / (ms / 1000.0)


def hardware_string() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}; CPU: {platform.processor() or platform.machine()}"
    except ImportError:
        pass
    return f"CPU: {platform.processor() or platform.machine()} ({os.cpu_count()} cores)"


def framework_string() -> str:
    parts = []
    for mod, label in (("torch", "PyTorch"), ("tensorflow", "TensorFlow"), ("sklearn", "scikit-learn")):
        try:
            parts.append(f"{label} {__import__(mod).__version__}")
        except ImportError:
            pass
    return ", ".join(parts[:1]) if parts else "unknown"


# --------------------------------------------------------------------------- #
# Saving one row of the comparison table
# --------------------------------------------------------------------------- #
def build_results_row(*, model_name: str, family: str, owner: str, input_representation: str,
                      run_tag: str, y_train, proba_train, y_val, proba_val, y_test, proba_test,
                      subj_test, efficiency: dict | None = None, training: dict | None = None,
                      seed: int = 42, notes: str = "", hardware: str | None = None,
                      framework: str | None = None, processed_dir=None) -> dict:
    meta = load_meta(processed_dir)
    test = compute_metrics(y_test, proba_test, prefix="test_")
    val = compute_metrics(y_val, proba_val, prefix="val_")
    train = compute_metrics(y_train, proba_train, prefix="train_")
    y_pred_test = np.asarray(proba_test).argmax(1)
    row = {
        "model_name": model_name, "family": family, "owner": owner,
        "input_representation": input_representation, "run_tag": run_tag,
        "dataset_version": meta["dataset_version"],
    }
    row.update({k: v for k, v in test.items() if not k.startswith("test_f1_")})
    for c in CLASS_NAMES:
        row[f"f1_{c}"] = test[f"test_f1_{c}"]
    row.update(confusion_rates(y_test, y_pred_test))
    ss = subject_stats(np.asarray(y_test), y_pred_test, np.asarray(subj_test))
    row.update({k: ss[k] for k in ("subject_acc_mean", "subject_acc_std", "subject_acc_min")})
    row.update({"val_accuracy": val["val_accuracy"], "val_macro_f1": val["val_macro_f1"],
                "train_macro_f1": train["train_macro_f1"],
                "gap_train_test_f1": train["train_macro_f1"] - test["test_macro_f1"]})
    for k in ("params_total", "model_size_mb", "flops_per_window", "latency_ms_cpu_b1",
              "latency_ms_gpu_b1", "throughput_gpu_wps", "peak_gpu_mem_mb"):
        row[k] = (efficiency or {}).get(k, np.nan)
    for k in ("train_time_s", "epochs_trained", "best_epoch", "time_per_epoch_s"):
        row[k] = (training or {}).get(k, np.nan)
    row.update({"hardware": hardware or hardware_string(), "framework": framework or framework_string(),
                "seed": seed, "notes": notes})
    return {k: row.get(k, np.nan) for k in RESULT_COLUMNS}


def save_results(csv_path: str | os.PathLike, **kwargs) -> dict:
    """Compute every checklist metric and write/append one row to `csv_path`.

    A row with the same (model_name, run_tag) is replaced, so re-running a notebook
    does not create duplicates. Returns the row as a dict.
    Arguments: see build_results_row.
    """
    row = build_results_row(**kwargs)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row], columns=RESULT_COLUMNS)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        keep = ~((old["model_name"] == row["model_name"]) & (old["run_tag"] == row["run_tag"]))
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(csv_path, index=False)
    return row


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, y_pred, path=None, title="Confusion matrix"):
    """Two panels: counts and row-normalised recall."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    short = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, mat, fmt, ttl in ((axes[0], cm, "d", "counts"), (axes[1], cmn, ".2f", "row-normalised (recall)")):
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(6), short)
        ax.set_yticks(range(6), short)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} - {ttl}")
        thr = mat.max() / 2
        for i in range(6):
            for j in range(6):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thr else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig

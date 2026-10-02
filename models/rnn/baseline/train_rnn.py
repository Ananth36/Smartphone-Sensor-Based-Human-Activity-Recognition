"""Train and evaluate the recurrent family (LSTM / BiLSTM / DeepConvLSTM).

Model 4 of the ICT-4442 comparison, owner: Nandan Reddy Maram (230911032).

The script is the exact counterpart of Transformer_Baseline.ipynb: same data
(`hc.load_data`), same optimiser, same early-stopping rule on validation macro-F1,
same efficiency measurements, and the same `hc.save_results` record, so the three
recurrent rows are directly comparable with the MLP / CNN / Transformer rows.

    python train_rnn.py                          # all three, baseline protocol
    python train_rnn.py --models lstm            # one model
    python train_rnn.py --models bilstm --epochs 60

HAR_PROJECT_DIR points at the folder that contains processed/ (the notebook mounts
Google Drive at /content/drive/MyDrive/DL Proj).
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = os.environ.get("HAR_PROJECT_DIR", "/content/drive/MyDrive/DL Proj")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

PROC_DIR = os.path.join(PROJECT_DIR, "processed")
assert os.path.isfile(os.path.join(PROC_DIR, "har_common.py")), "Run HAR_Preprocessing.ipynb first."
sys.path.insert(0, PROC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_common as hc
import rnn_models

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
if os.environ.get("HAR_NUM_THREADS"):
    torch.set_num_threads(int(os.environ["HAR_NUM_THREADS"]))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# Training protocol is shared with the CNN and Transformer baselines: AdamW, constant
# learning rate, no augmentation, no class weights, early stopping on validation
# macro-F1, single test evaluation at the end.
HP = dict(
    batch_size=64, lr=1e-3, weight_decay=1e-2, optimizer="AdamW",
    epochs=150, early_stopping_patience=20, grad_clip=1.0,
    augmentation=False, class_weights=False, lr_schedule=None,
)
MODEL_NAMES = list(rnn_models.MODELS)
print(f"Device: {DEVICE} {torch.cuda.get_device_name(0) if USE_AMP else ''} | torch {torch.__version__}")
print(f"Dataset version: {hc.DATASET_VERSION} | classes: {hc.CLASSES}")
print(f"Owner: {rnn_models.OWNER}")


@torch.no_grad()
def predict_proba(m, X, bs=512):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = m(xb)
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


def learning_curves(history, best_epoch, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("loss")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val")
    ax[1].set_title("accuracy")
    ax[2].plot(ep, [h["val_macro_f1"] for h in history], color="C2")
    ax[2].set_title("val macro-F1")
    for a in ax:
        a.axvline(best_epoch, ls="--", c="grey", lw=1)
        a.set_xlabel("epoch")
    ax[0].legend()
    fig.suptitle(f"{title} — best epoch {best_epoch}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def export(model, model_name, run_tag, hp, data, best_epoch):
    """TorchScript / ONNX export of raw-window -> probabilities, as for the Transformer."""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    model_dir = os.path.join(PROJECT_DIR, "models", "rnn", run_tag, model_name)
    os.makedirs(model_dir, exist_ok=True)
    artifacts = {}
    p = os.path.join(model_dir, f"{model_name}_state_dict.pt")
    torch.save(model.state_dict(), p)
    artifacts["state_dict"] = p
    cfg = {"model": model_name, "class": type(model).__name__, "hyperparameters": {**HP, **hp},
           "input": {"shape": ["N", int(data["X_train"].shape[1]), int(data["X_train"].shape[2])],
                     "units": "raw UCI HAR inertial signals", "channel_order": hc.SIGNALS},
           "output": {"type": "probabilities", "classes": hc.CLASSES},
           "normalization": hc.SEQ_NORM, "dataset_version": hc.DATASET_VERSION, "best_epoch": best_epoch}
    p = os.path.join(model_dir, "model_config.json")
    json.dump(cfg, open(p, "w"), indent=2)
    artifacts["config"] = p

    norm = hc.SEQ_NORM

    class DeployModel(nn.Module):
        """raw (N,128,9) -> clip -> z-score -> rnn -> softmax probabilities (N,6)"""

        def __init__(self, net):
            super().__init__()
            self.net = net
            for k in ("clip_lo", "clip_hi", "mean", "std"):
                self.register_buffer(k, torch.tensor(norm[k], dtype=torch.float32))

        def forward(self, x):
            x = torch.maximum(torch.minimum(x, self.clip_hi), self.clip_lo)
            return torch.softmax(self.net((x - self.mean) / self.std), dim=-1)

    model_cpu = copy.deepcopy(model).cpu().eval()
    deploy = DeployModel(model_cpu).eval()
    raw_test = data["X_test"] * np.array(norm["std"], np.float32) + np.array(norm["mean"], np.float32)
    example = torch.from_numpy(raw_test[:2])
    ref = deploy(example).detach().numpy()
    try:
        p = os.path.join(model_dir, f"{model_name}_deploy.torchscript.pt")
        torch.jit.trace(deploy, example, check_trace=False).save(p)
        err = float(np.abs(torch.jit.load(p)(example).detach().numpy() - ref).max())
        artifacts["torchscript"] = p
        print(f"TorchScript saved  (max |diff| vs PyTorch = {err:.2e})")
    except Exception as e:
        print("TorchScript export skipped:", repr(e)[:200])
    return artifacts


def run_one(model_name, data, run_tag="baseline", epochs=None, aug_override=None):
    hp_full = dict(HP)
    if epochs:
        hp_full["epochs"] = epochs
    if aug_override is not None:
        hp_full["augmentation"] = aug_override

    d = data if not hp_full["augmentation"] else hc.load_data("sequence", augmented_train=True)
    seq_len, n_ch = d["X_train"].shape[1:]
    model, hp = rnn_models.build(model_name, n_ch, seq_len, hc.N_CLASSES)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 78}\n{model_name} | parameters {n_params:,} | hp {hp}\n{model}")

    g = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(d["X_train"]), torch.from_numpy(d["y_train"]).long()),
                          batch_size=hp_full["batch_size"], shuffle=True, generator=g, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp_full["lr"], weight_decay=hp_full["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    if USE_AMP:
        torch.cuda.reset_peak_memory_stats()

    history, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t_train = time.time()
    for epoch in range(1, hp_full["epochs"] + 1):
        model.train()
        t0 = time.time()
        loss_sum = correct = seen = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), hp_full["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)

        pv = predict_proba(model, d["X_val"])
        yv = d["y_val"]
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_acc=correct / seen,
                   val_loss=log_loss(yv, np.clip(pv, 1e-7, 1), labels=range(6)),
                   val_acc=accuracy_score(yv, pv.argmax(1)),
                   val_macro_f1=f1_score(yv, pv.argmax(1), average="macro"),
                   lr=optimizer.param_groups[0]["lr"], epoch_time_s=time.time() - t0)
        history.append(row)
        improved = row["val_macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad = row["val_macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"ep {epoch:3d} | loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} F1 {row['val_macro_f1']:.4f} | "
              f"{row['epoch_time_s']:.1f}s {'*' if improved else ''}")
        if bad >= hp_full["early_stopping_patience"]:
            print(f"Early stopping: no val macro-F1 improvement for {bad} epochs.")
            break

    train_time = time.time() - t_train
    model.load_state_dict(best_state)
    training = dict(train_time_s=round(train_time, 2), epochs_trained=len(history), best_epoch=best_epoch,
                    time_per_epoch_s=round(float(np.mean([h["epoch_time_s"] for h in history])), 3),
                    best_val_macro_f1=best_f1,
                    early_stopping=f"patience {hp_full['early_stopping_patience']} on val macro-F1",
                    peak_gpu_mem_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if USE_AMP else None)
    print(f"\nBest epoch {best_epoch}  val macro-F1 {best_f1:.4f}  |  {train_time:.0f}s total")

    result_dir = os.path.join(hc.RESULTS_DIR, model_name, run_tag)
    os.makedirs(result_dir, exist_ok=True)
    learning_curves(history, best_epoch, os.path.join(result_dir, "learning_curves.png"),
                    f"{model_name} {run_tag}")

    artifacts = export(model, model_name, run_tag, {**hp_full, **hp}, data, best_epoch)

    p_train = predict_proba(model, d["X_train"])
    p_val = predict_proba(model, d["X_val"])
    p_test = predict_proba(model, d["X_test"])          # first and only use of the test set

    model_cpu = copy.deepcopy(model).cpu().eval()
    x1 = torch.from_numpy(d["X_test"][:1])
    x256 = torch.from_numpy(d["X_test"][:256])
    with torch.inference_mode():
        cpu = hc.measure_latency(lambda x: model_cpu(x).numpy(), x1, x256)
    gpu = {}
    if USE_AMP:
        model.eval()
        with torch.inference_mode():
            gpu = hc.measure_latency(lambda x: model(x).cpu(), x1.to(DEVICE),
                                     torch.from_numpy(d["X_test"][:256]).to(DEVICE))

    efficiency = dict(
        params_total=n_params, params_trainable=n_params,
        model_size_mb=round(os.path.getsize(artifacts["state_dict"]) / 2**20, 3),
        flops_per_window=rnn_models.count_flops(model, seq_len),
        flops_tool="analytic GEMM count from weight shapes, 2 x MACs (torch flop_counter cannot trace the fused ATen LSTM kernels)",
        latency_ms_cpu_b1=round(cpu["latency_ms_b1_mean"], 3),
        latency_ms_cpu_b1_p95=round(cpu["latency_ms_b1_p95"], 3),
        cpu_threads=torch.get_num_threads(), throughput_cpu_wps=round(cpu["throughput_wps"], 1),
        latency_ms_gpu_b1=round(gpu["latency_ms_b1_mean"], 3) if gpu else None,
        throughput_gpu_wps=round(gpu["throughput_wps"], 1) if gpu else None,
        peak_gpu_mem_mb=training["peak_gpu_mem_mb"])

    note = (f"{model_name}: 2x LSTM(hidden={hp['hidden']}, dropout={hp['dropout']})" if model_name != "deepconvlstm"
            else "DeepConvLSTM: 5x Conv1D(64,k=5)+BN+ReLU then 2x LSTM(128), pooling={}".format(hp["pooling"]))
    note += ("; shared baseline protocol (no augmentation, no class weights, constant LR, "
             "early stop on val macro-F1)")
    if not USE_AMP:
        note += f"; trained on CPU only ({hc.hardware_info().get('cpu')}) - latency/throughput are not comparable with the Colab Tesla T4 runs of the CNN and Transformer"

    record = hc.save_results(
        model_name=model_name, family=rnn_models.FAMILY, owner=rnn_models.OWNER,
        input_representation="raw_9ch_128", run_tag=run_tag,
        y_test=d["y_test"], p_test=p_test, s_test=d["s_test"],
        y_val=d["y_val"], p_val=p_val, y_train=d["y_train"], p_train=p_train,
        efficiency=efficiency, training=training, hyperparameters={**hp_full, **hp}, history=history,
        framework=f"PyTorch {torch.__version__}", artifacts=artifacts, seed=SEED, notes=note)

    t = record["test"]
    print(f"\nTEST  accuracy {t['accuracy']:.4f} | macro-F1 {t['macro_f1']:.4f} | kappa {t['cohen_kappa']:.4f} | "
          f"sit/stand confusion {t['sit_stand_confusion_rate']:.3f} | worst subject acc {t['subject_acc_min']:.3f}")
    print(f"VAL   macro-F1 {record['val']['macro_f1']:.4f} | TRAIN macro-F1 {record['train']['macro_f1']:.4f}")
    print(open(os.path.join(result_dir, "classification_report.txt")).read())
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_NAMES), help="comma separated: " + ",".join(MODEL_NAMES))
    ap.add_argument("--run-tag", default="baseline")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--augmentation", dest="augmentation", action="store_true", default=None)
    ap.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    a = ap.parse_args()

    data = hc.load_data("sequence", augmented_train=False)
    for sp in ("train", "val", "test"):
        print(f"{sp:5s} X {data[f'X_{sp}'].shape}  y {np.bincount(data[f'y_{sp}'], minlength=6)}")

    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        run_one(name, data, run_tag=a.run_tag, epochs=a.epochs, aug_override=a.augmentation)


if __name__ == "__main__":
    main()

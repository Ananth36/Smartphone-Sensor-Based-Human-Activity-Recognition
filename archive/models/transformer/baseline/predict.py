"""Run the exported HAR Transformer on raw smartphone sensor windows.

  python predict.py --uci-dir "path/to/UCI HAR Dataset/test"   # reads Inertial Signals (+ y_test.txt if present)
  python predict.py --npy windows.npy                          # raw windows, shape (N, 128, 9)
  add --onnx to use onnxruntime instead of PyTorch
Channel order: body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z (see model_config.json).
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "model_config.json")))
CLASSES = CFG["output"]["classes"]; SIGNALS = CFG["input"]["channel_order"]


def load_uci_split(split_dir):
    split = os.path.basename(os.path.normpath(split_dir))
    X = np.stack([np.loadtxt(os.path.join(split_dir, "Inertial Signals", f"{s}_{split}.txt")) for s in SIGNALS], -1)
    y_path = os.path.join(split_dir, f"y_{split}.txt")
    y = np.loadtxt(y_path, dtype=int) - 1 if os.path.exists(y_path) else None
    return X.astype(np.float32), y


def predict(X, use_onnx=False):
    if use_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.join(HERE, "transformer_deploy.onnx"), providers=["CPUExecutionProvider"])
        return np.concatenate([sess.run(None, {"raw_window": X[i:i + 512]})[0] for i in range(0, len(X), 512)])
    import torch
    m = torch.jit.load(os.path.join(HERE, "transformer_deploy.torchscript.pt"), map_location="cpu").eval()
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(X[i:i + 512])).numpy() for i in range(0, len(X), 512)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--uci-dir"); g.add_argument("--npy")
    ap.add_argument("--onnx", action="store_true"); ap.add_argument("--out", help="optional CSV of predictions")
    a = ap.parse_args()
    X, y = load_uci_split(a.uci_dir) if a.uci_dir else (np.load(a.npy).astype(np.float32), None)
    assert X.ndim == 3 and X.shape[1:] == (128, 9), f"expected (N,128,9), got {X.shape}"
    t0 = time.time(); P = predict(X, a.onnx); dt = time.time() - t0
    pred = P.argmax(1)
    print(f"{len(X)} windows in {dt:.2f}s ({'onnx' if a.onnx else 'torchscript'})")
    for i, c in enumerate(CLASSES):
        print(f"  {c:20s} {int((pred == i).sum()):5d}")
    if y is not None:
        print(f"accuracy vs labels: {(pred == y).mean():.4f}")
    if a.out:
        import csv
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["window", "predicted"] + [f"p_{c}" for c in CLASSES])
            for i, (k, p) in enumerate(zip(pred, P)):
                w.writerow([i, CLASSES[k]] + [f"{v:.5f}" for v in p])
        print("saved", a.out)

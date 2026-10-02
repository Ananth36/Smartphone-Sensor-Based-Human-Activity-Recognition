# Model 4 — Recurrent family (LSTM / BiLSTM / DeepConvLSTM)

Owner: **Nandan Reddy Maram** (230911032). Papers [3] Ordóñez & Roggen (2016) and
[4] Hammerla et al. (2016).

## What is here

| file | purpose |
|---|---|
| `RNN_Baseline.ipynb` | Colab notebook; one cell per model, prints the comparison table |
| `rnn_models.py` | the three architectures + the analytic FLOP counter |
| `train_rnn.py` | shared training / evaluation loop, `hc.save_results` records, export |
| `../../results/rnn/` | the produced metrics, curves and confusion matrices |
| `../../results/rnn/FAMILY_COMPARISON.md` | RNN vs CNN vs Transformer vs MLP family comparison |

## The three variants

| run | architecture | parameters | FLOPs/window | reference |
|---|---|---|---|---|
| `lstm` | 2 x LSTM(128), dropout 0.5, mean+max temporal pooling | 204,806 | 51,514,368 | [4] |
| `bilstm` | 2 x BiLSTM(78 per direction), dropout 0.5, mean+max pooling | 204,678 | 51,281,568 | [3] |
| `deepconvlstm` | 5 x Conv1D(64, kernel 5) + BN + ReLU → 2 x LSTM(128), last-step pooling | 317,958 | 80,430,592 | [3] |

`bilstm` is deliberately parameter- and FLOP-matched to `lstm` (204,678 vs 204,806 parameters,
51.28M vs 51.51M FLOPs), so the LSTM/BiLSTM difference isolates bidirectionality instead of
capacity. The control is required because [3] shows that going bidirectional roughly doubles
the parameter count, which would otherwise confound the family comparison that [4] performs.

## Protocol

Identical to the CNN and Transformer baselines, so the four rows are comparable:

* `hc.load_data("sequence")` → `N x 128 x 9` per-channel z-scored windows; no per-model preprocessing.
* AdamW, `lr = 1e-3`, `weight_decay = 0.01`, batch 64, gradient clipping 1.0, constant LR.
* No data augmentation, no class weights, no LR schedule.
* Early stopping on **validation macro-F1**, patience 20, max 150 epochs.
* Seed 42. Test set touched once, at the end.
* All metrics via `har_common.save_results(...)`; `dataset_version` must equal the fingerprint
  in `meta.json` (`uci-har-65a1fc6120e8`).

## Running

```bash
# Colab: run HAR_Preprocessing.ipynb first, then this notebook or
!python train_rnn.py --models lstm,bilstm,deepconvlstm

# local
HAR_PROJECT_DIR="C:/path/to/DL Proj" python train_rnn.py --models lstm
HAR_PROJECT_DIR="C:/path/to/DL Proj" python train_rnn.py --models bilstm --run-tag tuned
```

`--epochs` overrides the epoch budget, `--augmentation` switches to the shared
pre-augmented training set (`sequence_train_aug.npz`, 11,948 windows) for the tuned runs.

## Metric conventions used here

* `flops_per_window` is counted from the real weight shapes of the instantiated model
  (`rnn_models.count_flops`, GEMMs only, `2 x MACs`). `torch.utils.flop_counter` does not
  trace the fused ATen LSTM kernels — on `lstm` it reports 3,072 FLOPs, i.e. the classifier
  head only — so using it here would understate the RNN cost by ~4 orders of magnitude.
  The CNN/Transformer rows use the counter; both numbers are GEMM counts under the same
  convention.
* `latency_ms_cpu_b1` is measured with `hc.measure_latency` at batch size 1 after warm-up,
  on the host that produced the row. Rows trained on different hardware (CPU here vs Tesla T4
  for the CNN/Transformer) are **not** latency-comparable; each row says so in `notes`.
* `peak_gpu_mem_mb`, `latency_ms_gpu_b1` and `throughput_gpu_wps` are empty for these rows
  because they were produced on a CPU-only host. Re-running the same script on a Colab T4
  instance fills them in without changing anything else.

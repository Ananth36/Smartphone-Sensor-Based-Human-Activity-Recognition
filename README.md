# HAR model comparison: what every model must record

**ICT-4442 · Comparative Analysis of Deep Learning Architectures for Smartphone Sensor-Based HAR**

The Transformer notebook already produces everything listed here. If the MLP, 1D CNN and LSTM/BiLSTM
produce the **same files, under the same names, with the same columns**, the four models can be compared
directly in one table and one set of figures. Please follow this exactly: names included.

| Model | Owner | Folder | `model_name` | `family` | `input_representation` |
|---|---|---|---|---|---|
| Transformer encoder | K V Ananth Karantha | `transformer/` | `transformer` | `Transformer` | `raw_9ch_128` |
| 1D CNN | Akshaj Vinod Chandak | `cnn/` | `cnn` | `CNN` | `raw_9ch_128` |
| LSTM / BiLSTM | Nandan Reddy Maram | `lstm/` | `lstm` / `bilstm` | `RNN` | `raw_9ch_128` |
| MLP | Daasharathi Darahaas Yandam | `mlp/` | `mlp` | `MLP` | `tabular_561` |

---

## 1. Folder layout

```
DL PROJ/
├── 01_HAR_Preprocessing.ipynb   run ONCE (already done; do not re-run)
├── har_common.py                shared loader + metrics + results writer
├── Dataset/                     raw UCI files (never modified)
├── processed/                   the dataset everyone uses
│   ├── sequence.npz             (N, 128, 9)  → CNN, LSTM, Transformer
│   ├── tabular.npz              (N, 561)     → MLP
│   ├── sequence_train_augmented.npz   optional shared augmented train set
│   └── meta.json                contains dataset_version
├── transformer/   cnn/   lstm/   mlp/      one folder per model
```

Everything your notebook writes goes **inside your own folder**.

---

## 2. Protocol rules (must be the same for everyone)

1. Load data only from `processed/`. **Never refit scalers, re-split, or re-run preprocessing.**
2. Split: train 5,974 / validation 1,378 (subjects 3, 6, 17, 29) / test 2,947 (official 9 UCI test subjects).
3. Choose hyper-parameters and early-stop on **validation macro-F1**. Restore the best-epoch weights.
4. Evaluate on the **test set once**, at the very end, with the final model.
5. Seed **42**. Validation and test sets are **never augmented**.
6. Tuned runs: max **200 epochs**, early-stopping patience **≥ 20** (Transformer uses 30).
7. CPU latency: batch size 1, after warm-up, **1 CPU thread** (`hc.measure_latency` does this).
8. `run_tag = "baseline"` for the untuned model, `"tuned"` for the final tuned model.
9. Check that `dataset_version` in your results row matches `processed/meta.json` and everyone else's.

---

## 3. Loading the data

```python
from google.colab import drive; drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/DL PROJ')
import har_common as hc

d = hc.load_data('sequence')                       # CNN / LSTM: X is (N, 128, 9)
d = hc.load_data('sequence', channels_first=True)  # PyTorch Conv1d: (N, 9, 128)
d = hc.load_data('tabular')                        # MLP: X is (N, 561)

X_train, y_train = d['X_train'], d['y_train']      # labels are 0..5
X_val,   y_val   = d['X_val'],   d['y_val']
X_test,  y_test  = d['X_test'],  d['y_test']
subj_test        = d['subj_test']
```

Class order (do not change): `0 WALKING, 1 WALKING_UPSTAIRS, 2 WALKING_DOWNSTAIRS, 3 SITTING, 4 STANDING, 5 LAYING`.

---

## 4. Metrics: the results CSV

**File:** `<your folder>/<model_name>_results.csv` (e.g. `cnn/cnn_results.csv`): one row per run tag.

Do not compute these by hand. Get the class **probabilities** (softmax output, shape `(N, 6)`) for train,
validation and test, then call `hc.save_results`, which fills every column below in the same order for
everyone:

```python
P_tr, P_va, P_te = ...   # softmax probabilities from your FINAL (best-epoch) model

row = hc.save_results(
    '/content/drive/MyDrive/DL PROJ/cnn/cnn_results.csv',
    model_name='cnn', family='CNN', owner='Akshaj Vinod Chandak',
    input_representation='raw_9ch_128', run_tag='tuned',
    y_train=d['y_train'], proba_train=P_tr,
    y_val=d['y_val'],     proba_val=P_va,
    y_test=d['y_test'],   proba_test=P_te, subj_test=d['subj_test'],
    efficiency=dict(
        params_total=hc.count_params(model),
        model_size_mb=hc.file_size_mb('.../cnn/final_model/cnn_tuned_weights.pt'),
        flops_per_window=hc.count_flops(model, (1, 128, 9)),       # (1, 561) for the MLP
        latency_ms_cpu_b1=hc.measure_latency(model, (1, 128, 9), 'cpu'),
        latency_ms_gpu_b1=hc.measure_latency(model, (1, 128, 9), 'cuda'),
        throughput_gpu_wps=hc.measure_throughput(model, (128, 9), 256, 'cuda'),
        peak_gpu_mem_mb=peak_mem_mb,       # torch.cuda.max_memory_allocated()/2**20 after training
    ),
    training=dict(train_time_s=..., epochs_trained=..., best_epoch=..., time_per_epoch_s=...),
    notes='architecture + optimiser + LR schedule + augmentation + FLOPs tool used')
```

If your model expects channels-first input, use the input shape your model's `forward` takes, e.g. `(1, 9, 128)`.

**Keras / TensorFlow users:** `hc.save_results` works unchanged (it only needs numpy arrays).
`count_params`, `count_flops`, `measure_latency` are PyTorch-only, so measure these yourself
(`model.count_params()`, timed `model(x, training=False)` on batch 1 after 50 warm-up calls,
`tf.config.experimental.get_memory_info('GPU:0')['peak']`) and pass the numbers in the same dict.
Write `n/a` reasons in `notes` for anything you cannot measure.

### Columns (filled automatically)

| Group | Columns |
|---|---|
| Identity | `model_name, family, owner, input_representation, run_tag, dataset_version` |
| Test performance | `test_accuracy, test_macro_f1` (**primary ranking metric**)`, test_macro_precision, test_macro_recall, test_weighted_f1, test_balanced_accuracy, test_cohen_kappa, test_mcc, test_log_loss, test_roc_auc_macro_ovr, test_top2_accuracy` |
| Per class | `f1_WALKING, f1_WALKING_UPSTAIRS, f1_WALKING_DOWNSTAIRS, f1_SITTING, f1_STANDING, f1_LAYING` |
| Error analysis | `sit_stand_confusion_rate, walking_variants_confusion_rate, static_dynamic_confusion_rate` |
| Robustness | `subject_acc_mean, subject_acc_std, subject_acc_min` |
| Generalization | `val_accuracy, val_macro_f1, train_macro_f1, gap_train_test_f1` |
| Efficiency | `params_total, model_size_mb, flops_per_window, latency_ms_cpu_b1, latency_ms_gpu_b1, throughput_gpu_wps, peak_gpu_mem_mb` |
| Training cost | `train_time_s, epochs_trained, best_epoch, time_per_epoch_s` |
| Reproducibility | `hardware, framework, seed, notes` |

---

## 5. Other files to save (same names in every folder)

| File | What it contains | Why it is needed |
|---|---|---|
| `training_history_final.csv` | one row per epoch of the final model, columns below | learning curves of all models on one plot |
| `classification_report_test.txt` | `sklearn.metrics.classification_report(..., digits=4)` | per-class precision / recall |
| `per_subject_accuracy_test.csv` | columns `subject, accuracy` (9 rows) | subject-10-style robustness comparison |
| `final_model/predictions.npz` | `P_train, P_val, P_test, y_test, subj_test` | error overlap between models, ensembles |
| `final_model/<model>_tuned_weights.pt` (or `.keras`) | best-epoch weights | reproducibility; `model_size_mb` |
| `final_model/config.json` | every hyper-parameter of the final model | report / viva |
| `grid_search_results.csv` | one row per configuration tried: all hyper-parameters + `val_macro_f1, val_acc, best_epoch, epochs_trained, train_time_s` | shows the tuning was done on validation |
| `baseline_vs_tuned_test.csv` | your baseline vs tuned test metrics + change | before/after table |

### `training_history_final.csv` columns (exactly these)

```
epoch, train_loss, train_acc, train_macro_f1, val_loss, val_acc, val_macro_f1, lr, epoch_time_s
```

* `train_loss` / `val_loss`: **plain cross-entropy** (no label smoothing), computed in **eval mode**
  (no dropout, no augmentation) at the end of each epoch, so that losses are comparable across models.
* If you also log the loss you optimise (with augmentation / label smoothing), add it as an extra column
  `train_loss_optimised` (optional).
* `lr`: learning rate at the end of the epoch.

Saving it (PyTorch): collect one dict per epoch in a list `hist`, then
`pd.DataFrame(hist).to_csv('.../cnn/training_history_final.csv', index=False)`.
Keras: use a callback that evaluates train and val with `model.predict` each epoch, or at minimum rename
`history.history` keys to the names above (`loss→train_loss`, `val_loss`, `accuracy→train_acc`,
`val_accuracy→val_acc`) and compute `val_macro_f1` with a callback.

---

## 6. Figures (save to `<your folder>/figures/`, PNG, dpi 150)

| File name | Content | How |
|---|---|---|
| `learning_curves_final.png` | 4 panels: loss (train + val, **log y-axis**), accuracy, macro-F1, learning rate; dashed vertical line at the best epoch | from `training_history_final.csv` |
| `confusion_matrix_test.png` | counts + row-normalised recall, side by side | `hc.plot_confusion_matrix(y_test, P_te.argmax(1), path, 'Test')` |
| `confusion_matrix_val.png` | same, on validation | `hc.plot_confusion_matrix(y_val, P_va.argmax(1), path, 'Validation')` |
| `per_subject_and_per_class.png` | left: test accuracy per subject (bar, dashed mean line); right: test F1 per class | from the per-subject CSV / results row |
| `grid_search*.png` | validation macro-F1 of every configuration tried (bar chart or heat-map; several files are fine, e.g. the Transformer has `grid_stageA_architecture.png` and `grid_stageB_recipe.png`) | from `grid_search_results.csv` |
| `baseline_vs_tuned_curves.png` | validation loss and validation macro-F1 of the baseline vs the tuned run, same axes | from both history files |

Use `hc.plot_confusion_matrix` so the class order, abbreviations (`WALK, UP, DOWN, SIT, STAND, LAY`) and
colour scale are identical in every report figure.

Minimal learning-curve code (same layout as the Transformer's):

```python
import pandas as pd, matplotlib.pyplot as plt
h = pd.read_csv('.../cnn/training_history_final.csv'); best = int(h.val_macro_f1.idxmax()) + 1
fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
ax[0].plot(h.epoch, h.train_loss, label='train'); ax[0].plot(h.epoch, h.val_loss, label='validation')
ax[0].set_yscale('log'); ax[0].set_title('Cross-entropy loss')
ax[1].plot(h.epoch, h.train_acc, label='train'); ax[1].plot(h.epoch, h.val_acc, label='validation'); ax[1].set_title('Accuracy')
ax[2].plot(h.epoch, h.train_macro_f1, label='train'); ax[2].plot(h.epoch, h.val_macro_f1, label='validation'); ax[2].set_title('Macro-F1')
ax[3].plot(h.epoch, h.lr); ax[3].set_yscale('log'); ax[3].set_title('Learning rate')
for a in ax: a.axvline(best, ls='--', c='gray'); a.set_xlabel('epoch'); a.grid(alpha=.3)
for a in ax[:3]: a.legend()
fig.tight_layout(); fig.savefig('.../cnn/figures/learning_curves_final.png', dpi=150)
```

---

## 7. Hand-in checklist

- [ ] `<model_name>_results.csv` has a `baseline` row and a `tuned` row, all columns filled
- [ ] `dataset_version` matches `processed/meta.json`
- [ ] test set evaluated only once, with the best-validation-macro-F1 weights
- [ ] `training_history_final.csv` with the exact column names above
- [ ] `classification_report_test.txt`, `per_subject_accuracy_test.csv`, `final_model/predictions.npz`
- [ ] `grid_search_results.csv` with every configuration tried
- [ ] all six figures in `figures/` with the exact file names above
- [ ] `notes` column describes the architecture, optimiser, schedule, augmentation and FLOPs tool

When all four folders are complete, the comparison table is just:

```python
import pandas as pd, glob
rows = pd.concat([pd.read_csv(f) for f in glob.glob('/content/drive/MyDrive/DL PROJ/*/*_results.csv')])
rows[rows.run_tag == 'tuned'].sort_values('test_macro_f1', ascending=False)
```

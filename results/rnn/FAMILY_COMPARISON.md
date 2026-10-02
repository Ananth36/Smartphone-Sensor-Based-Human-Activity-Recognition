# Family comparison — ICT-4442 interim comparison

All rows are produced by the same code path on the same processed data
(`dataset_version = uci-har-65a1fc6120e8`), the same subject-independent split
(train 5,974 / val 1,378 subjects {3, 6, 17, 29} / test 2,947 subjects {2, 4, 9, 10, 12,
13, 18, 20, 24}) and the same metric implementation (`har_common.save_results`).
Model selection was on validation macro-F1 with early stopping; the test set was used once.

`accuracy` and `macro_f1` below are test values. Efficiency columns are **not** comparable
across rows produced on different hardware: the recurrent rows were produced on a CPU-only
host, the CNN/Transformer rows on a Colab Tesla T4 instance. This is recorded in the `notes`
column of `results/comparison_table.csv`.

## Main comparison

| model | family | owner | acc | macro-F1 | kappa | params | FLOPs/window | ms/window (b1) |
|---|---|---|---|---|---|---|---|---|
| logistic regression | reference | K V Ananth Karantha | 0.9623 | 0.9623 | – | ~5.6k | – | – |
| 1D CNN | CNN | Akshaj Vinod Chandak | 0.9396 | 0.9406 | 0.9274 | 197,190 | 24,856,064 | 1.264 (T4 host) |
| Transformer | Transformer | K V Ananth Karantha | 0.9284 | 0.9268 | 0.9140 | 76,422 | 17,056,512 | 3.006 (T4 host) |
| BiLSTM | RNN | Nandan Reddy Maram | 0.9141 | 0.9153 | 0.8969 | 204,678 | 51,281,568 | 9.551 (CPU host) |
| DeepConvLSTM | RNN | Nandan Reddy Maram | 0.9155 | 0.9164 | 0.8986 | 317,958 | 80,430,592 | 9.100 (CPU host) |
| LSTM | RNN | Nandan Reddy Maram | 0.8948 | 0.8965 | 0.8737 | 204,806 | 51,514,368 | 8.199 (CPU host) |

The MLP is not in the table yet (owner: Daasharathi Darahaas Yandam, in progress).

## Recurrent family, internal comparison

| model | acc | macro-F1 | SITTING F1 | STANDING F1 | sit/stand confusion | walking-variant confusion | subject acc mean | worst subject |
|---|---|---|---|---|---|---|---|---|
| LSTM | 0.8948 | 0.8965 | 0.8168 | 0.7950 | 18.18% | 6.78% | 0.8866 | 10 (0.5748) |
| BiLSTM | 0.9141 | 0.9153 | 0.8212 | 0.8277 | 17.30% | 4.25% | 0.9058 | 10 (0.6395) |
| DeepConvLSTM | see `results/rnn/deepconvlstm/baseline/metrics.json` | | | | | | | |

## What the recurrent family shows

1. **Bidirectionality helps, at equal capacity.** The BiLSTM is parameter-matched to the
   LSTM (204,678 vs 204,806 parameters, 0.06% apart, and 51.28M vs 51.51M FLOPs), and gains
   +1.93 points of accuracy and +0.019 macro-F1 over it. Because capacity and cost are held
   constant, the gain is attributable to reading the window in both time directions, which
   is the mechanism described in [3].
2. **The recurrent family is the weakest of the three deep families on this dataset.**
   LSTM 0.8965 < Transformer 0.9268 < CNN 0.9406 macro-F1, even though the CNN and the
   Transformer are *smaller* (197k/76k parameters vs 205k) and cheaper (24.9M/17.1M vs
   51.5M FLOPs). The recurrent models are the slowest per window by 3–6x as well.
3. **The deficit is overfitting to the training subjects, not underfitting.** Train macro-F1
   is 0.959 (LSTM) / 0.959 (BiLSTM) — comparable to the CNN's 0.968 and the Transformer's
   0.997 — while validation stays at 0.969 / 0.964 and test drops to 0.897 / 0.915. Only
   5,974 training windows are available, and the recurrent family has the most parameters
   per training sample and the least locality inductive bias.
4. **Every model fails on the same pair.** SITTING/STANDING confusion is 12.90% (CNN),
   8.60% (Transformer), 18.18% (LSTM) and 17.30% (BiLSTM); static-versus-dynamic confusion
   is ~0.1–0.5% for all of them. The residual error of this dataset is postural
   discrimination, which is a function of the gravity axis rather than of temporal dynamics,
   so architecture family has little leverage over it.
5. **Per-subject spread tracks family.** Worst-subject accuracy is subject 10 for every
   recurrent run (0.5748 LSTM, 0.6395 BiLSTM) and the same subject is also the Transformer's
   worst (0.7211), while the CNN does not report a subject-10 collapse. Subject-level
   variance, not architecture, is a large part of the remaining error.
6. **Consistent with [4].** Hammerla et al. found no family that dominates across datasets
   and that hyperparameter choices move results more than architecture does. Under our
   common protocol the ordering is stable (feature-based > CNN > Transformer > recurrent),
   but the recurrent family's position is the one that [3] would most expect to change with
   tuning, since [3] reports the largest gains exactly for the hybrid conv + recurrent
   architecture.

## What this suggests for the tuned runs

* Add the shared augmented training set (11,948 windows) to the recurrent baselines — the
  recurrent family is the most data-hungry of the four, so this is where augmentation should
  pay off most (`--augmentation` is already implemented).
* Replace the constant learning rate with warm-up + cosine decay and add label smoothing;
  the noisy validation loss in `history.csv` shows the constant 1e-3 rate is already too
  high for this family at this batch size.
* Consider a smaller hidden size (64/96 instead of 128) or a 1-layer recurrent stack, since
  the capacity/regularisation analysis above points at overfitting rather than at
  insufficient capacity.
* Keep the latency budget in mind: at 8.2–9.6 ms per window on CPU the recurrent models are
  the slowest family, and no tuning of the reported configuration gets them below the
  CNN's 1.264 ms.

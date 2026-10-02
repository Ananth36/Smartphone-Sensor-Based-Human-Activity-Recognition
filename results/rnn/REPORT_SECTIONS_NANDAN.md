# Report content for Nandan Reddy Maram (230911032)

Everything below is the text to paste into the interim report under this member's name,
together with the numbers actually produced by `models/rnn/baseline/train_rnn.py`.
Figures come from `results/rnn/<model>/baseline/`.

> **One correction for the whole team.** The processed-data fingerprint is
> `uci-har-65a1fc6120e8` (digit `one` in position 4). The interim report and the synopsis
> currently print `uci-har-65alfc6120e8` (letter `l`). `results/comparison_table.csv`
> already carries the correct value and the RNN rows reproduce it, so every number in the
> report is on one dataset version — but the report text should be corrected.

---

## 1. Section 3 — MODEL 4: LSTM / BiLSTM (replace the placeholder block)

```
MODEL 4: LSTM / BiLSTM
======================

Owner:
Nandan Reddy Maram

Registration No.:
230911032

Status:
Baseline complete for LSTM, BiLSTM and DeepConvLSTM; tuned runs scheduled for the
next milestone.

Papers assigned:
[3] Ordóñez & Roggen (2016) -> DeepConvLSTM
[4] Hammerla et al. (2016)   -> LSTM / BiLSTM family comparison

Input representation:
Raw sequential view of the shared processed dataset, shape = 128 x 9 per window.
The same sequence.npz, the same subject-independent split and the same
preprocessing as the CNN and Transformer models. No model-specific preprocessing.

Protocol (identical to the CNN and Transformer baselines):
- AdamW, learning rate = 1e-3, weight decay = 0.01
- Batch size = 64, gradient clipping = 1.0, constant learning rate
- No data augmentation, no class weights, no learning-rate schedule
- Dropout = 0.5 between recurrent layers and before the classifier
- Early stopping on validation Macro-F1, patience = 20, max 150 epochs
- Seed = 42; the test set is used once, after model selection
- All metrics computed by har_common.save_results

Architectures:
- LSTM        2 x LSTM(128), mean+max temporal pooling
              204,806 parameters, 51,514,368 FLOPs per window
- BiLSTM      2 x BiLSTM(78 per direction), mean+max temporal pooling
              204,678 parameters, 51,281,568 FLOPs per window
              (parameter-matched to the LSTM to 0.06% so that bidirectionality
              is compared at equal capacity, as required by [4])
- DeepConvLSTM  5 x Conv1D(64, kernel 5) + BatchNorm + ReLU, then 2 x LSTM(128),
              last-hidden-state classifier (Ordóñez & Roggen architecture)
              317,958 parameters, 80,430,592 FLOPs per window
```

---

## 2. Section 3A (new) — RECURRENT FAMILY RESULTS

```
RECURRENT FAMILY RESULTS (BASELINE, COMMON PROTOCOL)
===================================================

RECORD LSTM
-----------

Metric                    Train       Validation      Test

Accuracy                  95.56%      96.73%          89.48%

Macro-F1 (primary)        0.9592       0.9695          0.8965

Macro Precision           0.9607       0.9696          0.8985

Macro Recall              0.9598       0.9701          0.8977

Weighted F1               0.9556       0.9674          0.8952

Balanced Accuracy         0.9598       0.9701          0.8977

Cohen's Kappa / MCC       0.9466/      0.9607/         0.8737/
                           0.9471       0.9609          0.8742

Log Loss                  0.1034       0.0880          0.3794

ROC-AUC / Top-2 Accuracy  0.9956/      0.9972/         0.9850/
                           99.98%       100.0%          98.17%

Training:
- Epochs trained = 29
- Best epoch = 9 (selected on validation Macro-F1, 0.9695)
- Training time = 1182.2 s, 40.8 s per epoch
- Hardware: CPU (8 threads); no GPU was available for this run
```

(identical structure for the BiLSTM and DeepConvLSTM records)

---

## 3. LSTM ERROR ANALYSIS

```
LSTM ERROR ANALYSIS
===================

Strongest classes:
LAYING:
- F1 = 0.974
WALKING_DOWNSTAIRS:
- F1 = 0.940
WALKING:
- F1 = 0.935
WALKING_UPSTAIRS:
- F1 = 0.919

Weakest classes — the same two as every other model in this project:
SITTING:
- F1 = 0.817
STANDING:
- F1 = 0.795

Confusions:
- 116 SITTING windows classified as STANDING
- 70 STANDING windows classified as SITTING

SITTING/STANDING confusion rate:
- 18.18%   (1D CNN 12.90%, Transformer 8.60%)

Walking-variant confusion:
- 6.78%    (1D CNN 1.44%, Transformer 8.22%)

Static vs dynamic confusion:
- 0.10%    (Transformer 0.10%) — the recurrent model never mixes a posture
  class with a locomotion class, but it does mix the walking variants

Subject-level robustness:
- Mean per-test-subject accuracy = 88.66%
- Spread across subjects = 13.23% (largest of the three implemented families)
- Worst subject = 10, accuracy 57.48%
  (Transformer 72.11% on the same subject)
- Best subject = 13, accuracy 100.0%

Generalization:
- Train Macro-F1 0.9592 vs Test Macro-F1 0.8965
- Train-test Macro-F1 gap = 0.0627

Interpretation: the recurrent family memorises the training subjects'
movement style. Validation subjects (4 people) score 0.9695 while the 9 unseen
test subjects score 0.8965, and the whole deficit is concentrated in subject 10
and in the SITTING/STANDING pair. This is the behaviour predicted by [4]: the
recurrent family is the most hyper-parameter and data hungry of the three, and
UCI HAR gives it only 5,974 training windows.
```

---

## 4. BiLSTM RECORD AND LSTM vs BiLSTM

```
RECORD BiLSTM
-------------

Metric                    Train       Validation      Test

Accuracy                  95.48%      96.23%          91.41%

Macro-F1 (primary)        0.9585       0.9644          0.9153

Macro Precision           0.9602       0.9643          0.9157

Macro Recall              0.9590       0.9650          0.9171

Weighted F1               0.9547       0.9623          0.9144

Balanced Accuracy         0.9590       0.9650          0.9171

Cohen's Kappa / MCC       0.9456/      0.9546/         0.8969/
                           0.9462       0.9547          0.8973

Log Loss                  0.1114       0.0998          0.2224

ROC-AUC / Top-2 Accuracy  0.9955/      0.9969/         0.9923/
                           100.0%       100.0%          99.59%

Training:
- Epochs trained = 29
- Best epoch = 9 (selected on validation Macro-F1, 0.9644)
- Training time = 1264.4 s, 43.6 s per epoch
- Hardware: CPU (8 threads); no GPU was available for this run


LSTM vs BiLSTM (same split, same protocol, matched capacity)
===========================================================

                        LSTM              BiLSTM
Parameters              204,806           204,678      (-0.06%)
FLOPs per window        51,514,368        51,281,568   (-0.45%)
CPU latency b1          8.199 ms          9.551 ms
Test accuracy           89.48%            91.41%       (+1.93)
Test Macro-F1           0.8965            0.9153       (+0.019)
SITTING F1              0.8168            0.8212
STANDING F1             0.7950            0.8277       (+0.033)
Walking-variant conf.   6.78%             4.25%        (-2.53)
Worst subject (10)      57.48%            63.95%       (+6.47)
Subject acc std         13.23%            11.92%       (-1.31)
Train-test F1 gap       0.0627            0.0432       (-0.019)

The BiLSTM is parameter- and FLOP-matched to the LSTM on purpose. Because
capacity and per-window cost are held constant to within 0.5%, the +1.93 points of
accuracy and +0.019 macro-F1 cannot be attributed to a larger model: they come from
reading each 2.56 s window in both time directions, which is exactly the mechanism
[3] credits for the recurrent stage of DeepConvLSTM. Every error bucket moves in
the same direction - both static classes improve, the walking-variant confusion
halves, the worst subject recovers 6.5 points and the spread across subjects
shrinks - so the bidirectionality is buying generalisation on unseen subjects, not
a better fit to the training subjects.
```

---

## 5. BiLSTM ERROR ANALYSIS

```
BiLSTM ERROR ANALYSIS
====================

Strongest classes:
LAYING:          F1 = 0.987
WALKING:         F1 = 0.966
WALKING_UPSTAIRS F1 = 0.948
WALKING_DOWNSTAIRS F1 = 0.943

Weakest classes:
SITTING:         F1 = 0.821
STANDING:        F1 = 0.828

Confusions:
- 107 SITTING windows classified as STANDING
- 70 STANDING windows classified as SITTING
- SITTING/STANDING confusion rate = 17.30%

Walking-variant confusion = 4.25%
Static vs dynamic confusion = 0.51%

Subject-level robustness:
- Mean per-subject accuracy = 90.58%
- Spread = 11.92%
- Worst subject = 10, accuracy 63.95%
- Three subjects at 100% (18, 20) or 99.5% (24)

Generalization:
- Train Macro-F1 0.9585 vs Test Macro-F1 0.9153
- Train-test Macro-F1 gap = 0.0432
```

---

## 6. Efficiency and training cost (recurrent family)

```
RECURRENT FAMILY EFFICIENCY AND TRAINING COST
=============================================

LSTM              BiLSTM           DeepConvLSTM
Parameters           204,806           204,678          317,958
Model size (MB)      0.785             0.787            1.230
FLOPs per window     51,514,368        51,281,568       80,430,592
CPU latency b1 (ms)  8.199             9.551            9.100
CPU latency b1 p95   19.217            16.767           15.917
CPU throughput       1,346.3 w/s       1,006.2 w/s      1,082.2 w/s
Train time (s)       1,182.2           1,264.4          1,656.0
Epochs / best epoch  29 / 9            29 / 9           44 / 24
Time per epoch (s)   40.8              43.6             37.6
Best val Macro-F1    0.9695            0.9644           0.9667
Hardware             CPU, 8 threads    CPU, 8 threads   CPU, 8 threads

FLOPs are GEMM counts at 2 x MACs, taken from the real weight shapes
(torch.utils.flop_counter cannot see the fused ATen LSTM kernels - it reports
3,072 FLOPs for the LSTM, i.e. the classifier head only, so it would understate
this family by four orders of magnitude). GPU latency, GPU throughput and peak GPU
memory are empty for these rows because no GPU was available; re-running the same
script on a Colab T4 instance fills them in.

CAUTION for the comparison table: the CNN and Transformer latency and throughput
values were measured on a Colab Tesla T4 host, so they are NOT comparable with the
CPU numbers above. The efficiency columns of the four families are only directly
comparable in the parameter count and the FLOP count.
```

---

## 7. Family comparison (my contribution) — see `results/rnn/FAMILY_COMPARISON.md`

```
COMPARISON ACROSS ARCHITECTURE FAMILIES (updated with the recurrent rows)
=======================================================================

log-reg*    1D CNN     Transformer   DeepConvLSTM  BiLSTM    LSTM
Test accuracy     0.9623      0.9396     0.9284        0.9155        0.9141    0.8948
Test Macro-F1     0.9623      0.9406     0.9268        0.9164        0.9153    0.8965
Parameters        ~5.6k       197,190    76,422        317,958       204,678   204,806
FLOPs/window      -           24,856,064 17,056,512    80,430,592    51,281,568 51,514,368
CPU latency b1    -           1.264 ms   3.006 ms      9.100 ms      9.551 ms  8.199 ms
(* reference model, not one of the four; latency/FLOPs are not comparable across
 different hosts, see the caution note above)

Findings:
1. On UCI HAR the ranking is feature-based > CNN > Transformer > recurrent, even
   though the recurrent models are the largest and the most expensive of the deep
   models. This is a data-volume effect: 5,974 training windows for ~205k recurrent
   parameters, and the 561 UCI features already encode the discriminative statistics.
2. Bidirectionality is worth +1.93 points of accuracy at equal capacity, so the
   recurrent family is not broken - it is under-regularised for this dataset size.
3. All four families share the same dominant error, SITTING vs STANDING
   (12.90% CNN, 8.60% Transformer, 17.30% BiLSTM, 18.18% LSTM) and essentially zero
   static-vs-dynamic confusion. The residual error is postural discrimination along
   the gravity axis, which architecture family has little leverage over.
4. Subject 10 is the worst subject for every recurrent run and for the Transformer,
   so per-subject variance is a bigger lever than the remaining architectural
   differences. Consistent with [4]'s recommendation to report spread rather than a
   single number.
5. On the accuracy/cost trade-off the CNN is the Pareto choice at this dataset size:
   it beats both recurrent models while using 3x fewer FLOPs per window. The
   recurrent family only becomes competitive if the augmented training set closes
   its generalisation gap, which is the next planned run.
6. Within the recurrent family, adding the convolutional front-end of [3] buys
   almost nothing on this dataset: DeepConvLSTM reaches Macro-F1 0.9164 against the
   BiLSTM's 0.9153, a difference of 0.001, while costing 55% more parameters and 57%
   more FLOPs per window and 31% more training time. [3] measured its gains on
   OPPORTUNITY and Skoda, where the input is multimodal and the label set has 17-18
   gesture classes; on UCI HAR's six classes and a waist-mounted accelerometer with
   only 5,974 training windows the convolutional stem is redundant. Its strongest
   per-class result is WALKING at F1 0.985, the best of any model in the project,
   which suggests the convolutional filters do help the locomotion classes
   specifically - the classes where temporal patterns matter most.
```

---

## 7b. DeepConvLSTM record and error analysis (reproduce of [3])

```
RECORD DeepConvLSTM
-------------------

Architecture (Ordóñez & Roggen, 2016):
- 5 x [Conv1D(64 filters, kernel 5, stride 1, 'same') + BatchNorm + ReLU]
- 2 x LSTM(128), dropout 0.5
- Classifier on the last hidden state (the pooling used in [3])

Metric                    Train       Validation      Test

Accuracy                  95.75%      96.44%          91.55%

Macro-F1 (primary)        0.9609       0.9667          0.9164

Macro Precision           0.9622       0.9666          0.9161

Macro Recall              0.9614       0.9672          0.9183

Weighted F1               0.9574       0.9644          0.9155

Balanced Accuracy         0.9614       0.9672          0.9183

Cohen's Kappa / MCC       0.9488/      0.9572/         0.8986/
                           0.9493       0.9573          0.8988

Log Loss                  0.1008       0.0878          0.3620

ROC-AUC / Top-2 Accuracy  0.9956/      0.9973/         0.9857/
                           100.0%       99.93%          98.13%

Training:
- Epochs trained = 44
- Best epoch = 24 (selected on validation Macro-F1, 0.9667 - the best validation
  score of the three recurrent models)
- Training time = 1656.0 s, 37.6 s per epoch

Per-class F1:
WALKING 0.985 | WALKING_UPSTAIRS 0.925 | WALKING_DOWNSTAIRS 0.949
SITTING 0.827 | STANDING 0.839 | LAYING 0.974

Error analysis:
- 100 SITTING -> STANDING, 63 STANDING -> SITTING
- SITTING/STANDING confusion rate = 15.93%   (best of the three recurrent models)
- Walking-variant confusion = 3.24%           (best of the three recurrent models)
- Static vs dynamic confusion = 1.39%        (worst of the three recurrent models)
- Subject 10 again the worst subject: 62.93%; mean 90.71%, spread 12.15%
- Train-test Macro-F1 gap = 0.0445

Reading of the result: the convolutional stem of [3] gives the best validation
Macro-F1 of my three models (0.9667 vs 0.9695 / 0.9644) but essentially the same test
Macro-F1 as the plain BiLSTM, so the extra capacity buys fit, not generalisation. It
is the best recurrent model on the two error buckets that matter here (SITTING/STANDING
15.93%, walking variants 3.24%) and the worst on static/dynamic confusion, and it is
noticeably less well calibrated (log loss 0.362 vs 0.222 for the BiLSTM).
```

---

## 8. Section 4 — INDIVIDUAL CONTRIBUTION LOG (replace this member's entry)

```
Nandan Reddy Maram
Registration No.:
230911032

Tasks completed:
- Literature review:
  - [3] Ordóñez & Roggen (2016), DeepConvLSTM - method, results, relevance
    to this project, and the deviation of our instantiation from the paper
  - [4] Hammerla et al. (2016), DNN vs CNN vs (Bi)LSTM family comparison - method,
    results, relevance, and the parameter-matched control it motivates
- Model 4 implementation (recurrent family), all three variants:
  - LSTM            - 2 x LSTM(128), mean+max temporal pooling
  - BiLSTM          - 2 x BiLSTM(78 per direction), parameter-matched to the LSTM
  - DeepConvLSTM    - 5 x Conv1D(64, k=5) + BN + ReLU then 2 x LSTM(128),
                      the architecture of [3]
- Shared protocol compliance: same processed data, same subject-independent split,
  early stopping on validation Macro-F1, single test evaluation, seed 42,
  all metrics through har_common.save_results
- Efficiency measurement for the recurrent family, including a shape-based FLOP
  counter because torch.utils.flop_counter cannot trace the fused ATen LSTM kernels
- TorchScript export of each recurrent model (raw window -> probabilities, with
  clipping and z-scoring folded in) and model_config.json for each run
- Recurrent family error analysis and the cross-family comparison
  (results/rnn/FAMILY_COMPARISON.md), including the parameter-matched LSTM/BiLSTM
  control that [4] shows is necessary
- Recurrence-family tuned-run plan for the next milestone

Branch: nandan-reddy-maram
Files added:
  models/rnn/baseline/{rnn_models.py, train_rnn.py, RNN_Baseline.ipynb, README.md}
  results/rnn/{lstm, bilstm, deepconvlstm}/baseline/*
  results/rnn/{FAMILY_COMPARISON.md, LITERATURE_REVIEW_3_4.md, REPORT_SECTIONS_NANDAN.md}
  rows appended to results/comparison_table.csv
```

---

## 9. Figures and files to attach under this member's name

```
Figure 5.  LSTM baseline, test confusion matrix (counts and recall).
           -> results/rnn/lstm/baseline/confusion_matrix.png
Figure 6.  LSTM baseline learning curves (train/val loss, train/val accuracy,
           validation Macro-F1, dashed line at the best epoch = 9).
           -> results/rnn/lstm/baseline/learning_curves.png
Figure 7.  BiLSTM baseline, test confusion matrix (counts and recall).
           -> results/rnn/bilstm/baseline/confusion_matrix.png
Figure 8.  BiLSTM baseline learning curves (best epoch = 9).
           -> results/rnn/bilstm/baseline/learning_curves.png
Figure 9.  DeepConvLSTM baseline, test confusion matrix (counts and recall).
           -> results/rnn/deepconvlstm/baseline/confusion_matrix.png
Figure 10. DeepConvLSTM baseline learning curves (best epoch = 24).
           -> results/rnn/deepconvlstm/baseline/learning_curves.png
Figure 11. Family comparison: test Macro-F1 against FLOPs per window and against
           parameters, for the four families under the common protocol.
           -> results/rnn/FAMILY_COMPARISON.md (table in this interim version)

Supporting files per model: metrics.json (every column of
METRICS_TO_REPORT.md), classification_report.txt, history.csv
(per-epoch train/val loss, accuracy, Macro-F1, learning rate, epoch time).
```

---

## 10. Remaining work owned by this member (for section 5)

```
RECURRENT FAMILY - REMAINING WORK (owner: Nandan Reddy Maram)
==========================================================

1. Tuned runs (run_tag = "tuned") of LSTM, BiLSTM and DeepConvLSTM:
   - shared augmented training set (11,948 windows) - the recurrent family is the
     most data hungry, so this is the highest-value single change
   - warm-up + cosine learning-rate schedule; the constant 1e-3 rate produces noisy
     validation loss in every recurrent history.csv
   - label smoothing 0.05 and dropout 0.3-0.4 instead of 0.5
   - hidden size 64/96 and one recurrent layer, since the capacity analysis points at
     overfitting rather than insufficient capacity
   - early stopping and selection stay on validation Macro-F1, test untouched

2. Re-run the same three scripts on the Colab Tesla T4 instance so that
   latency_ms_gpu_b1, throughput_gpu_wps and peak_gpu_mem_mb are filled in and the
   efficiency columns become comparable with the CNN and Transformer rows.

3. Flag for the team: the CNN and Transformer latency numbers were measured on the
   Colab T4 host while the recurrent rows were produced on a CPU-only host. Until the
   recurrent rows are re-run on the same host, the accuracy-efficiency trade-off
   section must compare parameters and FLOPs only, and must not compare latency
   across families. This is stated in the notes column of every recurrent row.

4. Optional: 5-fold grouped cross-validation over the training subjects, to check
   whether the family ordering survives the small 4-subject validation split (the
   optimistic-model-selection risk listed in the report).
```

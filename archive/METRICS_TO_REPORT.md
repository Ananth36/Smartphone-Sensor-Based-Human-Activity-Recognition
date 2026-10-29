# Metrics every model must report

Dataset version: see `DL Proj/processed/meta.json` → `dataset_version`. Every row in the comparison table must show the same value.

Use `har_common.save_results(...)` and all of these are filled in automatically, except the efficiency / training-cost numbers, which you pass in.


## Identity

| Column | Meaning |
|---|---|
| `model_name` | Short id: mlp / cnn / lstm / bilstm / transformer |
| `family` | MLP, CNN, RNN or Transformer |
| `owner` | Team member who ran it |
| `input_representation` | tabular_561 or raw_9ch_128 |
| `run_tag` | baseline / tuned / augmented ... |
| `dataset_version` | Fingerprint of the processed data; must be identical across all models |

## Test performance

| Column | Meaning |
|---|---|
| `test_accuracy` | Overall accuracy on the 9 held-out test subjects |
| `test_macro_f1` | PRIMARY ranking metric: unweighted mean F1 over the 6 classes |
| `test_macro_precision` | Unweighted mean precision over classes |
| `test_macro_recall` | Unweighted mean recall over classes |
| `test_weighted_f1` | Support-weighted F1 |
| `test_balanced_accuracy` | Mean per-class recall |
| `test_cohen_kappa` | Agreement corrected for chance |
| `test_mcc` | Matthews correlation coefficient (multiclass) |
| `test_log_loss` | Cross-entropy of predicted probabilities (calibration) |
| `test_roc_auc_macro_ovr` | Macro one-vs-rest ROC-AUC |
| `test_top2_accuracy` | True class within the top-2 predictions |

## Per-class

| Column | Meaning |
|---|---|
| `f1_WALKING` | Test F1 for WALKING |
| `f1_WALKING_UPSTAIRS` | Test F1 for WALKING_UPSTAIRS |
| `f1_WALKING_DOWNSTAIRS` | Test F1 for WALKING_DOWNSTAIRS |
| `f1_SITTING` | Test F1 for SITTING |
| `f1_STANDING` | Test F1 for STANDING |
| `f1_LAYING` | Test F1 for LAYING |

## Error analysis

| Column | Meaning |
|---|---|
| `sit_stand_confusion_rate` | Share of SITTING/STANDING windows predicted as the other one |
| `walking_variants_confusion_rate` | Share of walking windows confused among walk / upstairs / downstairs |
| `static_dynamic_confusion_rate` | Share of all windows where a static activity was confused with a moving one or vice versa |

## Robustness

| Column | Meaning |
|---|---|
| `subject_acc_mean` | Mean of per-test-subject accuracy |
| `subject_acc_std` | Spread of accuracy across test subjects (lower = more consistent) |
| `subject_acc_min` | Accuracy on the worst test subject |

## Generalization

| Column | Meaning |
|---|---|
| `val_accuracy` | Accuracy on the validation subjects |
| `val_macro_f1` | Macro-F1 on validation (used for model selection / early stopping) |
| `train_macro_f1` | Macro-F1 on the training set |
| `gap_train_test_f1` | train_macro_f1 - test_macro_f1 (overfitting indicator) |

## Efficiency

| Column | Meaning |
|---|---|
| `params_total` | Total number of parameters |
| `model_size_mb` | Size on disk of the saved weights file |
| `flops_per_window` | FLOPs for one 2.56 s window (state the tool used in notes) |
| `latency_ms_cpu_b1` | Mean CPU inference time for ONE window, batch size 1 (deployment proxy) |
| `latency_ms_gpu_b1` | Mean GPU inference time for one window |
| `throughput_gpu_wps` | Windows/second on GPU at batch 256 |
| `peak_gpu_mem_mb` | Peak GPU memory during training |

## Training cost

| Column | Meaning |
|---|---|
| `train_time_s` | Wall-clock training time |
| `epochs_trained` | Epochs actually run (after early stopping) |
| `best_epoch` | Epoch with best val macro-F1 |
| `time_per_epoch_s` | Mean seconds per epoch |

## Reproducibility

| Column | Meaning |
|---|---|
| `hardware` | GPU / CPU used |
| `framework` | e.g. PyTorch 2.x, TensorFlow 2.x, scikit-learn |
| `seed` | Random seed |
| `notes` | Anything a reader needs to interpret the row |

## Protocol rules

1. Use the processed data as saved. Never refit scalers or re-split.
2. Select hyperparameters / early-stop on **validation macro-F1**. Evaluate on test once, at the end.
3. Seed 42. Same test subjects for everyone (UCI official test split).
4. Measure CPU latency with batch size 1, after warm-up (`hc.measure_latency`).
5. Baseline runs use `run_tag='baseline'`; tuned runs use `run_tag='tuned'`.

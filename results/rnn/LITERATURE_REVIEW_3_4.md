# Literature review — papers assigned to Nandan Reddy Maram (230911032)

Both papers are assigned to this member because they are the recurrent-family
references for Model 4: [3] is the architecture that `deepconvlstm` reproduces, and [4]
is the family comparison under a fixed protocol that this project reproduces.

---

## [3] F. J. Ordóñez and D. Roggen, "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition," *Sensors*, vol. 16, no. 1, 115, 2016.

**Method.** A hybrid end-to-end network that consumes raw inertial signals without any
hand-crafted feature extraction. A stack of temporal 1-D convolutional layers (64 feature
maps per layer, kernel size 5, stride 1) acts as a learned feature extractor over the
time axis; its output sequence is fed to two stacked LSTM layers (64 hidden units,
dropout 0.5) which model longer-range temporal dependencies; the final LSTM state is
classified by a fully connected softmax layer. Convolutions are applied per sensing
modality and the modality-specific outputs are fused before the recurrent stage.

**Dataset.** OPPORTUNITY and Skoda Mini Checkpoint — real, multimodal recordings (body-worn
accelerometers, object-worn sensors on carried items, and in-vehicle location) of complex
everyday gestures rather than the six locomotion/posture classes of UCI HAR.

**Key result.** The hybrid beats purely feed-forward deep networks by roughly 4% on
average on OPPORTUNITY and improves on prior published results by up to 9%. Ablations in the
paper show that either half alone is worse than the combination: the convolutional part
learns local, short-range motion patterns, and the recurrent part adds the temporal context
that a fixed-length convolutional receptive field cannot represent, while the recurrent
part alone is harder to optimise because it must learn the feature extraction as well.

**Relevance to this project.**
1. `deepconvlstm` in Model 4 is a direct re-implementation of this architecture, so [3] is
   the reference against which our recurrent model's design choices are justified.
2. It establishes the raw-signal (no hand-crafted features) setting that this project
   contrasts with the feature-based MLP.
3. Its reported gains come from a different dataset, modality fusion and split, so the
   numbers cannot be compared with ours directly — the limitation this project addresses by
   holding data, split and evaluation fixed across all four families.
4. Being a 2016 pre-attention architecture, it also provides the baseline that the
   attention/Transformer references [5], [9], [10] claim to improve upon.

**Deviation from the paper (stated for honesty).** Our instantiation uses 5 convolutional
layers and 128-unit LSTM layers on the 9 inertial channels only (the paper uses 4
convolutional and 64-unit LSTM layers with multimodal fusion), and we keep the paper's
last-hidden-state classification instead of adding pooling. The depth/width changes were
needed to sit in the same parameter range as our LSTM and BiLSTM so that the three
recurrent variants remain comparable to each other.

---

## [4] N. Y. Hammerla, S. Halloran, and T. Plötz, "Deep, Convolutional, and Recurrent Models for Human Activity Recognition using Wearables," *IJCAI*, 2016, pp. 1533–1540.

**Method.** A large-scale, controlled comparison of the three dominant architecture
families — fully connected DNN, CNN, and (Bi)LSTM — trained over thousands of randomly
sampled hyperparameter configurations rather than a single hand-picked configuration. The
three families are given the same input representation (a fixed-length window of raw
sensor channels) and the same train/validation/test protocol within each dataset.

**Dataset.** OPPORTUNITY, PAMAP2 and Daphnet Free Fall — three different sensing setups
with different numbers of channels, sampling rates, activities and label sets.

**Key result.** No single architecture wins across datasets: BiLSTM is strongest on
OPPORTUNITY, CNN is strongest on Daphnet Free Fall, and a well-tuned DNN reaches a mean F1
of about 0.937 on PAMAP2. The paper's central methodological conclusion is that
hyperparameter choice produces differences *larger* than the differences between the
architecture families themselves, so single-configuration comparisons between families are
not informative.

**Relevance to this project.**
1. [4] is the direct justification for this project's premise. If hyperparameter noise can
   exceed family differences, then a fair family comparison requires one fixed protocol,
   one split, one metric implementation and a documented, identical configuration for every
   model — which is exactly what `har_common.py` and `METRICS_TO_REPORT.md` enforce.
2. "No single winner" is the expected outcome of this project, so the report frames the
   comparison as a trade-off between accuracy, robustness and inference cost rather than as
   a search for a single best model.
3. The LSTM and BiLSTM of Model 4 follow the recurrent family as instantiated in [4]. Since
   [4] found the family highly hyper-parameter sensitive, we deliberately use one fixed,
   documented configuration per variant rather than a search, and we match the parameter
   count of the BiLSTM to the LSTM (204,678 vs 204,806) so that bidirectionality is not
   confounded with capacity.
4. The paper recommends reporting variability across subjects/datasets rather than a single
   number, which is why every row in our comparison table includes per-subject mean, spread
   and worst-subject accuracy.

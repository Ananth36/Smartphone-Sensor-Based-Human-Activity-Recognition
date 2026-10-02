"""Recurrent architectures for the ICT-4442 UCI HAR comparison (owner: Nandan Reddy Maram).

Model 4 of the project is the recurrent family. Three variants are implemented:

    lstm          plain 2-layer LSTM over the raw 9-channel window      -> [4] Hammerla et al.
    bilstm        bidirectional counterpart of the above, parameter-matched to `lstm`
    deepconvlstm  5x Conv1D feature extractor + 2x LSTM                 -> [3] Ordonez & Roggen

All three consume the shared processed sequence view (N, 128, 9) produced by
HAR_Preprocessing.ipynb and are trained under the protocol in METRICS_TO_REPORT.md,
so their rows are directly comparable with the MLP / CNN / Transformer rows.

Only the architecture differs between the variants: the pooling head, optimiser,
schedule, seed, early-stopping rule and evaluation code are shared, which is what
makes the family comparison in results/rnn/FAMILY_COMPARISON.md meaningful.
"""
import torch
import torch.nn as nn


class _RNNHead(nn.Module):
    """Shared classification head.

    pooling="mean_max" averages and max-pools the last RNN layer over time, then
    classifies the concatenated vector. pooling="last" uses only the final
    timestep, which is what Ordonez & Roggen use for DeepConvLSTM.
    """

    def __init__(self, hidden, n_classes, dropout=0.5, pooling="mean_max"):
        super().__init__()
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if pooling == "mean_max" else 1), n_classes)

    def forward(self, out):
        # out: (B, T, H) output of the last RNN layer
        h = out[:, -1] if self.pooling == "last" else torch.cat([out.mean(1), out.max(1).values], dim=-1)
        return self.head(self.drop(h))


class HARLSTM(nn.Module):
    """Stacked unidirectional LSTM on raw inertial windows (Hammerla et al. family)."""

    def __init__(self, n_channels=9, seq_len=128, n_classes=6, hidden=128, n_layers=2,
                 dropout=0.5, pooling="mean_max", **_):
        super().__init__()
        self.rnn = nn.LSTM(n_channels, hidden, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.head = _RNNHead(hidden, n_classes, dropout, pooling)

    def forward(self, x):                      # x: (B, T, C) normalized signals
        out, _ = self.rnn(x)
        return self.head(out)                   # logits


class HARBiLSTM(nn.Module):
    """Bidirectional counterpart of HARLSTM.

    `hidden` is per direction. The default 78 is not arbitrary: it makes the model
    parameter-matched to HARLSTM (204,678 vs 204,806 parameters, 0.06% apart), so the
    comparison isolates bidirectionality instead of extra capacity.
    """

    def __init__(self, n_channels=9, seq_len=128, n_classes=6, hidden=78, n_layers=2,
                 dropout=0.5, pooling="mean_max", **_):
        super().__init__()
        self.rnn = nn.LSTM(n_channels, hidden, num_layers=n_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = _RNNHead(hidden * 2, n_classes, dropout, pooling)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out)


class DeepConvLSTM(nn.Module):
    """Ordinalez & Roggen (2016): temporal CNN feature extractor followed by 2 LSTM layers.

    5 x Conv1D(64, kernel 5, 'same') each followed by BatchNorm + ReLU, then two
    128-unit LSTM layers with 50% dropout, then a fully connected classifier.
    """

    def __init__(self, n_channels=9, seq_len=128, n_classes=6, n_filters=64, kernel_size=5,
                 hidden=128, n_layers=2, dropout=0.5, pooling="last", **_):
        super().__init__()
        convs, in_ch = [], n_channels
        for _ in range(5):
            convs += [nn.Conv1d(in_ch, n_filters, kernel_size, padding=kernel_size // 2),
                      nn.BatchNorm1d(n_filters), nn.ReLU()]
            in_ch = n_filters
        self.convs = nn.Sequential(*convs)
        self.rnn = nn.LSTM(n_filters, hidden, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.head = _RNNHead(hidden, n_classes, dropout, pooling)

    def forward(self, x):
        h = self.convs(x.transpose(1, 2)).transpose(1, 2)     # (B, T, F)
        out, _ = self.rnn(h)
        return self.head(out)


# (class, default hyperparameters) for each model id used in results/comparison_table.csv
MODELS = {
    "lstm":         (HARLSTM,     dict(hidden=128, n_layers=2, dropout=0.5, pooling="mean_max")),
    "bilstm":       (HARBiLSTM,   dict(hidden=78,  n_layers=2, dropout=0.5, pooling="mean_max")),
    "deepconvlstm": (DeepConvLSTM, dict(hidden=128, n_layers=2, dropout=0.5, pooling="last")),
}
FAMILY = "RNN"
OWNER = "Nandan Reddy Maram"


def build(model_name, n_channels, seq_len, n_classes, **hp):
    if model_name not in MODELS:
        raise KeyError(f"unknown model '{model_name}'; choose from {sorted(MODELS)}")
    cls, defaults = MODELS[model_name]
    merged = {**defaults, **hp}
    return cls(n_channels=n_channels, seq_len=seq_len, n_classes=n_classes, **merged), merged


def count_flops(model, seq_len=128):
    """FLOPs for one window, counted from the real weight shapes of the instantiated model.

    torch.utils.flop_counter cannot see the fused ATen LSTM kernels (it reports only the
    classifier head for a pure LSTM), so the recurrent part is counted here: every weight
    matmul is 2 x MACs, which is the convention used for the CNN and Transformer rows.

    nn.Linear   -> in_features * out_features
    nn.Conv1d   -> out_length * out_channels * (in_channels / groups) * kernel_size
    nn.LSTM     -> seq_len * sum over layers and directions of
                   4H * input_size + 4H * H   (weight matmuls only, as torch counts them)
    """
    macs = 0
    shapes = dict(model.named_modules())
    for name, layer in shapes.items():
        if isinstance(layer, nn.Linear):
            macs += layer.in_features * layer.out_features
        elif isinstance(layer, nn.Conv1d):
            out_len = seq_len + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1
            out_len = out_len // layer.stride[0] + 1
            macs += out_len * layer.out_channels * (layer.in_channels // layer.groups) * layer.kernel_size[0]
        elif isinstance(layer, nn.LSTM):
            directions = 2 if layer.bidirectional else 1
            for k in range(layer.num_layers):
                hidden = layer.hidden_size
                in_size = layer.input_size if k == 0 else hidden * (directions if layer.bidirectional else 1)
                macs += seq_len * directions * (4 * hidden * in_size + 4 * hidden * hidden)
    return int(2 * macs)

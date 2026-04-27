from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 64,
        num_classes: int = 4,
        rnn_type: str = "rnn",
        num_layers: int = 1,
        bidirectional: bool = False,
        pretrained_emb: Optional[torch.Tensor] = None,
        freeze_emb: bool = False,
        pad_idx: int = 0,
    ):
        super().__init__()
        if rnn_type not in {"rnn", "lstm"}:
            raise ValueError(f"rnn_type must be 'rnn' or 'lstm', got {rnn_type!r}")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        if pretrained_emb is not None:
            assert pretrained_emb.shape == (vocab_size, embedding_dim), (
                f"pretrained_emb shape {pretrained_emb.shape} "
                f"!= ({vocab_size}, {embedding_dim})"
            )
            self.embedding.weight.data.copy_(pretrained_emb)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        rnn_cls = nn.RNN if rnn_type == "rnn" else nn.LSTM
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        head_in = hidden_dim * (2 if bidirectional else 1)
        self.linear = nn.Linear(head_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq)
        emb = self.embedding(x)             # (batch, seq, emb)
        out, hidden = self.rnn(emb)
        # For LSTM, hidden is a tuple (h_n, c_n); we want h_n.
        h_n = hidden[0] if self.rnn_type == "lstm" else hidden
        # h_n: (num_layers * num_directions, batch, hidden)
        if self.bidirectional:
            # last layer's two directions are at positions [-2] (forward) and [-1] (backward)
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            summary = torch.cat([forward_last, backward_last], dim=1)
        else:
            summary = h_n[-1]
        logits = self.linear(summary)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

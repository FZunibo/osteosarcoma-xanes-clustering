"""Autoencoder architecture used for low-dimensional XANES representations."""

from __future__ import annotations

import torch
from torch import nn


class SpectralAutoencoder(nn.Module):
    """1D convolutional autoencoder with a configurable 2D bottleneck.

    This is a compact extraction of the autoencoder architecture used in the study.
    The input is expected to have shape ``(batch, channels, spectral_length)``.
    ``siglen`` must match the spectral length supplied to the model.
    """

    def __init__(
        self,
        bottleneck_size: int = 2,
        n_layers: int = 4,
        activation: nn.Module | None = None,
        siglen: int = 345,
        n_signals: int = 1,
        h_dims: int = 64,
        final_conv_dim: int = 8,
        kernel_size: int = 16,
        stride: int | None = None,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.n_signals = n_signals
        self.n_layers = n_layers
        self.bottleneck_size = bottleneck_size
        self.activation = activation if activation is not None else nn.SiLU()
        self.siglen = siglen
        self.stride_val = stride if stride is not None else (4 if n_layers < 3 else 2)

        self.conv_hdims = [n_signals] + [h_dims] * (n_layers - 1) + [final_conv_dim]
        self.deconv_hdims = [final_conv_dim] + [h_dims] * (n_layers - 1) + [final_conv_dim]
        self.conv_kernelsizes = [kernel_size] * n_layers
        self.conv_strides = [self.stride_val] * n_layers

        self.output_paddings: list[int] = []
        length = siglen
        for i in range(len(self.conv_hdims) - 1):
            kernel = self.conv_kernelsizes[i]
            step = self.conv_strides[i]
            self.output_paddings.append((length - kernel) % step)
            length = ((length - kernel) // step) + 1
            if length < 1:
                raise ValueError(
                    "The requested convolutional configuration collapses the spectral length."
                )

        self.flattened_size = self.conv_hdims[-1] * length
        decoder_paddings = self.output_paddings[::-1]

        self.conv_encoder = nn.ModuleList(
            [
                nn.Conv1d(
                    self.conv_hdims[i],
                    self.conv_hdims[i + 1],
                    self.conv_kernelsizes[i],
                    self.conv_strides[i],
                )
                for i in range(len(self.conv_hdims) - 1)
            ]
        )

        self.encoder_linear = nn.Linear(self.flattened_size, bottleneck_size)
        self.decoder_linear = nn.Linear(bottleneck_size, self.flattened_size)

        self.conv_decoder = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    self.deconv_hdims[i],
                    self.deconv_hdims[i + 1],
                    self.conv_kernelsizes[i],
                    self.conv_strides[i],
                    output_padding=decoder_paddings[i],
                )
                for i in range(len(self.deconv_hdims) - 1)
            ]
            + [nn.Conv1d(self.deconv_hdims[-1], n_signals, kernel_size=1)]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for conv in self.conv_encoder:
            y = self.activation(conv(y))
        y = y.reshape(x.shape[0], -1)
        return self.activation(self.encoder_linear(y))

    def decode(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        y = self.activation(self.decoder_linear(z))
        y = y.reshape(z.shape[0], self.conv_hdims[-1], -1)
        for deconv in self.conv_decoder:
            y = self.activation(deconv(y))
        if output_length is not None:
            y = y[:, :, :output_length]
        return y

    def forward(
        self, x: torch.Tensor, return_code: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        decoded = self.decode(encoded, output_length=x.shape[2])
        if return_code:
            return encoded, decoded
        return decoded

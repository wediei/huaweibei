"""Complex Beam-Delay codecs adapted from HOSVD/BTD factorization ideas.

The reference BTD code operates on synthetic non-negative 3-D radio maps. This
module reimplements only the HOSVD subspace idea for complex five-mode
MIMO-OFDM tensors and bounded batches; it does not depend on TensorLy or the
reference training loops.
"""

from __future__ import annotations

import numpy as np

from .transforms import AntennaLayout, beam_delay, inverse_beam_delay


def _mode_product(
    tensor: np.ndarray,
    matrix: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Multiply ``matrix[new, old]`` into one tensor axis."""

    result = np.tensordot(matrix, tensor, axes=(1, axis))
    return np.moveaxis(result, 0, axis)


class SharedTuckerCodec:
    """Shared complex Tucker subspaces fitted only from selected anchors.

    Modes are ordered ``(H, V, P, N, S)`` after the configured Beam-Delay
    transform. The batch dimension is never factorized.
    """

    def __init__(self, layout: AntennaLayout, ranks: tuple[int, ...]) -> None:
        self.layout = layout
        self.ranks = tuple(int(rank) for rank in ranks)
        dimensions = layout.structured_tail
        if len(self.ranks) != len(dimensions) or any(
            rank < 1 or rank > dimension
            for rank, dimension in zip(self.ranks, dimensions)
        ):
            raise ValueError(
                f"rank tuple must contain five values within mode dimensions "
                f"{dimensions}, got {self.ranks}"
            )

    def _check_fitted(self) -> None:
        if not hasattr(self, "factors"):
            raise RuntimeError("codec must be fitted before encode/decode")

    def fit(
        self,
        channel_source: np.ndarray,
        train_indices: np.ndarray | None = None,
        batch_size: int = 2,
    ) -> "SharedTuckerCodec":
        source = np.asarray(channel_source)
        if source.ndim != 4 or tuple(source.shape[1:]) != self.layout.config.channel_shape:
            raise ValueError(
                "channel source must have shape "
                f"(P,{self.layout.config.m},{self.layout.config.n},"
                f"{self.layout.config.s}), got {source.shape}"
            )
        if not np.iscomplexobj(source):
            raise TypeError("channel source must be complex-valued")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if train_indices is None:
            indices = np.arange(len(source), dtype=np.int64)
        else:
            indices = np.asarray(train_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("train_indices must be a non-empty one-dimensional array")
        if int(indices.min()) < 0 or int(indices.max()) >= len(source):
            raise IndexError("train index outside channel source")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("train_indices must not contain duplicates")

        dimensions = self.layout.structured_tail
        covariances = [
            np.zeros((dimension, dimension), dtype=np.complex128)
            for dimension in dimensions
        ]
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            channels = np.asarray(source[batch_indices], dtype=np.complex64)
            transformed = beam_delay(channels, self.layout)
            for mode, dimension in enumerate(dimensions):
                unfolded = np.moveaxis(transformed, mode + 1, 0).reshape(
                    dimension, -1
                )
                unfolded_128 = unfolded.astype(np.complex128, copy=False)
                covariances[mode] += unfolded_128 @ unfolded_128.conj().T

        factors: list[np.ndarray] = []
        mode_energy: list[np.ndarray] = []
        for covariance, rank in zip(covariances, self.ranks):
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order].real, 0.0)
            factors.append(
                np.asarray(eigenvectors[:, order[:rank]], dtype=np.complex64)
            )
            mode_energy.append(np.asarray(eigenvalues, dtype=np.float64))

        self.factors = tuple(factors)
        self.mode_energy = tuple(mode_energy)
        self.fitted_indices = indices.copy()
        self.fit_sample_count = int(len(indices))
        return self

    def encode(self, channels: np.ndarray) -> np.ndarray:
        self._check_fitted()
        channels = np.asarray(channels)
        if channels.ndim != 4:
            raise ValueError("channels must include an explicit batch dimension")
        core = beam_delay(channels, self.layout)
        for mode, factor in enumerate(self.factors):
            core = _mode_product(core, factor.conj().T, mode + 1)
        return np.asarray(core, dtype=np.complex64)

    def decode(self, core: np.ndarray) -> np.ndarray:
        self._check_fitted()
        core = np.asarray(core)
        if core.ndim != 6 or tuple(core.shape[1:]) != self.ranks:
            raise ValueError(
                f"core must have shape (B,{','.join(map(str, self.ranks))}), "
                f"got {core.shape}"
            )
        if not np.iscomplexobj(core):
            raise TypeError("core must be complex-valued")
        reconstructed = np.asarray(core, dtype=np.complex64)
        for mode, factor in enumerate(self.factors):
            reconstructed = _mode_product(reconstructed, factor, mode + 1)
        return inverse_beam_delay(
            np.asarray(reconstructed, dtype=np.complex64), self.layout
        )

    def reconstruct(self, channels: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(channels))

    def compression_report(self, amortized_samples: int = 1) -> dict[str, float | int]:
        self._check_fitted()
        if amortized_samples <= 0:
            raise ValueError("amortized_samples must be positive")
        original = int(np.prod(self.layout.structured_tail))
        latent = int(np.prod(self.ranks))
        shared = int(
            sum(factor.shape[0] * factor.shape[1] for factor in self.factors)
        )
        amortized = latent + shared / amortized_samples
        return {
            "original_complex_values_per_sample": original,
            "latent_complex_values_per_sample": latent,
            "shared_complex_values": shared,
            "amortized_complex_values_per_sample": float(amortized),
            "latent_ratio": float(latent / original),
            "amortized_ratio": float(amortized / original),
        }


class GlobalSupportCodec:
    """Keep a fixed Beam-Delay support learned from anchor average power."""

    def __init__(self, layout: AntennaLayout, coefficient_count: int) -> None:
        self.layout = layout
        self.coefficient_count = int(coefficient_count)
        self.total_coefficients = int(np.prod(layout.structured_tail))
        if not 1 <= self.coefficient_count <= self.total_coefficients:
            raise ValueError(
                "coefficient_count must be within [1, "
                f"{self.total_coefficients}], got {self.coefficient_count}"
            )

    def _check_fitted(self) -> None:
        if not hasattr(self, "support_indices"):
            raise RuntimeError("codec must be fitted before encode/decode")

    def fit(
        self,
        channel_source: np.ndarray,
        train_indices: np.ndarray | None = None,
        batch_size: int = 2,
    ) -> "GlobalSupportCodec":
        source = np.asarray(channel_source)
        if source.ndim != 4 or tuple(source.shape[1:]) != self.layout.config.channel_shape:
            raise ValueError(
                "channel source must have shape "
                f"(P,{self.layout.config.m},{self.layout.config.n},"
                f"{self.layout.config.s}), got {source.shape}"
            )
        if not np.iscomplexobj(source):
            raise TypeError("channel source must be complex-valued")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if train_indices is None:
            indices = np.arange(len(source), dtype=np.int64)
        else:
            indices = np.asarray(train_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("train_indices must be a non-empty one-dimensional array")
        if int(indices.min()) < 0 or int(indices.max()) >= len(source):
            raise IndexError("train index outside channel source")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("train_indices must not contain duplicates")

        power_sum = np.zeros(self.total_coefficients, dtype=np.float64)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            transformed = beam_delay(
                np.asarray(source[batch_indices], dtype=np.complex64), self.layout
            ).reshape(len(batch_indices), -1)
            power_sum += np.sum(np.abs(transformed) ** 2, axis=0, dtype=np.float64)
        mean_power = power_sum / len(indices)
        deterministic_order = np.lexsort(
            (np.arange(self.total_coefficients, dtype=np.int64), -mean_power)
        )
        self.support_indices = np.asarray(
            deterministic_order[: self.coefficient_count], dtype=np.int64
        )
        self.mean_power = mean_power
        self.fitted_indices = indices.copy()
        self.fit_sample_count = int(len(indices))
        return self

    def encode(self, channels: np.ndarray) -> np.ndarray:
        self._check_fitted()
        channels = np.asarray(channels)
        if channels.ndim != 4:
            raise ValueError("channels must include an explicit batch dimension")
        transformed = beam_delay(channels, self.layout).reshape(len(channels), -1)
        return np.asarray(transformed[:, self.support_indices], dtype=np.complex64)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        self._check_fitted()
        latent = np.asarray(latent)
        if latent.ndim != 2 or latent.shape[1] != self.coefficient_count:
            raise ValueError(
                f"latent must have shape (B,{self.coefficient_count}), got {latent.shape}"
            )
        if not np.iscomplexobj(latent):
            raise TypeError("latent must be complex-valued")
        flat = np.zeros(
            (len(latent), self.total_coefficients), dtype=np.complex64
        )
        flat[:, self.support_indices] = latent.astype(np.complex64, copy=False)
        structured = flat.reshape((len(latent),) + self.layout.structured_tail)
        return inverse_beam_delay(structured, self.layout)

    def reconstruct(self, channels: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(channels))

    def compression_report(self) -> dict[str, float | int]:
        return {
            "original_complex_values_per_sample": self.total_coefficients,
            "latent_complex_values_per_sample": self.coefficient_count,
            "support_integer_values": self.coefficient_count,
            "latent_ratio": float(
                self.coefficient_count / self.total_coefficients
            ),
        }


def oracle_topk_reconstruction(
    channels: np.ndarray,
    layout: AntennaLayout,
    coefficient_count: int,
) -> np.ndarray:
    """Reconstruct each sample from its own strongest coefficients.

    The sample-specific indices use target information, so this function is an
    upper-bound diagnostic only and must never be used as a deployable predictor.
    """

    channels = np.asarray(channels)
    if channels.ndim != 4:
        raise ValueError("channels must include an explicit batch dimension")
    total = int(np.prod(layout.structured_tail))
    coefficient_count = int(coefficient_count)
    if not 1 <= coefficient_count <= total:
        raise ValueError(f"coefficient_count must be within [1, {total}]")
    flat = beam_delay(channels, layout).reshape(len(channels), total)
    power = np.abs(flat) ** 2
    if coefficient_count == total:
        retained = flat.copy()
    else:
        indices = np.argpartition(
            power, total - coefficient_count, axis=1
        )[:, -coefficient_count:]
        retained = np.zeros_like(flat, dtype=np.complex64)
        rows = np.arange(len(flat), dtype=np.int64)[:, None]
        retained[rows, indices] = flat[rows, indices]
    structured = retained.reshape((len(channels),) + layout.structured_tail)
    return inverse_beam_delay(
        np.asarray(structured, dtype=np.complex64), layout
    )

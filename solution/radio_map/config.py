from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoundConfig:
    """Typed view of the organizer-provided Round1 setup file."""

    p_train_declared: int
    p_test: int
    m: int
    m_h: int
    m_v: int
    m_p: int
    n: int
    n_h: int
    n_v: int
    n_p: int
    s: int
    q: int
    bs_position: tuple[float, float, float]
    weights: tuple[float, float, float]

    @classmethod
    def from_json(cls, path: str | Path) -> "RoundConfig":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = cls(
            p_train_declared=int(raw["P_Train"]),
            p_test=int(raw["P_Test"]),
            m=int(raw["M"]),
            m_h=int(raw["M_H"]),
            m_v=int(raw["M_V"]),
            m_p=int(raw["M_P"]),
            n=int(raw["N"]),
            n_h=int(raw["N_H"]),
            n_v=int(raw["N_V"]),
            n_p=int(raw["N_P"]),
            s=int(raw["S"]),
            q=int(raw["Q"]),
            bs_position=tuple(float(value) for value in raw["X"]),
            weights=tuple(float(value) for value in raw["w"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.m != self.m_h * self.m_v * self.m_p:
            raise ValueError(
                f"M={self.m} does not equal M_H*M_V*M_P="
                f"{self.m_h * self.m_v * self.m_p}"
            )
        if self.n != self.n_h * self.n_v * self.n_p:
            raise ValueError(
                f"N={self.n} does not equal N_H*N_V*N_P="
                f"{self.n_h * self.n_v * self.n_p}"
            )
        if len(self.bs_position) != 3:
            raise ValueError("base-station position X must contain three values")
        if len(self.weights) != 3:
            raise ValueError("score weights w must contain three values")
        if any(value <= 0 for value in (self.m, self.n, self.s, self.q)):
            raise ValueError("channel dimensions must be positive")

    @property
    def channel_shape(self) -> tuple[int, int, int]:
        return self.m, self.n, self.s

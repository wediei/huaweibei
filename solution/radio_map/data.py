from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import RoundConfig


@dataclass(frozen=True)
class DatasetAudit:
    data_dir: str
    declared_train_count: int
    actual_train_count: int
    declared_test_count: int
    actual_test_count: int
    train_position_shape: tuple[int, ...]
    test_position_shape: tuple[int, ...]
    channel_shape: tuple[int, ...]
    channel_dtype: str
    channel_bytes: int
    map_bytes: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RoundDataset:
    data_dir: Path
    config: RoundConfig
    train_pos: np.ndarray
    test_pos: np.ndarray
    train_channel: np.memmap
    map_path: Path

    @classmethod
    def open(cls, data_dir: str | Path) -> "RoundDataset":
        data_dir = Path(data_dir).resolve()
        paths = {
            "setup": data_dir / "Round1_Setup.json",
            "train_pos": data_dir / "Round1_Train_Pos.npy",
            "test_pos": data_dir / "Round1_Test_Pos.npy",
            "channel": data_dir / "Round1_Train_Channel.npy",
            "map": data_dir / "Round1_Map.ply",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing Round1 files: " + ", ".join(missing))

        config = RoundConfig.from_json(paths["setup"])
        train_pos = np.load(paths["train_pos"], mmap_mode="r")
        test_pos = np.load(paths["test_pos"], mmap_mode="r")
        train_channel = np.load(paths["channel"], mmap_mode="r")

        cls._validate_arrays(config, train_pos, test_pos, train_channel)
        return cls(
            data_dir=data_dir,
            config=config,
            train_pos=train_pos,
            test_pos=test_pos,
            train_channel=train_channel,
            map_path=paths["map"],
        )

    @staticmethod
    def _validate_arrays(
        config: RoundConfig,
        train_pos: np.ndarray,
        test_pos: np.ndarray,
        train_channel: np.ndarray,
    ) -> None:
        if train_pos.ndim != 2 or train_pos.shape[1] != 3:
            raise ValueError(
                f"training positions must have shape (P,3), got {train_pos.shape}"
            )
        if test_pos.ndim != 2 or test_pos.shape[1] != 3:
            raise ValueError(
                f"test positions must have shape (P,3), got {test_pos.shape}"
            )
        expected_tail = config.channel_shape
        if train_channel.ndim != 4 or train_channel.shape[1:] != expected_tail:
            raise ValueError(
                "channel shape mismatch: expected "
                f"(P,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]}), "
                f"got {train_channel.shape}"
            )
        if train_channel.shape[0] != train_pos.shape[0]:
            raise ValueError(
                "training position/channel count mismatch: "
                f"{train_pos.shape[0]} versus {train_channel.shape[0]}"
            )
        if not np.issubdtype(train_channel.dtype, np.complexfloating):
            raise ValueError(
                f"training channel must use a complex dtype, got {train_channel.dtype}"
            )
        if test_pos.shape[0] != config.p_test:
            raise ValueError(
                f"test count mismatch: config={config.p_test}, file={test_pos.shape[0]}"
            )
        if not np.issubdtype(train_pos.dtype, np.floating):
            raise ValueError(f"training positions must be real floating values, got {train_pos.dtype}")
        if not np.issubdtype(test_pos.dtype, np.floating):
            raise ValueError(f"test positions must be real floating values, got {test_pos.dtype}")

    def channel_batch(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.ndim != 1:
            raise ValueError("channel indices must be one-dimensional")
        if index_array.size and (
            int(index_array.min()) < 0
            or int(index_array.max()) >= self.train_channel.shape[0]
        ):
            raise IndexError("channel index outside the training range")
        return np.asarray(self.train_channel[index_array], dtype=np.complex64)

    def audit(self) -> DatasetAudit:
        warnings: list[str] = []
        actual_train = int(self.train_pos.shape[0])
        if self.config.p_train_declared != actual_train:
            warnings.append(
                "P_Train mismatch: setup declares "
                f"{self.config.p_train_declared}, but the training files contain "
                f"{actual_train}; declared P_Train equals train+test for the official Round1 files."
            )
        return DatasetAudit(
            data_dir=str(self.data_dir),
            declared_train_count=self.config.p_train_declared,
            actual_train_count=actual_train,
            declared_test_count=self.config.p_test,
            actual_test_count=int(self.test_pos.shape[0]),
            train_position_shape=tuple(int(v) for v in self.train_pos.shape),
            test_position_shape=tuple(int(v) for v in self.test_pos.shape),
            channel_shape=tuple(int(v) for v in self.train_channel.shape),
            channel_dtype=str(self.train_channel.dtype),
            channel_bytes=int((self.data_dir / "Round1_Train_Channel.npy").stat().st_size),
            map_bytes=int(self.map_path.stat().st_size),
            warnings=tuple(warnings),
        )

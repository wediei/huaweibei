from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def create_round_dir(parent: Path) -> Path:
    data_dir = parent / "Round1_Map"
    data_dir.mkdir(parents=True)

    config = {
        "P_Train": 8,
        "P_Test": 2,
        "M": 8,
        "M_H": 2,
        "M_V": 2,
        "M_P": 2,
        "N": 2,
        "N_H": 1,
        "N_V": 1,
        "N_P": 2,
        "S": 4,
        "Q": 2,
        "X": [0.0, 0.0, 5.0],
        "w": [0.4, 0.4, 0.2],
    }
    (data_dir / "Round1_Setup.json").write_text(
        json.dumps(config), encoding="utf-8"
    )

    train_pos = np.array(
        [
            [0.0, 0.0, 1.5],
            [1.0, 0.0, 1.5],
            [0.0, 1.0, 1.5],
            [1.0, 1.0, 1.5],
            [2.0, 0.0, 1.5],
            [2.0, 1.0, 1.5],
        ],
        dtype=np.float64,
    )
    test_pos = np.array(
        [[0.5, 0.5, 1.5], [1.5, 0.5, 1.5]], dtype=np.float64
    )
    rng = np.random.default_rng(7)
    channel = (
        rng.standard_normal((6, 8, 2, 4))
        + 1j * rng.standard_normal((6, 8, 2, 4))
    ).astype(np.complex64)

    np.save(data_dir / "Round1_Train_Pos.npy", train_pos)
    np.save(data_dir / "Round1_Test_Pos.npy", test_pos)
    np.save(data_dir / "Round1_Train_Channel.npy", channel)
    map_vertices = np.array(
        [(0, 0, 0, 0, 0, 1), (1, 1, 1, 1, 0, 0)],
        dtype=[
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
            ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
        ],
    )
    map_header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property double nx\n"
        "property double ny\n"
        "property double nz\n"
        "end_header\n"
    ).encode("ascii")
    (data_dir / "Round1_Map.ply").write_bytes(
        map_header + map_vertices.tobytes()
    )
    return data_dir

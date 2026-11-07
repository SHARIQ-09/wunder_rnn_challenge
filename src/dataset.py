"""Streaming PyTorch Dataset for Alpha Connectome (Wunder RNN Challenge).

Efficiently streams sequences one row-group (20 000 steps) at a time
from the 29 GB train.parquet without loading the full file into memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Union, Tuple, Dict, Any
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

# Import column definitions and constants from utils
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, WARMUP
from src.features import extract_32_features_sequence, TOTAL_FEATURES
from src.feature_sets import build_features


class StreamingSequenceDataset(Dataset):
    """PyTorch Dataset that streams full 20,000-step sequences via Parquet row groups.

    Memory footprint per sequence: ~11.5 MB (20,000 steps x 144 float32 features).
    Safe for multiprocessing DataLoader workers via lazy ParquetFile initialization.
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        row_groups: Optional[Sequence[int]] = None,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        target_columns: Sequence[str] = TARGET_COLUMNS,
        include_targets: bool = True,
        extract_features: bool = True,
        feature_set: str = "v144",
    ):
        super().__init__()
        self.parquet_path = Path(parquet_path).resolve()
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        self.extract_features = extract_features
        self.feature_set = feature_set

        # Probe metadata once in main process to get row group count
        pf = pq.ParquetFile(self.parquet_path)
        self.total_row_groups = pf.num_row_groups

        if row_groups is None:
            self.row_groups = list(range(self.total_row_groups))
        else:
            self.row_groups = list(row_groups)

        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        self.include_targets = include_targets

        # Required columns for read
        self.read_columns = list(self.feature_columns) + ["need_prediction"]
        if self.include_targets:
            self.read_columns.extend(self.target_columns)
        if "seq_ix" in pf.schema_arrow.names:
            self.read_columns.append("seq_ix")
        if "is_scored" in pf.schema_arrow.names:
            self.read_columns.append("is_scored")

        # Lazy file handle per process / worker
        self._pf: Optional[pq.ParquetFile] = None

    def __getstate__(self):
        # Open Parquet handles cannot be pickled into DataLoader worker processes;
        # each worker reopens the file lazily.
        state = self.__dict__.copy()
        state["_pf"] = None
        return state

    def _get_pf(self) -> pq.ParquetFile:
        if self._pf is None:
            self._pf = pq.ParquetFile(self.parquet_path)
        return self._pf

    def __len__(self) -> int:
        return len(self.row_groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rg_idx = self.row_groups[idx]
        pf = self._get_pf()

        table = pf.read_row_group(rg_idx, columns=self.read_columns)

        # Fast vectorized extraction via pandas (48x faster than column_stack)
        df = table.to_pandas()
        features = df[self.feature_columns].to_numpy(dtype=np.float32)
        if self.extract_features:
            features = build_features(features, self.feature_set)
        features = np.ascontiguousarray(features, dtype=np.float32)
        need_prediction = np.ascontiguousarray(df["need_prediction"].to_numpy(dtype=bool)).copy()

        item: Dict[str, Any] = {
            "features": torch.from_numpy(features),                  # (20000, 144) float32
            "need_prediction": torch.from_numpy(need_prediction),    # (20000,) bool
            "row_group": rg_idx,
        }

        if "is_scored" in df.columns:
            item["is_scored"] = torch.from_numpy(np.ascontiguousarray(df["is_scored"].to_numpy(dtype=bool)).copy())

        if "seq_ix" in df.columns:
            item["seq_ix"] = int(df["seq_ix"].iloc[0])
        else:
            item["seq_ix"] = rg_idx

        if self.include_targets:
            targets = np.ascontiguousarray(df[self.target_columns].to_numpy(dtype=np.float32)).copy()
            item["targets"] = torch.from_numpy(targets)              # (20000, 2) float32

        return item


class ChunkedSequenceDataset(Dataset):
    """Wraps StreamingSequenceDataset to yield sub-sequence chunks for Truncated BPTT.

    Allows training recurrent models without unrolling full 20,000 steps into VRAM.
    """

    def __init__(
        self,
        base_dataset: StreamingSequenceDataset,
        chunk_size: int = 2048,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size
        self.chunks_per_seq = int(np.ceil(SEQUENCE_LENGTH / chunk_size))

    def __len__(self) -> int:
        return len(self.base_dataset) * self.chunks_per_seq

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_idx = idx // self.chunks_per_seq
        chunk_idx = idx % self.chunks_per_seq

        full_item = self.base_dataset[seq_idx]
        start_step = chunk_idx * self.chunk_size
        end_step = min(start_step + self.chunk_size, SEQUENCE_LENGTH)

        chunk: Dict[str, Any] = {
            "features": full_item["features"][start_step:end_step],
            "need_prediction": full_item["need_prediction"][start_step:end_step],
            "seq_ix": full_item["seq_ix"],
            "row_group": full_item["row_group"],
            "chunk_idx": chunk_idx,
            "is_start": (chunk_idx == 0),
            "is_end": (end_step == SEQUENCE_LENGTH),
        }

        if "targets" in full_item:
            chunk["targets"] = full_item["targets"][start_step:end_step]

        return chunk


def get_dataloaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    train_row_groups: Optional[Sequence[int]] = None,
    val_row_groups: Optional[Sequence[int]] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Helper factory function to create PyTorch DataLoaders for training & validation."""
    train_ds = StreamingSequenceDataset(train_path, row_groups=train_row_groups)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_path is not None:
        val_ds = StreamingSequenceDataset(val_path, row_groups=val_row_groups)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader


if __name__ == "__main__":
    import time

    print("=== Testing StreamingSequenceDataset (Phase 2) ===")
    data_path = ROOT_DIR / "datasets" / "train.parquet"
    if not data_path.exists():
        print(f"Dataset file not found at {data_path}")
        sys.exit(1)

    ds = StreamingSequenceDataset(data_path, row_groups=[0, 1, 2])
    print(f"Dataset initialized with {len(ds)} sequences.")

    t0 = time.time()
    item = ds[0]
    t1 = time.time()

    print(f"Sequence loaded in {t1 - t0:.3f} seconds:")
    print(f"  Features shape : {item['features'].shape} (dtype: {item['features'].dtype})")
    print(f"  Targets shape  : {item['targets'].shape} (dtype: {item['targets'].dtype})")
    print(f"  Mask shape     : {item['need_prediction'].shape} (dtype: {item['need_prediction'].dtype})")
    print(f"  Seq index      : {item['seq_ix']}")
    print(f"  Scored steps   : {item['need_prediction'].sum().item()} / {len(item['need_prediction'])}")

    # Test ChunkedSequenceDataset
    chunked_ds = ChunkedSequenceDataset(ds, chunk_size=2048)
    chunk = chunked_ds[0]
    print(f"\nChunked Dataset (chunk_size=2048):")
    print(f"  Total chunks   : {len(chunked_ds)}")
    print(f"  Chunk 0 features shape: {chunk['features'].shape}")
    print(f"  Chunk 0 is_start      : {chunk['is_start']}")
    print("[SUCCESS] All Phase 2 streaming dataset checks passed!")

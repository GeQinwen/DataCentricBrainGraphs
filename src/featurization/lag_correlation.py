#!/usr/bin/env python3
"""
Build a PyTorch Geometric dataset with lagged-correlation featurization.

---------------------
Given per-subject z-scored time series (shape [T, N]), for each subject it:
1) Computes the original Pearson FC:  N×N.
2) Computes a lagged FC at lag=L by correlating X(t) with X(t+L):
   - Build an expanded time series [T-L, 2N] = [X(0..T-L-1, :), X(L..T-1, :)].
   - Pearson correlation over the expanded TS gives a 2N×2N block matrix.
   - Extract the cross blocks:
       F_lag        = Corr[0:N,     N:2N]   (X vs X_lag)
       F_lag_reverse= Corr[N:2N,    0:N]    (X_lag vs X)
3) Uses the original FC to build edges by keeping the top edge_pct% among
   strictly positive correlations (symmetrized).
4) Sets node features x according to --feature_mode:
   - concat: [F_orig | F_lag | F_lag_reverse] → N×(k*N)
     (k = 3 if --include_reverse else 2)
5) Saves a PyG InMemoryDataset to <root>/processed/<name>.pt.

Example
-------
python src/featurization/lag_correlation.py \
  --root data/rs_100/rs_100_lag5_concat \
  --name HCPGender \
  --ts_dir data/raw/HCPGender/time_series_100 \
  --lag 5 \
  --edge_pct 5 \
  --include_reverse \
  --n_jobs 1
"""

import argparse
import os
import glob
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
from nilearn.connectome import ConnectivityMeasure


# ----------------------------- helpers ------------------------------------ #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_ts_path(ts_dir: str, sid: str) -> Optional[str]:
    """
    Locate a subject's time-series file.
    Primary: {sid}_time_series.npy
    Fallback: {sid}_*_time_series.npy (prefer files containing 'REST1_LR').
    """
    primary = os.path.join(ts_dir, f"{sid}_time_series.npy")
    if os.path.exists(primary):
        return primary
    pattern = os.path.join(ts_dir, f"{sid}_*_time_series.npy")
    matches = glob.glob(pattern)
    if not matches:
        return None
    preferred = [m for m in matches if "REST1_LR" in os.path.basename(m)]
    return preferred[0] if preferred else matches[0]


def _top_positive_percentile_adj(corr_tensor: torch.Tensor, edge_pct: float) -> torch.Tensor:
    """
    Keep the top 'edge_pct' percent among strictly positive correlations.
    Returns a symmetrized binary adjacency (torch.float32) of shape [N, N].
    """
    A = corr_tensor.clone()
    pos_vals = A[A > 0].detach().cpu().numpy()
    if pos_vals.size == 0:
        A[:] = 0.0
        return A
    cutoff = np.percentile(pos_vals, 100 - edge_pct)
    A[A < cutoff] = 0.0
    A[A >= cutoff] = 1.0
    A = torch.maximum(A, A.t())
    return A


def _pearson_fc(ts: np.ndarray) -> np.ndarray:
    """
    Pearson FC using nilearn ConnectivityMeasure. Returns N×N with zero diagonal.
    """
    conn = ConnectivityMeasure(kind='correlation')
    fc = conn.fit_transform([ts])[0].astype(np.float32)
    np.fill_diagonal(fc, 0.0)
    return fc


def _expand_time_series(ts: np.ndarray, lag: int) -> np.ndarray:
    """
    Concatenate X(t) and X(t+lag) column-wise to produce [T-lag, 2N].
    - ts: [T, N]
    - lag: positive integer < T
    """
    T, N = ts.shape
    if lag <= 0 or lag >= T:
        raise ValueError(f"--lag must be in [1, T-1]; got lag={lag}, T={T}")
    truncated = ts[:-lag, :]   # [T-lag, N]  -> X(t)
    shifted  = ts[lag:,  :]    # [T-lag, N]  -> X(t+lag)
    expanded = np.concatenate([truncated, shifted], axis=1).astype(np.float32)  # [T-lag, 2N]
    return expanded


def _lagged_blocks(ts: np.ndarray, lag: int, include_reverse: bool) -> np.ndarray:
    """
    Compute lagged cross-correlation blocks from expanded TS.
    Returns either:
      - [F_lag]                       (N×N) if include_reverse=False
      - [F_lag | F_lag_reverse]      (N×(2N)) if include_reverse=True and concatenated horizontally
    """
    T, N = ts.shape
    expanded = _expand_time_series(ts, lag)                 # [T-lag, 2N]
    conn = ConnectivityMeasure(kind='correlation')
    big_fc = conn.fit_transform([expanded])[0].astype(np.float32)  # [2N, 2N]
    # Extract blocks
    F_lag = big_fc[0:N,     N:2*N].copy()
    np.fill_diagonal(F_lag, 0.0)
    if include_reverse:
        F_rev = big_fc[N:2*N, 0:N].copy()
        np.fill_diagonal(F_rev, 0.0)
        return np.concatenate([F_lag, F_rev], axis=1)       # N×(2N)
    else:
        return F_lag                                        # N×N


# -------------------------- dataset definition ---------------------------- #

class BrainConnectomeLaggedFC(InMemoryDataset):
    """
    Creates a PyG dataset where:
      - edge_index: built from top edge_pct% positive edges of the ORIGINAL FC.
      - x: node features determined by --feature_mode:
           * 'concat' (default): [F_orig | F_lag | (optional F_rev)] → N×(k*N)
           * 'original': F_orig only → N×N
      - y: tensor([gender, ageclass, listsort, pmat])
    """

    def __init__(
        self,
        root: str,
        name: str,
        ts_dir: str,
        ids_pkl: str,
        behavior_csv: str,
        lag: int,
        edge_pct: float,
        feature_mode: str = "concat",
        include_reverse: bool = True,
        n_jobs: int = 1,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.dataset_name = name
        self.ts_dir = ts_dir
        self.ids_pkl = ids_pkl
        self.behavior_csv = behavior_csv
        self.lag = int(lag)
        self.edge_pct = float(edge_pct)
        self.feature_mode = feature_mode
        self.include_reverse = bool(include_reverse)
        self.n_jobs = int(n_jobs)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.dataset_name}.pt"]

    # -- core per-subject processing --
    def _process_one(self, sid: str, behavioral_df: pd.DataFrame) -> Optional[Data]:
        try:
            sid_str = str(sid)
            ts_path = _find_ts_path(self.ts_dir, sid_str)
            if ts_path is None:
                print(f"[WARN] time series not found for {sid_str} in {self.ts_dir}")
                return None

            # Load time series (assumed z-scored). Shape: [T, N]
            ts = np.load(ts_path)
            if ts.ndim != 2:
                raise ValueError(f"Expected 2D time series [T, N], got {ts.shape} for SID {sid_str}")
            T, N = ts.shape

            # Original FC for edges
            F_orig = _pearson_fc(ts)                  # N×N
            corr_orig = torch.from_numpy(F_orig)

            # Build adjacency from original FC (top positive %)
            A = _top_positive_percentile_adj(corr_orig, self.edge_pct)
            edge_index = A.nonzero().t().to(torch.long)

            # Build node features
            if self.feature_mode == "original":
                X = F_orig
            elif self.feature_mode == "concat":
                F_lag_blocks = _lagged_blocks(ts, self.lag, self.include_reverse)  # N×N or N×(2N)
                X = np.concatenate([F_orig, F_lag_blocks], axis=1).astype(np.float32)  # N×(k*N)
            else:
                raise ValueError("--feature_mode must be 'concat' or 'original'")

            x = torch.from_numpy(X)

            # Labels: [gender, ageclass, listsort, pmat]
            sid_int = int(sid)
            gender = behavioral_df.loc[sid_int, 'Gender']
            g = 1 if gender == 'M' else 0
            labels = torch.tensor([
                g,
                behavioral_df.loc[sid_int, 'AgeClass'],
                behavioral_df.loc[sid_int, 'ListSort_AgeAdj'],
                behavioral_df.loc[sid_int, 'PMAT24_A_CR'],
            ], dtype=torch.float32)

            return Data(x=x, edge_index=edge_index, y=labels)

        except Exception as e:
            print(f"[ERROR] subject {sid}: {e}")
            return None

    def process(self) -> None:
        # Behavior & IDs (HCP-style)
        behavioral_df = pd.read_csv(self.behavior_csv).set_index('Subject')[
            ['Gender', 'Age', 'ListSort_AgeAdj', 'PMAT24_A_CR']
        ]
        mapping = {'22-25': 0, '26-30': 1, '31-35': 2, '36+': 3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)

        with open(self.ids_pkl, 'rb') as f:
            ids = pickle.load(f)

        results = Parallel(n_jobs=self.n_jobs, prefer='processes')(
            delayed(self._process_one)(sid, behavioral_df) for sid in tqdm(ids)
        )
        data_list = [d for d in results if d is not None]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        _ensure_dir(self.processed_dir)
        torch.save((data, slices), self.processed_paths[0])


# ------------------------------- CLI -------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Build PyG dataset with lagged-correlation featurization")
    p.add_argument('--root', type=str, required=True,
                   help='Dataset root (PyG style); output .pt goes under <root>/processed')
    p.add_argument('--name', type=str, required=True,
                   help='Dataset name; saved as <name>.pt')
    p.add_argument('--ts_dir', type=str, required=True,
                   help='Directory containing {id}_time_series.npy files')
    p.add_argument('--ids_pkl', type=str, default='data/ids.pkl',
                   help='Path to ids.pkl (default: data/ids.pkl)')
    p.add_argument('--behavior_csv', type=str, default='data/HCP_behavioral.csv',
                   help='Path to HCP_behavioral.csv (default: data/HCP_behavioral.csv)')

    p.add_argument('--lag', type=int, default=5,
                   help='Time lag (in TRs) for lagged correlation (default: 5)')
    p.add_argument('--edge_pct', type=float, default=5.0,
                   help='Top positive percent edges when building adjacency from ORIGINAL FC (default: 5)')
    p.add_argument('--feature_mode', type=str, choices=['concat', 'original'], default='concat',
                   help="Node feature mode: 'concat' ([F_orig|F_lag|F_rev?]) or 'original' (F_orig only)")
    p.add_argument('--include_reverse', action='store_true',
                   help='If set, also include reverse-lag block (X_lag vs X) in features when using concat mode')
    p.add_argument('--n_jobs', type=int, default=1,
                   help='Parallel workers (default: 1)')

    args = p.parse_args()

    BrainConnectomeLaggedFC(
        root=args.root,
        name=args.name,
        ts_dir=args.ts_dir,
        ids_pkl=args.ids_pkl,
        behavior_csv=args.behavior_csv,
        lag=args.lag,
        edge_pct=args.edge_pct,
        feature_mode=args.feature_mode,
        include_reverse=args.include_reverse,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main()

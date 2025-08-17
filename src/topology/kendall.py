#!/usr/bin/env python3
"""
Build a PyTorch Geometric dataset from precomputed time series by
(1) computing functional connectivity,
(2) constructing a top-positives-% adjacency for edges, and
(3) saving a single .pt file compatible with PyG's InMemoryDataset.

Inputs
------
- A directory of per-subject time series .npy files named "{id}_time_series.npy"
  (shape: [T, N], where N is #ROIs). Time series are assumed to be z-scored upstream.
- ids.pkl: list of subject IDs to include.
- HCP_behavioral.csv: to construct labels [gender, ageclass, listsort, pmat].

Outputs
-------
- <root>/processed/<name>.pt : a collated PyG dataset (Data list -> (data, slices)).

Example
-------
python src/topology/kendall.py \
  --root data/rs_100/rs_100_kendall \
  --name HCPGender \
  --ts_dir data/raw/HCPGender/time_series_100 \
  --edge_pct 5 \
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

from connectivity_matrices import KendallConnectivityMeasure


# ----------------------------- helpers ------------------------------------ #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _top_positive_percentile_adj(corr_tensor: torch.Tensor, edge_pct: float) -> torch.Tensor:
    """
    Construct a binary adjacency by keeping the top 'edge_pct' percent among strictly positive correlations.
    - corr_tensor: [N, N] torch.float32 (diagonal assumed 0)
    Returns:
      A (torch.float32) of shape [N, N] with {0,1} entries (symmetrized).
    """
    A = corr_tensor.clone()
    pos_vals = A[A > 0].detach().cpu().numpy()
    if pos_vals.size == 0:
        A[:] = 0.0
        return A
    cutoff = np.percentile(pos_vals, 100 - edge_pct)
    A[A < cutoff] = 0.0
    A[A >= cutoff] = 1.0
    # enforce symmetry for undirected usage
    A = torch.maximum(A, A.t())
    return A


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


def _kendall_fc(ts: np.ndarray) -> np.ndarray:
    """
    Compute Kendall's tau correlation matrix using the local KendallConnectivityMeasure.
    - ts: [T, N] z-scored time series
    Returns:
      fc: [N, N] float32 Kendall correlation matrix with zeroed diagonal
    """
    conn = KendallConnectivityMeasure(kind='correlation')
    fc = conn.fit_transform([ts])[0].astype(np.float32)
    np.fill_diagonal(fc, 0.0)
    return fc


# -------------------------- dataset definition ---------------------------- #

class BrainConnectomeKendallFC(InMemoryDataset):
    """
    Creates a PyG dataset where:
      - x: N×N Kendall correlation matrix (diagonal zeroed)
      - edge_index: edges from top edge_pct% positive correlations
      - y: tensor([gender, ageclass, listsort, pmat])
    """

    def __init__(
        self,
        root: str,
        name: str,
        ts_dir: str,
        ids_pkl: str,
        behavior_csv: str,
        edge_pct: float,
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
        self.edge_pct = float(edge_pct)
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

            # Load time series (assumed already z-scored upstream). Shape: [T, N]
            ts = np.load(ts_path)
            if ts.ndim != 2:
                raise ValueError(f"Expected 2D time series [T, N], got {ts.shape} for SID {sid_str}")

            # Kendall FC
            fc = _kendall_fc(ts)  # [N, N]
            corr = torch.from_numpy(fc)  # x

            # Build top-positive-% adjacency
            A = _top_positive_percentile_adj(corr, self.edge_pct)
            edge_index = A.nonzero().t().to(torch.long)

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

            data = Data(x=corr, edge_index=edge_index, y=labels)
            return data

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

        # Parallel subjects
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
    p = argparse.ArgumentParser(description="Build FC PyG dataset from time series (Kendall correlation)")
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
    p.add_argument('--edge_pct', type=float, default=5.0,
                   help='Top positive percent edges when building adjacency (default: 5)')
    p.add_argument('--n_jobs', type=int, default=1,
                   help='Parallel workers (default: 1)')
    args = p.parse_args()

    BrainConnectomeKendallFC(
        root=args.root,
        name=args.name,
        ts_dir=args.ts_dir,
        ids_pkl=args.ids_pkl,
        behavior_csv=args.behavior_csv,
        edge_pct=args.edge_pct,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main()

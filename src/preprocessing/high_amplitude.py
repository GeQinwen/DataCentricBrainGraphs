#!/usr/bin/env python3
"""
Build a PyTorch Geometric dataset from precomputed time series by
(1) applying high-amplitude retention on the time series, and
(2) computing FC (Pearson correlation), then
(3) constructing a top-positives-% adjacency for edges, and
(4) saving a single .pt file compatible with PyG's InMemoryDataset.

Inputs
------
- A directory of per-subject time series .npy files named "{id}_time_series.npy"
  (shape: [T, N], where N is #ROIs).
- ids.pkl: list of subject IDs to include.
- HCP_behavioral.csv: to construct labels [gender, ageclass, listsort, pmat].

Outputs
-------
- <root>/processed/<name>.pt : a collated PyG dataset (Data list -> (data, slices)).

Example
-------
python src/preprocessing/high_amplitude.py \
  --root data/rs_100/rs_100_sd1=1_pearson \
  --name HCPGender \
  --ts_dir data/raw/HCPGender/time_series_100 \
  --thresh_type sd --thresh 1.0 \
  --retain binary \
  --edge_pct 5 \
  --n_jobs 1

"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
from nilearn.connectome import ConnectivityMeasure
from typing import Optional
import glob

# ----------------------------- helpers ------------------------------------ #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def _apply_high_amplitude_retention(ts: np.ndarray, thresh_type: str, thresh: float, retain: str) -> np.ndarray:
    """Apply high-amplitude retention on standardized signals.
    - thresh_type in {"sd", "pct"}
      - sd: keep entries >= thresh (on standardized data)
      - pct: compute threshold over positive standardized entries, keep top-p% (>= cutoff)
    - retain in {"value", "binary"}
      - value: keep original (pre-threshold) *standardized* values when above threshold, else 0
      - binary: set above-threshold to 1, else 0
    Returns processed time series with same shape as input.
    """
    Z = ts 

    if thresh_type == "sd":
        cutoff = float(thresh)
    elif thresh_type == "pct":
        p = float(thresh)
        pos_vals = Z[Z > 0]
        if pos_vals.size == 0:
            cutoff = np.inf  # no positive entries; mask will be all False
        else:
            cutoff = np.percentile(pos_vals, 100 - p)
    else:
        raise ValueError("thresh_type must be 'sd' or 'pct'")

    mask = Z >= cutoff

    if retain == "value":
        out = np.where(mask, Z, 0.0)
    elif retain == "binary":
        out = np.where(mask, 1.0, 0.0)
    else:
        raise ValueError("retain must be 'value' or 'binary'")

    return out


def _top_positive_percentile_adj(corr_tensor: torch.Tensor, edge_pct: float) -> torch.Tensor:
    """Construct a binary adjacency by keeping top 'edge_pct' percent positive correlations.
    corr_tensor: [N, N] torch.float (diagonal assumed 0)
    Returns binary adjacency as torch.float of shape [N, N].
    """
    A = corr_tensor.clone()
    # Extract strictly positive entries to compute percentile cutoff
    pos_vals = A[A > 0].detach().cpu().numpy()
    if pos_vals.size == 0:
        # no positive correlations: return all-zero adjacency
        A[:] = 0.0
        return A
    cutoff = np.percentile(pos_vals, 100 - edge_pct)
    A[A < cutoff] = 0.0
    A[A >= cutoff] = 1.0
    return A


# -------------------------- dataset definition ---------------------------- #

class BrainConnectomeFC(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        ts_dir: str,
        ids_pkl: str,
        behavior_csv: str,
        thresh_type: str,
        thresh: float,
        retain: str,
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
        self.thresh_type = thresh_type
        self.thresh = float(thresh)
        self.retain = retain
        self.edge_pct = float(edge_pct)
        self.n_jobs = int(n_jobs)

        super().__init__(root, transform, pre_transform, pre_filter)
        # After processing, load the collated data
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}.pt"]

    # -- core per-subject processing --
    def _process_one(self, sid: str, behavioral_df: pd.DataFrame) -> Optional[Data]:
        try:
            # Load raw time series; expect file named "{sid}_time_series.npy"
            sid_str = str(sid)
            ts_path = os.path.join(self.ts_dir, f"{sid_str}_time_series.npy")

            if not os.path.exists(ts_path):
                pattern = os.path.join(self.ts_dir, f"{sid_str}_*_time_series.npy")
                matches = glob.glob(pattern)
                if not matches:
                    print(f"[WARN] time series not found for {sid_str}: tried {ts_path} and pattern {pattern}")
                    return None
                preferred = [m for m in matches if "REST1_LR" in os.path.basename(m)]
                ts_path = preferred[0] if preferred else matches[0]

            ts = np.load(ts_path)  # shape [T, N]

            # High-amplitude retention on standardized signals
            ts_processed = _apply_high_amplitude_retention(ts, self.thresh_type, self.thresh, self.retain)

            # Pearson correlation FC
            conn = ConnectivityMeasure(kind='correlation')
            fc = conn.fit_transform([ts_processed])[0]  # [N, N]
            np.fill_diagonal(fc, 0.0)
            corr = torch.tensor(fc, dtype=torch.float32)

            # Build top-positive-% adjacency
            A = _top_positive_percentile_adj(corr, self.edge_pct)
            edge_index = A.nonzero().t().to(torch.long)

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
        # Behavior & IDs
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
    p = argparse.ArgumentParser(description="Build FC PyG dataset from time series with high-amplitude retention")
    p.add_argument('--root', type=str, required=True, help='Dataset root (PyG style); output .pt goes under <root>/processed')
    p.add_argument('--name', type=str, required=True, help='Dataset name; saved as <name>.pt')
    p.add_argument('--ts_dir', type=str, required=True, help='Directory containing {id}_time_series.npy files')
    p.add_argument('--ids_pkl', type=str, default=None, help='Path to ids.pkl')
    p.add_argument('--behavior_csv', type=str, default=None, help='Path to HCP_behavioral.csv')

    p.add_argument('--thresh_type', type=str, choices=['sd','pct'], default='sd', help='Thresholding rule: sd or pct')
    p.add_argument('--thresh', type=float, default=1.0, help='For sd: k standard deviations; for pct: top-p percent of positive entries')
    p.add_argument('--retain', type=str, choices=['value','binary'], default='binary', help='Above-threshold retention: keep value or binarize to 1')

    p.add_argument('--edge_pct', type=float, default=5.0, help='Top positive percent edges when building adjacency')
    p.add_argument('--n_jobs', type=int, default=1, help='Parallel workers')

    args = p.parse_args()

    ids_pkl = args.ids_pkl or 'data/ids.pkl'
    behavior_csv = args.behavior_csv or 'data/HCP_behavioral.csv'

    BrainConnectomeFC(
        root=args.root,
        name=args.name,
        ts_dir=args.ts_dir,
        ids_pkl=ids_pkl,
        behavior_csv=behavior_csv,
        thresh_type=args.thresh_type,
        thresh=args.thresh,
        retain=args.retain,
        edge_pct=args.edge_pct,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main()

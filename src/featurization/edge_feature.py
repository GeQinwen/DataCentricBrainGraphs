#!/usr/bin/env python3
"""
Edge-feature fusion framework (PSK + Lag) on existing PyG datasets.

----
From four existing PyG datasets built on the same subjects and ROI set:
  • Pearson FC         (P)
  • Spearman FC        (S)
  • Kendall FC         (K)
  • Lagged FC (lag=L)  (L)

produce a new PyG dataset where, for each subject:
  • Node features x  = concat[P | S | K | Lag | Lag_reverse]  →  N × (5N)
  • Edge index       = from Pearson (default) or union(P,S,K)
  • Edge features    = 3-dim binary per edge: [is_in_P, is_in_S, is_in_K]

Auto-detects input datasets using only:
  --name (e.g., HCPGender)
  --n_rois (e.g., 1000)
  --lag (e.g., 5)

Example
-------
python src/featurization/edge_feature.py \
  --name HCPGender \
  --n_rois 100 \
  --lag 5 \
  --edge_source pearson \
  --out_root data/rs_100/rs_100_edgefeat_lag5_pearson \
  --out_name HCPGender_edgefeat

"""

import argparse
import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_dense_adj
from joblib import Parallel, delayed
from tqdm import tqdm


# ----------------------------- path utils --------------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _candidate_paths(base_dir: str, n_rois: int, variant: str, name: str) -> List[str]:
    """
    Return prioritized candidate processed .pt paths for a given variant.
    """
    rs_dir = os.path.join(base_dir, f"rs_{n_rois}")
    preferred = os.path.join(rs_dir, f"rs_{n_rois}_{variant}", "processed", f"{name}.pt")
    # broad recursive fallback
    fallback_glob = os.path.join(rs_dir, "**", "processed", f"{name}.pt")
    paths = []
    if os.path.exists(preferred):
        paths.append(preferred)
    # collect fallbacks that contain the variant token in their parent directory name
    for p in glob.glob(fallback_glob, recursive=True):
        parent = os.path.basename(os.path.dirname(os.path.dirname(p)))  # the dir right above 'processed'
        if variant.lower() in parent.lower() and p not in paths:
            paths.append(p)
    return paths


def _candidate_paths_lag(base_dir: str, n_rois: int, lag: int, name: str) -> List[str]:
    """
    Return prioritized candidate processed .pt paths for lag dataset (lag=L).
    """
    rs_dir = os.path.join(base_dir, f"rs_{n_rois}")
    patterns = [
        f"rs_{n_rois}_lag{lag}",
        f"lag{lag}",
        f"{lag}lag",
    ]
    # First, common convention (explicit)
    preferred = []
    for token in patterns:
        preferred_path = os.path.join(rs_dir, f"{token}", "processed", f"{name}.pt")
        if os.path.exists(preferred_path):
            preferred.append(preferred_path)
    # Broad recursive fallback
    fallback_glob = os.path.join(rs_dir, "**", "processed", f"{name}.pt")
    fallbacks = []
    for p in glob.glob(fallback_glob, recursive=True):
        parent = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if any(tok in parent.lower() for tok in [t.lower() for t in patterns]):
            fallbacks.append(p)
    # Deduplicate, keep order (preferred first)
    seen = set()
    out = []
    for p in preferred + fallbacks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _pick_first_or_raise(paths: List[str], label: str) -> str:
    if not paths:
        raise FileNotFoundError(f"Could not locate processed dataset for: {label}")
    return paths[0]


# -------------------------- minimal dataset I/O ---------------------------- #

class _LoadedInMemoryDataset(InMemoryDataset):
    """
    Minimal loader around an existing <root>/processed/<name>.pt
    """
    def __init__(self, pt_path: str):
        # Expect path like ".../<root>/processed/<name>.pt"
        self._pt_path = pt_path
        root = os.path.dirname(os.path.dirname(pt_path))  # the <root>
        super().__init__(root)
        self.data, self.slices = torch.load(self._pt_path)

    @property
    def processed_file_names(self) -> List[str]:
        # Not used for reading; we load directly from _pt_path
        return [os.path.basename(self._pt_path)]

    def process(self):
        raise RuntimeError("This loader expects an existing processed .pt file.")


class _SaveOnlyDataset(InMemoryDataset):
    """
    Collate and save a list of Data objects into <root>/processed/<name>.pt
    """
    def __init__(self, root: str, name: str, data_list: List[Data]):
        self._root = root
        self._name = name
        super().__init__(root)
        data, slices = self.collate(data_list)
        _ensure_dir(self.processed_dir)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self._name}.pt"]

    def process(self):
        pass


# ---------------------------- feature helpers ----------------------------- #

def _extract_lag_blocks(lx: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given lag-dataset node feature matrix X_lag (shape: N×M), extract:
      F_lag        = X_lag[:, N:2N]
      F_lag_reverse= X_lag[:, 2N:3N]  (if present), else F_lag.T
    Assumes the lag dataset was built with 'concat' mode: [F_orig | F_lag | (F_lrev)].
    """
    N, M = lx.size(0), lx.size(1)
    if N != n:
        raise ValueError(f"Lag-dataset N mismatch: expected {n}, got {N}")
    if M >= 3 * n:
        F_lag = lx[:, n:2*n]
        F_rev = lx[:, 2*n:3*n]
    elif M >= 2 * n:
        F_lag = lx[:, n:2*n]
        F_rev = F_lag.t() 
    else:
        raise ValueError(f"Lag dataset features width={M} is insufficient; need ≥2N (got {M}, N={n}).")
    return F_lag, F_rev


def _edge_set(edge_index: torch.Tensor) -> set:
    """
    Convert edge_index [2, E] into a set of (u, v) tuples (directional).
    """
    return set(map(tuple, edge_index.t().tolist()))


def _edge_union(*edge_indices: torch.Tensor) -> torch.Tensor:
    """
    Union of multiple edge_index tensors (directional); returns [2, E_union].
    """
    s = set()
    for ei in edge_indices:
        s |= _edge_set(ei)
    if not s:
        return torch.empty(2, 0, dtype=torch.long)
    arr = torch.tensor(list(s), dtype=torch.long)
    return arr.t().contiguous()


def _build_edge_attr(edge_index: torch.Tensor,
                     set_p: set, set_s: set, set_k: set) -> torch.Tensor:
    """
    For each edge in edge_index (directional), produce 3-dim binary edge feature:
      [is_in_P, is_in_S, is_in_K]
    """
    E = edge_index.size(1)
    edge_attr = torch.zeros((E, 3), dtype=torch.float32)
    for i, (u, v) in enumerate(edge_index.t().tolist()):
        tup = (u, v)
        if tup in set_p: edge_attr[i, 0] = 1.0
        if tup in set_s: edge_attr[i, 1] = 1.0
        if tup in set_k: edge_attr[i, 2] = 1.0
    return edge_attr


# ------------------------------- pipeline --------------------------------- #

def fuse_edge_features(
    name: str,
    n_rois: int,
    lag: int,
    out_root: str,
    out_name: str,
    edge_source: str = "pearson",
    base_dir: str = "data",
    n_jobs: int = 1,
) -> None:
    """
    Build fused dataset and save to <out_root>/processed/<out_name>.pt
    """
    # 1) Resolve .pt paths
    p_path = _pick_first_or_raise(_candidate_paths(base_dir, n_rois, "pearson",  name), "pearson")
    s_path = _pick_first_or_raise(_candidate_paths(base_dir, n_rois, "spearman", name), "spearman")
    k_path = _pick_first_or_raise(_candidate_paths(base_dir, n_rois, "kendall",  name), "kendall")
    l_path = _pick_first_or_raise(_candidate_paths_lag(base_dir, n_rois, lag,    name), f"lag={lag}")

    print("[INFO] Using datasets:")
    print("  P:", p_path)
    print("  S:", s_path)
    print("  K:", k_path)
    print("  L:", l_path)

    # 2) Load datasets
    P = _LoadedInMemoryDataset(p_path)
    S = _LoadedInMemoryDataset(s_path)
    K = _LoadedInMemoryDataset(k_path)
    L = _LoadedInMemoryDataset(l_path)

    # Basic sanity
    if not (len(P) == len(S) == len(K) == len(L)):
        raise ValueError(f"Dataset length mismatch: P={len(P)}, S={len(S)}, K={len(K)}, L={len(L)}")

    # 3) Build fused Data per subject (optionally in parallel)
    def _do_one(idx: int) -> Optional[Data]:
        try:
            p: Data = P[idx]
            s: Data = S[idx]
            k: Data = K[idx]
            l: Data = L[idx]

            # Infer N
            N = p.x.size(0)
            if N != n_rois:
                raise ValueError(f"N mismatch at idx={idx}: expected {n_rois}, P.x rows={N}")

            # Extract lag blocks (N×N each)
            F_lag, F_rev = _extract_lag_blocks(l.x, N)

            # Node features: concat along feature dimension → N × (5N)
            x = torch.cat([p.x, s.x, k.x, F_lag, F_rev], dim=1).to(torch.float32)

            # Edge index and edge features
            if edge_source == "pearson":
                edge_index = p.edge_index
            elif edge_source == "union":
                edge_index = _edge_union(p.edge_index, s.edge_index, k.edge_index)
            else:
                raise ValueError("--edge_source must be 'pearson' or 'union'")

            set_p = _edge_set(p.edge_index)
            set_s = _edge_set(s.edge_index)
            set_k = _edge_set(k.edge_index)
            edge_attr = _build_edge_attr(edge_index, set_p, set_s, set_k)

            # Labels: keep from Pearson (all should match)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=p.y)
        except Exception as e:
            print(f"[ERROR] sample {idx}: {e}")
            return None

    if n_jobs == 1:
        fused = [ _do_one(i) for i in tqdm(range(len(P))) ]
    else:
        fused = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_do_one)(i) for i in tqdm(range(len(P)))
        )

    fused = [d for d in fused if d is not None]
    if not fused:
        raise RuntimeError("No samples were successfully fused.")

    # 4) Save
    _ = _SaveOnlyDataset(out_root, out_name, fused)
    print(f"[OK] Saved fused dataset to: {os.path.join(out_root, 'processed', out_name + '.pt')}")


# ------------------------------- CLI -------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Fuse edge features (P/S/K) and node features (P,S,K,Lag,Lag_rev).")
    ap.add_argument("--name", type=str, required=True, help="Dataset name, e.g., HCPGender")
    ap.add_argument("--n_rois", type=int, required=True, help="Number of ROIs, e.g., 1000")
    ap.add_argument("--lag", type=int, required=True, help="Lag L used to locate the lag dataset")
    ap.add_argument("--out_root", type=str, required=True, help="Output root; will write <out_root>/processed/<out_name>.pt")
    ap.add_argument("--out_name", type=str, required=True, help="Output dataset filename (without .pt)")
    ap.add_argument("--edge_source", type=str, choices=["pearson", "union"], default="pearson",
                    help="Use Pearson edges only (default) or union of P/S/K")
    ap.add_argument("--base_dir", type=str, default="data", help="Base dir where rs_{N} lives (default: data)")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel workers for fusion (default: 1)")
    args = ap.parse_args()

    fuse_edge_features(
        name=args.name,
        n_rois=args.n_rois,
        lag=args.lag,
        out_root=args.out_root,
        out_name=args.out_name,
        edge_source=args.edge_source,
        base_dir=args.base_dir,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unify topology (global/shared graph) from an existing PyG dataset.

------------
Given an existing PyG dataset file <in_root>/processed/<in_name>.pt
(where each graph already has edge_index and x as an N×N FC matrix),
this script aggregates the per-subject adjacencies, selects the most
frequently occurring edges (by appearance count across subjects), and
replaces every graph's edge_index with this unified topology. It then
saves a new PyG dataset to <out_root>/processed/<out_name>.pt with the
same x and y per subject but a shared edge_index.

Selection rule
--------------
Specify either:
- --edge_pct P : keep the top P% of possible undirected edges (default)
- --top_k K    : keep the top K undirected edges (mutually exclusive)

Inputs
------------------
- <in_root>/processed/<in_name>.pt : a PyG (data, slices) file produced by your earlier pipeline.

Outputs
-------
- <out_root>/processed/<out_name>.pt : a PyG (data, slices) with unified topology.

Example
-------
python src/topology/unify_topology.py \
  --in_root data/rs_100/rs_100_pearson \
  --in_name HCPGender \
  --out_root data/rs_100/rs_100_pearson_unified \
  --out_name HCPGender_unified \
  --edge_pct 5
"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_dense_adj


# ----------------------------- helpers ------------------------------------ #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _infer_num_nodes(first_graph: Data) -> int:
    """
    Infer number of nodes. In your pipeline, x is N×N (FC matrix).
    """
    if first_graph.x is None:
        raise ValueError("Data.x is None; cannot infer num_nodes.")
    if first_graph.x.dim() != 2 or first_graph.x.size(0) != first_graph.x.size(1):
        raise ValueError(f"Expected x to be an N×N matrix; got shape {tuple(first_graph.x.shape)}.")
    return int(first_graph.x.size(0))


def _to_upper_triu_dense(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Convert edge_index -> dense adjacency (N×N), then keep upper-triangular (excluding diagonal).
    This ensures each undirected edge is counted once.
    """
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)  # [N, N], {0,1}
    A = torch.triu(A, diagonal=1)  # keep i<j
    return A


def _build_unified_edge_index_from_counts(
    counts: torch.Tensor,
    top_k: Optional[int],
    edge_pct: Optional[float],
    undirected: bool = True,
    print_top: int = 0
) -> torch.Tensor:
    """
    Given an [N, N] count matrix over upper triangle (i<j), select top edges.

    Returns a symmetrized edge_index (both directions) if undirected=True.
    """
    N = counts.size(0)
    # Flatten upper-tri entries only by keeping zeros elsewhere
    upper = torch.triu(counts, diagonal=1)

    # Consider only edges that appeared at least once
    flat = upper.flatten()
    pos_mask = flat > 0
    flat_idx = torch.arange(flat.numel(), device=flat.device)
    valid_vals = flat[pos_mask]
    valid_idx = flat_idx[pos_mask]

    # If nothing is present, return empty
    if valid_vals.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long)

    # Decide K
    if top_k is None:
        if edge_pct is None:
            edge_pct = 5.0
        # possible undirected edges = N*(N-1)/2
        possible = N * (N - 1) // 2
        calc_k = int(np.floor((edge_pct / 100.0) * possible))
        calc_k = max(calc_k, 1)
        K = min(calc_k, valid_vals.numel())
    else:
        K = min(int(top_k), valid_vals.numel())

    # Get top-K by count
    top_vals, top_pos = torch.topk(valid_vals, k=K, largest=True, sorted=True)
    top_flat_idx = valid_idx[top_pos]

    # Recover (i, j)
    rows = torch.div(top_flat_idx, N, rounding_mode='floor')
    cols = top_flat_idx % N

    if print_top > 0:
        to_print = min(print_top, K)
        print(f"Top {to_print} unified edges by frequency:")
        for i in range(to_print):
            print(f"  Edge ({rows[i].item()}, {cols[i].item()}) count = {top_vals[i].item()}")

    # Build edge_index (symmetrize if undirected)
    if undirected:
        ij = torch.stack([rows, cols], dim=0)
        ji = torch.stack([cols, rows], dim=0)
        edge_index = torch.cat([ij, ji], dim=1).to(torch.long)
    else:
        edge_index = torch.stack([rows, cols], dim=0).to(torch.long)

    return edge_index


# ------------------ dataset loaders/savers (PyG-style) -------------------- #

class _LoadedInMemoryDataset(InMemoryDataset):
    """
    Minimal loader that exposes __getitem__ over an existing <root>/processed/<name>.pt
    """
    def __init__(self, root: str, name: str):
        self._root = root
        self._name = name
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self._name}.pt"]

    def process(self):
        # Not used; file must already exist.
        raise RuntimeError("This dataset expects an existing processed file.")


class _SaveOnlyDataset(InMemoryDataset):
    """
    Helper to collate and save a list of Data objects into <root>/processed/<name>.pt
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
        # Work is done in __init__
        pass


# --------------------------------- main ----------------------------------- #

def unify_topology(
    in_root: str,
    in_name: str,
    out_root: str,
    out_name: str,
    edge_pct: Optional[float],
    top_k: Optional[int],
    undirected: bool,
    print_top: int
) -> None:
    # 1) Load existing dataset
    ds = _LoadedInMemoryDataset(in_root, in_name)
    if len(ds) == 0:
        raise ValueError("Input dataset is empty.")

    # 2) Infer N and accumulate counts over the upper triangle
    N = _infer_num_nodes(ds[0])
    counts = torch.zeros((N, N), dtype=torch.float32)

    for i in range(len(ds)):
        g: Data = ds[i]
        # Basic sanity checks
        if g.edge_index is None:
            continue
        curA = _to_upper_triu_dense(g.edge_index, num_nodes=N)  # [N, N], {0,1} on i<j
        counts += curA

    # 3) Select most frequent edges
    unified_edge_index = _build_unified_edge_index_from_counts(
        counts=counts,
        top_k=top_k,
        edge_pct=edge_pct,
        undirected=undirected,
        print_top=print_top
    )

    # 4) Apply unified topology to every sample (keep x and y unchanged)
    new_list: List[Data] = []
    for i in range(len(ds)):
        g: Data = ds[i]
        new_list.append(Data(x=g.x, edge_index=unified_edge_index, y=g.y))

    # 5) Save new dataset
    _ = _SaveOnlyDataset(out_root, out_name, new_list)
    print(f"Saved unified dataset to: {os.path.join(out_root, 'processed', out_name + '.pt')}")


def parse_args():
    ap = argparse.ArgumentParser(description="Unify topology across an existing PyG dataset (.pt).")
    ap.add_argument("--in_root", type=str, required=True,
                    help="Input dataset root (expects <in_root>/processed/<in_name>.pt).")
    ap.add_argument("--in_name", type=str, required=True,
                    help="Input dataset name (filename without .pt).")

    ap.add_argument("--out_root", type=str, required=True,
                    help="Output dataset root (will write <out_root>/processed/<out_name>.pt).")
    ap.add_argument("--out_name", type=str, required=True,
                    help="Output dataset name (filename without .pt).")

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--edge_pct", type=float, default=5.0,
                       help="Percent of possible undirected edges to keep (default: 5%%).")
    group.add_argument("--top_k", type=int, default=None,
                       help="Keep exactly top-K undirected edges by frequency (overrides --edge_pct).")

    ap.add_argument("--directed", action="store_true",
                    help="Treat edges as directed (no symmetry). Default: undirected.")
    ap.add_argument("--print_top", type=int, default=0,
                    help="Print the top-N edges by frequency for inspection (default: 0).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    unify_topology(
        in_root=args.in_root,
        in_name=args.in_name,
        out_root=args.out_root,
        out_name=args.out_name,
        edge_pct=None if args.top_k is not None else args.edge_pct,
        top_k=args.top_k,
        undirected=not args.directed,
        print_top=args.print_top
    )

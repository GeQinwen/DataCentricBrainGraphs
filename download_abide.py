'''
This script mainly refers to https://github.com/HennyJie/BrainGB/tree/master/examples/utils/get_abide
'''
import argparse
import os
import re
import glob
import shutil
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from nilearn import datasets

import torch
from torch_geometric.data import InMemoryDataset


# ----------------------------- helpers ------------------------------------ #

def _atlas_from_nrois(n_rois: int) -> str:
    if n_rois == 200:
        return "cc200"
    if n_rois == 400:
        return "cc400"
    raise ValueError("ABIDE PCP provides cc200/cc400; set --n_rois to 200 or 400.")


def _read_id_list(id_file_path: Optional[str]) -> Optional[List[str]]:
    if not id_file_path:
        return None
    with open(id_file_path, "r") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    # Normalize to plain numeric strings (ABIDE subjects are ints in CSV)
    ids = [re.sub(r"[^0-9]", "", x) for x in ids]
    return ids


def _find_phenotypic_csv(root: str) -> Optional[str]:
    base = os.path.join(root, "ABIDE_pcp")
    for pat in ["Phenotypic_V1_0b_preprocessed1.csv",
                "Phenotypic_V1_0b_preprocessed.csv",
                "*Phenotypic*.csv"]:
        hits = glob.glob(os.path.join(base, pat))
        if hits:
            hits.sort(key=len)
            return hits[0]
    return None


def _load_meta(phenotypic_csv: str) -> pd.DataFrame:
    df = pd.read_csv(phenotypic_csv)
    # Normalize subject id column name variants
    sid = None
    for c in ["subject", "SUB_ID", "SUBJECT_ID", "Subject"]:
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise ValueError(f"Cannot find subject id column in {phenotypic_csv}")
    keep_cols = [sid]
    for c in ["DX_GROUP", "SITE_ID"]:
        if c in df.columns:
            keep_cols.append(c)
    df = df[keep_cols].copy()
    df.rename(columns={sid: "SUBJECT_ID"}, inplace=True)
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(str)
    # strip non-digits to match our normalized ids
    df["SUBJECT_ID"] = df["SUBJECT_ID"].str.replace(r"[^0-9]", "", regex=True)
    return df


def _ensure_subject_folders(data_folder: str, subject_ids: List[str]):
    for sid in subject_ids:
        p = os.path.join(data_folder, sid)
        os.makedirs(p, exist_ok=True)


def _extract_sid_from_path_or_name(path: str) -> Optional[str]:
    """
    Try to infer 5-7 digit subject id from filename first (anywhere),
    then from any path token.
    """
    base = os.path.basename(path)
    m = re.search(r"(\d{5,7})", base)
    if m:
        return m.group(1)

    parts = path.replace(os.sep, "/").split("/")
    for token in reversed(parts):
        m = re.fullmatch(r"\d{5,7}", token)
        if m:
            return m.group(0)
        # also allow numbers embedded in token
        m = re.search(r"(\d{5,7})", token)
        if m:
            return m.group(1)
    return None


def _move_into_subject_folders(data_folder: str, atlas: str):
    """
    Place rois_*.1D and (if found) func_preproc.nii.gz under data_folder/<SID>/

    IMPORTANT FIX:
    - Use '**/*rois_{atlas}.1D' instead of '**/rois_{atlas}.1D' to catch
      files like 'NYU_0050003_rois_cc200.1D'.
    - Infer SID from filename anywhere, not only numeric prefix.
    """
    rois_paths = glob.glob(
        os.path.join(data_folder, "**", f"*rois_{atlas}.1D"),
        recursive=True
    )
    for rp in rois_paths:
        sid = _extract_sid_from_path_or_name(rp)
        if sid is None:
            continue

        dest_dir = os.path.join(data_folder, sid)
        os.makedirs(dest_dir, exist_ok=True)

        # Move rois file
        dest_rois = os.path.join(dest_dir, f"rois_{atlas}.1D")
        if os.path.abspath(rp) != os.path.abspath(dest_rois):
            try:
                shutil.move(rp, dest_rois)
            except Exception:
                # likely already moved; ignore
                pass

        # Try to move func_preproc if sitting near the original rois file
        base_dir = os.path.dirname(rp)
        cand = os.path.join(base_dir, "func_preproc.nii.gz")
        if os.path.exists(cand):
            dest_func = os.path.join(dest_dir, "func_preproc.nii.gz")
            if os.path.abspath(cand) != os.path.abspath(dest_func):
                try:
                    shutil.move(cand, dest_func)
                except Exception:
                    pass


def _load_timeseries(rois_file: str) -> np.ndarray:
    """Load ABIDE PCP rois_*.1D → return T×N array."""
    arr = np.loadtxt(rois_file)
    if arr.ndim == 1:
        arr = arr[:, None]
    # Keep as T x N
    return arr


def _standardize_time_series(Y: np.ndarray) -> np.ndarray:
    # Per-ROI standardization over time
    mu = np.nanmean(Y, axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (Y - mu) / sd


# ------------------------- Main dataset class ----------------------------- #

class Brain_Connectome_ABIDE_Download(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        n_rois: int,
        threshold: int,
        path_to_data: str,
        n_jobs: int,
        pipeline: str,
        download: bool,
        id_file_path: Optional[str] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.dataset_name = name
        self.n_rois = n_rois
        self.threshold = threshold  # kept for API symmetry
        self.target_path = path_to_data
        self.n_jobs = n_jobs
        self.pipeline = pipeline
        self.download_flag = download
        self.id_file_path = id_file_path
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return [self.dataset_name + ".pt"]

    def process(self):
        atlas = _atlas_from_nrois(self.n_rois)

        # 1) Fetch (optional)
        if self.download_flag:
            datasets.fetch_abide_pcp(
                data_dir=self.root,
                pipeline=self.pipeline,
                band_pass_filtering=True,
                global_signal_regression=False,
                derivatives=[f"rois_{atlas}", "func_preproc"],
                quality_checked=False,
            )

        # 2) Organize CPAC folder
        data_folder = os.path.join(self.root, "ABIDE_pcp", self.pipeline, "filt_noglobal")
        os.makedirs(data_folder, exist_ok=True)

        # 3) Subject list
        subset_ids = _read_id_list(self.id_file_path)
        if subset_ids is None:
            found_rois = glob.glob(
                os.path.join(data_folder, "**", f"*rois_{atlas}.1D"),
                recursive=True
            )
            subset_ids = []
            for rp in found_rois:
                sid = _extract_sid_from_path_or_name(rp)
                if sid:
                    subset_ids.append(sid)

        subset_ids = sorted(list(set(subset_ids)))
        _ensure_subject_folders(data_folder, subset_ids)

        # 4) Move files into subject folders (best-effort)
        _move_into_subject_folders(data_folder, atlas)

        # 5) Phenotypic meta → csv (SUBJECT_ID, DX_GROUP, SITE_ID)
        phen_csv = _find_phenotypic_csv(self.root)
        meta_df = _load_meta(phen_csv) if phen_csv else pd.DataFrame({"SUBJECT_ID": subset_ids})
        if not meta_df.empty:
            meta_out = os.path.join(self.target_path, "abide_meta.csv")
            os.makedirs(self.target_path, exist_ok=True)
            meta_df.to_csv(meta_out, index=False)

        # 6) Extract + standardize time series; save as .npy (T×N)
        out_dir = os.path.join(self.target_path, f"time_series_{self.n_rois}")
        os.makedirs(out_dir, exist_ok=True)

        def _process_one(sid: str) -> Optional[str]:
            try:
                # Preferred path after moving
                rois_file = os.path.join(data_folder, sid, f"rois_{atlas}.1D")
                if not os.path.exists(rois_file):
                    # Fallback search (FIX): find any file that contains this SID and matches rois_{atlas}.1D
                    cand_list = glob.glob(
                        os.path.join(data_folder, "**", f"*{sid}*rois_{atlas}.1D"),
                        recursive=True
                    )
                    if cand_list:
                        rois_file = cand_list[0]
                    else:
                        return f"{sid}: rois file missing"

                Y = _load_timeseries(rois_file)  # T×N
                if Y.shape[1] != self.n_rois:
                    return f"{sid}: ROI mismatch (got {Y.shape[1]}, expect {self.n_rois})"
                Ystd = _standardize_time_series(Y)
                out_path = os.path.join(out_dir, f"{sid}_rois_{atlas}_time_series.npy")
                np.save(out_path, Ystd)
                return None
            except Exception as e:
                return f"{sid}: {e}"

        errs = Parallel(n_jobs=self.n_jobs)(
            delayed(_process_one)(sid) for sid in tqdm(subset_ids)
        )
        errs = [e for e in errs if e]
        if errs:
            with open(os.path.join(self.target_path, "abide_preprocess_errors.log"), "w") as f:
                f.write("\n".join(errs))

        # 7) Save minimal .pt marker for PyG-style convention
        torch.save(({}, {}), self.processed_paths[0])


# --------------------------------- CLI ----------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Download & preprocess ABIDE (PCP/CPAC) into ROI time series (.npy)")
    p.add_argument('--n_rois', type=int, default=200, help='Number of ROIs (200 or 400) in ABIDE PCP atlas')
    p.add_argument('--root', type=str, default='data', help='Root directory (Nilearn will place ABIDE_pcp here)')
    p.add_argument('--name', type=str, default='ABIDE', help='Dataset name (used for marker .pt)')
    p.add_argument('--threshold', type=int, default=5, help='Kept for API parity; not used here')
    p.add_argument('--path_to_data', type=str, default='data/raw/ABIDE', help='Where to put outputs (time_series_*)')
    p.add_argument('--n_jobs', type=int, default=1, help='Parallel workers for saving')
    p.add_argument('--pipeline', type=str, default='cpac', help='ABIDE PCP pipeline (default: cpac)')
    p.add_argument('--download', type=lambda s: s.lower() in ['1','true','yes','y'], default=True, help='Fetch with Nilearn (default: True)')
    p.add_argument('--id_file_path', type=str, default=None, help='Optional path to subject_IDs.txt to subset')

    args = p.parse_args()

    Brain_Connectome_ABIDE_Download(
        root=args.root,
        name=args.name,
        n_rois=args.n_rois,
        threshold=args.threshold,
        path_to_data=args.path_to_data,
        n_jobs=args.n_jobs,
        pipeline=args.pipeline,
        download=args.download,
        id_file_path=args.id_file_path,
    )


if __name__ == '__main__':
    main()

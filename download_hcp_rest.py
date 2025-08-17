import argparse
import os
import boto3
from torch_geometric.data import InMemoryDataset
import torch
import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import load_img
from scipy.stats import zscore
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import pickle

class Brain_Connectome_Rest_Download(InMemoryDataset):
    def __init__(self, root, name, n_rois, threshold, path_to_data, n_jobs, s3,
                 transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name, self.n_rois, self.threshold = root, name, n_rois, threshold
        self.target_path, self.n_jobs, self.s3 = path_to_data, n_jobs, s3
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name + '.pt']

    def extract_from_3d_no(self, volume, fmri):
        subcor_ts = []
        for i in np.unique(volume):
            if i != 0:
                bool_roi = volume == i
                roi_ts_mean = [np.mean(fmri[..., t][bool_roi]) for t in range(fmri.shape[-1])]
                subcor_ts.append(np.array(roi_ts_mean))
        return np.array(subcor_ts).T

    def construct_Adj_postive_perc(self, corr):
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0],
                                  100 - self.threshold)
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy

    def get_data_obj(self, iid, behavioral_data, BUCKET_NAME, volume):
        try:
            mri_file_path = f"HCP_1200/{iid}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
            reg_path = f"HCP_1200/{iid}/MNINonLinear/Results/rfMRI_REST1_LR/Movement_Regressors.txt"

            local_mri = os.path.join(self.target_path, f"{iid}_{os.path.basename(mri_file_path)}")
            local_reg = os.path.join(self.target_path, f"{iid}_{os.path.basename(reg_path)}")

            if not os.path.exists(local_mri):
                self.s3.download_file(BUCKET_NAME, mri_file_path, local_mri)
            if not os.path.exists(local_reg):
                self.s3.download_file(BUCKET_NAME, reg_path, local_reg)

            img = nib.load(local_mri)
            if img.shape[3] < 1200:
                return None
            regs = np.loadtxt(local_reg)
            fmri = img.get_fdata()

            Y = self.extract_from_3d_no(volume, fmri)

            # detrending
            t = np.arange(1, Y.shape[0] + 1)
            tzd = zscore(np.vstack((t, t**2)), axis=1)
            XX = np.vstack((np.ones(Y.shape[0]), tzd))
            B = np.matmul(np.linalg.pinv(XX).T, Y)
            Yt = Y - np.matmul(XX.T, B)

            # regress out head motion
            B2 = np.matmul(np.linalg.pinv(regs), Yt)
            Ytm = Yt - np.matmul(regs, B2)

            zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)

            ts_path = os.path.join(self.target_path, f"time_series_{self.n_rois}",
                                   f"{iid}_{os.path.basename(mri_file_path).split('.')[0]}_time_series.npy")
            os.makedirs(os.path.dirname(ts_path), exist_ok=True)
            np.save(ts_path, zd_Ytm)
            print(f"Saved: {ts_path}")

        except Exception as e:
            print(f"Error processing {iid}: {e}")
            return None
        return None

    def process(self):
        behavioral_df = pd.read_csv(os.path.join(self.root, 'HCP_behavioral.csv')).set_index('Subject')[
            ['Gender', 'Age', 'ListSort_AgeAdj', 'PMAT24_A_CR']
        ]
        mapping = {'22-25': 0, '26-30': 1, '31-35': 2, '36+': 3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)

        with open(os.path.join(self.root, "ids.pkl"), 'rb') as f:
            ids = pickle.load(f)

        BUCKET_NAME = 'hcp-openaccess'
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois, yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()

        Parallel(n_jobs=self.n_jobs)(
            delayed(self.get_data_obj)(iid, behavioral_df, BUCKET_NAME, volume)
            for iid in tqdm(ids)
        )

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess HCP resting-state data")
    parser.add_argument('--n_rois', type=int, default=100, help='Number of ROIs in Schaefer atlas')
    parser.add_argument('--root', type=str, default='data', help='Root directory')
    parser.add_argument('--name', type=str, default='HCPGender', help='Dataset name')
    parser.add_argument('--threshold', type=int, default=5, help='Top % positive edges')
    parser.add_argument('--path_to_data', type=str, default='data/raw/HCPGender', help='Path to store raw scans')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--access_key', type=str, required=True, help='Your ConnectomeDB AWS access key')
    parser.add_argument('--secret_key', type=str, required=True, help='Your ConnectomeDB AWS secret key')

    args = parser.parse_args()

    s3 = boto3.client('s3',
                      aws_access_key_id=args.access_key,
                      aws_secret_access_key=args.secret_key)

    Brain_Connectome_Rest_Download(
        args.root, args.name, args.n_rois, args.threshold,
        args.path_to_data, args.n_jobs, s3
    )

if __name__ == "__main__":
    main()

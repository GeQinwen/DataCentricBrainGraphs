import pandas as pd
import os
import nibabel as nib
import pickle
import numpy as np
import argparse
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import load_img
from scipy.stats import zscore
import torch
from torch_geometric.data import InMemoryDataset
import itertools
import boto3
import random
from pathos.multiprocessing import ProcessingPool as Pool


def worker_function(args):
    iid, BUCKET_NAME, volume, target_path, access_key, secret_key, n_roi = args
    return Brain_Connectome_Task_Download.get_data_obj_task(iid, BUCKET_NAME, volume, target_path, access_key, secret_key, n_roi)


class Brain_Connectome_Task_Download(InMemoryDataset):
    def __init__(self, root, dataset_name, n_rois, threshold, path_to_data, n_jobs,
                 access_key, secret_key, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataset_name = dataset_name
        self.n_rois = n_rois
        self.threshold = threshold
        self.target_path = path_to_data
        self.n_jobs = n_jobs
        self.access_key = access_key
        self.secret_key = secret_key
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name + '.pt']

    @staticmethod
    def get_data_obj_task(iid, BUCKET_NAME, volume, target_path, access_key, secret_key, n_roi):
        try:
            all_paths = [
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR.nii.gz",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_GAMBLING_LR/tfMRI_GAMBLING_LR.nii.gz",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR.nii.gz",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_RELATIONAL_LR/tfMRI_RELATIONAL_LR.nii.gz",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_SOCIAL_LR/tfMRI_SOCIAL_LR.nii.gz",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR.nii.gz"
            ]
            reg_paths = [
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_EMOTION_LR/Movement_Regressors.txt",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_GAMBLING_LR/Movement_Regressors.txt",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_LANGUAGE_LR/Movement_Regressors.txt",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_MOTOR_LR/Movement_Regressors.txt",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_RELATIONAL_LR/Movement_Regressors.txt",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_SOCIAL_LR/Movement_Regressors.txt",
                f"HCP_1200/{iid}/MNINonLinear/Results/tfMRI_WM_LR/Movement_Regressors.txt"
            ]

            s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
            data_list = []

            for y, path in enumerate(all_paths):
                try:
                    ts_path = os.path.join(target_path, f"time_series_{n_roi}",
                                           f"{iid}_{os.path.basename(path).split('.')[0]}_time_series.npy")
                    if not os.path.exists(ts_path):
                        print("Downloading for IID:", iid)
                        s3.download_file(BUCKET_NAME, path, os.path.join(target_path, f"{iid}_{os.path.basename(path)}"))
                        rnd = random.randint(0, 1000)
                        reg_prefix = f"{iid}{rnd}"
                        s3.download_file(BUCKET_NAME, reg_paths[y],
                                         os.path.join(target_path, f"{reg_prefix}_{os.path.basename(reg_paths[y])}"))

                        image_path_LR = os.path.join(target_path, f"{iid}_{os.path.basename(path)}")
                        reg_path = os.path.join(target_path, f"{reg_prefix}_{os.path.basename(reg_paths[y])}")

                        img = nib.load(image_path_LR)
                        regs = np.loadtxt(reg_path)
                        fmri = img.get_fdata()
                        Y = Brain_Connectome_Task_Download.extract_from_3d_no(volume, fmri)

                        start = 1
                        stop = Y.shape[0]
                        step = 1
                        t = np.arange(start, stop + step, step)
                        tzd = zscore(np.vstack((t, t ** 2)), axis=1)
                        XX = np.vstack((np.ones(Y.shape[0]), tzd))
                        B = np.matmul(np.linalg.pinv(XX).T, Y)
                        Yt = Y - np.matmul(XX.T, B)

                        B2 = np.matmul(np.linalg.pinv(regs), Yt)
                        Ytm = Yt - np.matmul(regs, B2)

                        zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)

                        os.makedirs(os.path.dirname(ts_path), exist_ok=True)
                        np.save(ts_path, zd_Ytm)
                except:
                    print("file skipped!")
        except:
            return None
        return data_list

    @staticmethod
    def extract_from_3d_no(volume, fmri):
        subcor_ts = []
        for i in np.unique(volume):
            if i != 0:
                bool_roi = (volume == i)
                roi_ts_mean = [np.mean(fmri[:, :, :, t][bool_roi]) for t in range(fmri.shape[-1])]
                subcor_ts.append(np.array(roi_ts_mean))
        return np.array(subcor_ts).T

    def process(self):
        BUCKET_NAME = 'hcp-openaccess'
        with open(os.path.join("data", "ids.pkl"), 'rb') as f:
            ids = pickle.load(f)

        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois, yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()

        tasks = [(iid, BUCKET_NAME, volume, self.target_path, self.access_key, self.secret_key, self.n_rois) for iid in ids]
        with Pool(self.n_jobs) as pool:
            data_list = pool.map(worker_function, tasks)

        dataset = list(itertools.chain(*[d for d in data_list if d is not None]))
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process HCP State dataset")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--path_to_data", type=str, required=True)
    parser.add_argument("--n_rois", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--access_key", type=str, required=True)
    parser.add_argument("--secret_key", type=str, required=True)

    args = parser.parse_args()

    dataset = Brain_Connectome_Task_Download(
        root=args.root,
        dataset_name=args.name,
        n_rois=args.n_rois,
        threshold=args.threshold,
        path_to_data=args.path_to_data,
        n_jobs=args.n_jobs,
        access_key=args.access_key,
        secret_key=args.secret_key
    )

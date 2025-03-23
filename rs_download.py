import pandas as pd
import os
import nibabel as nib
import pickle
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import load_img
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import zscore
import torch
from torch_geometric.data import Data,InMemoryDataset
from random import randrange
import math
import zipfile
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import random
from pathos.multiprocessing import ProcessingPool as Pool

#Download and preprocess static datasets
# from NeuroGraph.preprocess import Brain_Connectome_Rest_Download
import boto3

root = "data/"
name = "HCPGender"
threshold = 5
path_to_data = "data/raw/HCPGender"  # store the raw downloaded scans
n_rois = 100
n_jobs = 1 # this script runs in parallel and requires the number of jobs is an input


ACCESS_KEY = ''  # your connectomeDB credentials
SECRET_KEY = ''
s3 = boto3.client('s3',
                  aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)
# this function requires both HCP_behavioral.csv and ids.pkl files under the root directory. Both files have been provided and can be found under the data directory
# rest_dataset = Brain_Connectome_Rest_Download(root, name, n_rois, threshold,
#   path_to_data, n_jobs, s3)

class Brain_Connectome_Rest_Download(InMemoryDataset):

    def __init__(self,
                 root,
                 name,
                 n_rois,
                 threshold,
                 path_to_data,
                 n_jobs,
                 s3,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.root, self.dataset_name, self.n_rois, self.threshold, self.target_path, self.n_jobs, self.s3 = root, name, n_rois, threshold, path_to_data, n_jobs, s3
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name + '.pt']

    def extract_from_3d_no(self, volume, fmri):
        ''' 
        Extract time-series data from a 3d atlas with non-overlapping ROIs.
        
        Inputs:
            path_to_atlas = '/path/to/atlas.nii.gz'
            path_to_fMRI = '/path/to/fmri.nii.gz'
            
        Output:
            returns extracted time series # volumes x # ROIs
        '''

        subcor_ts = []
        for i in np.unique(volume):
            if i != 0:
                #             print(i)
                bool_roi = np.zeros(volume.shape, dtype=int)
                bool_roi[volume == i] = 1
                bool_roi = bool_roi.astype(bool)
                #             print(bool_roi.shape)
                # extract time-series data for each roi
                roi_ts_mean = []
                for t in range(fmri.shape[-1]):
                    roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
                subcor_ts.append(np.array(roi_ts_mean))
        Y = np.array(subcor_ts).T
        return Y

    def construct_Adj_postive_perc(self, corr):
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0],
                                  100 - self.threshold)
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy

    def get_data_obj(self, iid, behavioral_data, BUCKET_NAME, volume):
        try:
            mri_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
            reg_path = "HCP_1200/" + iid + '/MNINonLinear/Results/rfMRI_REST1_LR/Movement_Regressors.txt'
            if not os.path.exists(os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path))):
                self.s3.download_file(BUCKET_NAME, mri_file_path,os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path)))
            if not os.path.exists(os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path))):
                self.s3.download_file(BUCKET_NAME, reg_path,os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path)))
            image_path_LR = os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path))
            reg_path = os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path))
            img = nib.load(image_path_LR)
            if img.shape[3] < 1200:
                return None
            regs = np.loadtxt(reg_path)
            fmri = img.get_fdata()
            Y = self.extract_from_3d_no(volume, fmri)
            start = 1
            stop = Y.shape[0]
            step = 1
            # detrending
            t = np.arange(start, stop + step, step)
            tzd = zscore(np.vstack((t, t**2)), axis=1)
            XX = np.vstack((np.ones(Y.shape[0]), tzd))
            B = np.matmul(np.linalg.pinv(XX).T, Y)
            Yt = Y - np.matmul(XX.T, B)
            # regress out head motion regressors
            B2 = np.matmul(np.linalg.pinv(regs), Yt)
            Ytm = Yt - np.matmul(regs, B2)
            # zscore over axis=0 (time)
            zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(
                Ytm, axis=0, ddof=1)
            ts_path = os.path.join(self.target_path+"/time_series_100", iid+"_"+os.path.basename(mri_file_path).split(".")[0]+"_time_series.npy")
            os.makedirs(os.path.dirname(ts_path), exist_ok=True)
            np.save(ts_path,zd_Ytm)
            print("saving path:", ts_path)
            # os.remove(image_path_LR)
            # os.remove(reg_path)
        except:
            return None
        return None


#         ...

    def process(self):
        behavioral_df = pd.read_csv(
            os.path.join(self.root,
                         'HCP_behavioral.csv')).set_index('Subject')[[
                             'Gender', 'Age', 'ListSort_AgeAdj', 'PMAT24_A_CR'
                         ]]
        mapping = {'22-25': 0, '26-30': 1, '31-35': 2, '36+': 3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)

        dataset = []
        BUCKET_NAME = 'hcp-openaccess'
        with open(os.path.join(self.root, "ids.pkl"), 'rb') as f:
            ids = pickle.load(f)

        print(len(ids))
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,
                                        yeo_networks=17,
                                        resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        data_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self.get_data_obj)(iid, behavioral_df, BUCKET_NAME, volume)
            for iid in tqdm(ids))
        dataset = [x for x in data_list if x is not None]
        # print(len(dataset))
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:", self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
rest_dataset = Brain_Connectome_Rest_Download(root, name, n_rois, threshold,path_to_data, n_jobs, s3)
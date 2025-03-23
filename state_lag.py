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
import torch
import NeuroGraph
from NeuroGraph import preprocess
from torch_geometric.data import Data,InMemoryDataset
import boto3
import random
from pathos.multiprocessing import ProcessingPool as Pool

def worker_function(args):
    # Unpack the arguments that were prepared for each task
    iid, BUCKET_NAME, volume,lag = args
    
    # Directly call the static processing method with the unpacked arguments
    return Brain_Connectome_Task_Download.get_data_obj_task(iid,BUCKET_NAME, volume,lag)


class Brain_Connectome_Task_Download(InMemoryDataset):
    
    def __init__(self, root, dataset_name,n_rois, threshold,path_to_data,n_jobs,s3, lag, transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.threshold,self.target_path,self.n_jobs,self.s3,self.lag = root, dataset_name,n_rois,threshold,path_to_data,n_jobs,s3,lag
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']

    @staticmethod
    def get_data_obj_task(iid,BUCKET_NAME,volume,lag):
        try:
            target_path = "data/raw/HCPState"
            emotion_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR.nii.gz"
            reg_emo_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_EMOTION_LR/Movement_Regressors.txt'

            gambling_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_GAMBLING_LR/tfMRI_GAMBLING_LR.nii.gz"
            reg_gamb_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_GAMBLING_LR/Movement_Regressors.txt'

            language_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR.nii.gz"
            reg_lang_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_LANGUAGE_LR/Movement_Regressors.txt'

            motor_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz"
            reg_motor_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_MOTOR_LR/Movement_Regressors.txt'

            relational_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_RELATIONAL_LR/tfMRI_RELATIONAL_LR.nii.gz"
            reg_rel_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_RELATIONAL_LR/Movement_Regressors.txt'

            social_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_SOCIAL_LR/tfMRI_SOCIAL_LR.nii.gz"
            reg_soc_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_SOCIAL_LR/Movement_Regressors.txt'

            wm_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR.nii.gz"
            reg_wm_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_WM_LR/Movement_Regressors.txt'
            all_paths = [emotion_path,gambling_path,language_path,motor_path,relational_path,social_path,wm_path]
            reg_paths = [reg_emo_path,reg_gamb_path,reg_lang_path,reg_motor_path,reg_rel_path,reg_soc_path,reg_wm_path]
            data_list = []
            
            for y, path in enumerate(all_paths):
                try:
                    ts_path = os.path.join(target_path+"/time_series_400", iid+"_"+os.path.basename(path).split(".")[0]+"_time_series.npy")
                    zd_Ytm = np.load(ts_path)
                    
                    positive_threshold_percentage = 30
                    positive_threshold_value = np.percentile(zd_Ytm, 100 - positive_threshold_percentage)
                    #positive_threshold_value = 1
                    zd_Ytm[zd_Ytm < positive_threshold_value] = 0
                    
                    expanded_corr_matrix = Brain_Connectome_Task_Download.construct_expanded_lagged_corr(zd_Ytm, lag)
                    
                    #lag_corr = expanded_corr_matrix[0:1000, 1000:2000]
                    lag_corr = expanded_corr_matrix[0:400, 400:800]
                    #lag_corr = expanded_corr_matrix[0:100, 100:200]
                    np.fill_diagonal(lag_corr, 0)
                    
                    #lag_corr_reverse = expanded_corr_matrix[1000:2000, 0:1000]
                    lag_corr_reverse = expanded_corr_matrix[400:800, 0:400]
                    #lag_corr_reverse = expanded_corr_matrix[100:200, 0:100]
                    np.fill_diagonal(lag_corr_reverse, 0)
                    
                    conn = ConnectivityMeasure(kind='correlation')
                    zd_fc = conn.fit_transform([zd_Ytm])[0]
                    np.fill_diagonal(zd_fc, 0)
                    corr_original = torch.tensor(zd_fc).to(torch.float)
                    A = Brain_Connectome_Task_Download.construct_Adj_postive_perc(corr_original, graph_threshold=5)
                    edge_index = A.nonzero().t().to(torch.long)
                    concat_matrix = np.concatenate((zd_fc, lag_corr,lag_corr_reverse), axis=1)
                    corr = torch.tensor(concat_matrix).to(torch.float)
                    data = Data(x = corr, edge_index=edge_index, y = y)
                    data_list.append(data)

                except:
                    print("file skipped!") 
        except:
            return None   
        return data_list
    
    def extract_from_3d_no(self,volume, fmri):
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
                bool_roi = np.zeros(volume.shape, dtype=int)
                bool_roi[volume == i] = 1
                bool_roi = bool_roi.astype(bool)
                roi_ts_mean = []
                for t in range(fmri.shape[-1]):
                    roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
                subcor_ts.append(np.array(roi_ts_mean))
        Y = np.array(subcor_ts).T
        return Y

    @staticmethod
    def construct_Adj_postive_perc(corr,graph_threshold):
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - graph_threshold)
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy
    
    @staticmethod
    def expand_time_series(time_series, lag):
        #time_series shape = (1200, 400) i.e. (timepoints, roi)
        expanded_ts = []
        num_time_points, num_rois = time_series.shape
        #ts_length = num_time_points - lag
        truncated_time_series = time_series[:-lag]
        lagged_time_series = time_series[lag:]
        expanded_ts.append(truncated_time_series)
        print("lagged_time_series", lagged_time_series.shape)
        print("truncated_time_series", truncated_time_series.shape)
        expanded_ts.append(lagged_time_series)
        return np.concatenate(expanded_ts, axis=1)

    @staticmethod
    def construct_expanded_correlation_matrix(expanded_ts):
        conn = ConnectivityMeasure(kind='correlation')
        corr_matrix = conn.fit_transform([expanded_ts])[0]
        np.fill_diagonal(corr_matrix, 0)
        return corr_matrix

    @staticmethod
    def construct_expanded_lagged_corr(time_series, lag):
        #for i in range(num_lag):
        expanded_ts = Brain_Connectome_Task_Download.expand_time_series(time_series, lag)
        expanded_corr_matrix = Brain_Connectome_Task_Download.construct_expanded_correlation_matrix(expanded_ts)
        print("expanded_corr_matrix", expanded_corr_matrix.shape)

        return expanded_corr_matrix
    
    
    def process(self):
        dataset = []
        BUCKET_NAME = 'hcp-openaccess'
        with open(os.path.join("data/","ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        #ids = ids[:2]
        
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        lag = self.lag
        
        tasks = [(iid, BUCKET_NAME, volume,lag) for iid in ids]
        with Pool(self.n_jobs) as pool:
            data_list = pool.map(worker_function, tasks)

        #dataset = [x for x in data_list if x is not None]
        #data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj_task)(iid,BUCKET_NAME,volume,lag) for iid in tqdm(ids))
        print("length of data list:", len(data_list))       
        dataset = list(itertools.chain(*data_list))
        
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])  


for i in range(3,9):
    root = "data/state_400/state_400_thre30_"+str(i)+"lag/"
    lag = i
    name = "HCPState"
    threshold = 5
    path_to_data = "data/raw/HCPState"  # store the raw downloaded scans
    n_rois = 400
    n_jobs = 25 # this script runs in parallel and requires the number of jobs is an input

    ACCESS_KEY = ''  # your connectomeDB credentials
    SECRET_KEY = ''
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    # this function requires both HCP_behavioral.csv and ids.pkl files under the root directory. Both files have been provided and can be found under the data directory
    state_dataset = Brain_Connectome_Task_Download(root, name,n_rois, threshold,path_to_data,n_jobs,s3,lag)
    print(state_dataset)
    print(state_dataset[0])
    
    

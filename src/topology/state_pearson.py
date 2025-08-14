import pandas as pd
import os
import nibabel as nib
import pickle
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import load_img
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
from scipy.stats import rankdata
#from connectivity_matrices import KendallConnectivityMeasure
from pathos.multiprocessing import ProcessingPool as Pool
from nilearn.connectome import ConnectivityMeasure

def worker_function(args):
    # Unpack the arguments that were prepared for each task
    iid, BUCKET_NAME, volume = args
    
    # Directly call the static processing method with the unpacked arguments
    return Brain_Connectome_Task_Download.get_data_obj_task(iid, BUCKET_NAME, volume)



class Brain_Connectome_Task_Download(InMemoryDataset):
    
    def __init__(self, root, dataset_name,n_rois, threshold,path_to_data,n_jobs,s3,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.threshold,self.target_path,self.n_jobs,self.s3 = root, dataset_name,n_rois,threshold,path_to_data,n_jobs,s3
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']

    @staticmethod
    def get_data_obj_task(iid,BUCKET_NAME,volume):
        try:
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
            target_path = "data/raw/HCPState"
            
            for y, path in enumerate(all_paths):
                try:
                    ts_path = os.path.join(target_path+"/time_series_1000", iid+"_"+os.path.basename(path).split(".")[0]+"_time_series.npy")
                    zd_Ytm = np.load(ts_path)
                    
                    #positive_threshold_percentage = 30
                    #positive_threshold_value = np.percentile(zd_Ytm, 100 - positive_threshold_percentage)
                    positive_threshold_value = 1
                    zd_Ytm[zd_Ytm < positive_threshold_value] = 0
                    zd_Ytm[zd_Ytm >= positive_threshold_value] = 1
                    
                    conn = ConnectivityMeasure(kind='correlation')
                    zd_fc = conn.fit_transform([zd_Ytm])[0]
                    np.fill_diagonal(zd_fc, 0)
                    corr = torch.tensor(zd_fc).to(torch.float)
                    A = Brain_Connectome_Task_Download.construct_Adj_postive_perc(corr,graph_threshold=5)
                    edge_index = A.nonzero().t().to(torch.long)
                
                    data = Data(x = corr, edge_index=edge_index, y = y)
                    data_list.append(data)

                except:
                    print("file skipped!") 
        except:
            return None   
        return data_list
    
    @staticmethod
    def extract_from_3d_no(volume, fmri):
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
    
    def process(self):
        dataset = []
        BUCKET_NAME = 'hcp-openaccess'
        with open(os.path.join("data/","ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        #ids = ids[:2]

        
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        #data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj_task)(iid,BUCKET_NAME,volume) for iid in tqdm(ids))
        #print("length of data list:", len(data_list))
        tasks = [(iid, BUCKET_NAME, volume) for iid in ids]
        with Pool(self.n_jobs) as pool:
            data_list = pool.map(worker_function, tasks)       
        dataset = list(itertools.chain(*data_list))
        
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])  

root = "data/state_1000/state_1000_sd1=1_pearson/"
name = "HCPState"
threshold = 5
path_to_data = "data/raw/HCPState"  # store the raw downloaded scans
n_rois = 1000
n_jobs = 30 # this script runs in parallel and requires the number of jobs is an input

ACCESS_KEY = ''  # your connectomeDB credentials
SECRET_KEY = ''
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
# this function requires both HCP_behavioral.csv and ids.pkl files under the root directory. Both files have been provided and can be found under the data directory
state_dataset = Brain_Connectome_Task_Download(root, name,n_rois, threshold,path_to_data,n_jobs,s3)
print(state_dataset)
print(state_dataset[0])

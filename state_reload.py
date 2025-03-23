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
            s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
            
            for y, path in enumerate(all_paths):
                try:
                    ts_path = os.path.join(target_path+"/time_series_100", iid+"_"+os.path.basename(path).split(".")[0]+"_time_series.npy")
                    if not os.path.exists(ts_path):
                        print("iid:",iid)
                        s3.download_file(BUCKET_NAME, path,os.path.join(target_path, iid+"_"+os.path.basename(path)))
                        rnd = random.randint(0,1000)
                        reg_prefix = iid+str(rnd)
                        s3.download_file(BUCKET_NAME, reg_paths[y],os.path.join(target_path, reg_prefix+"_"+os.path.basename(reg_paths[y])))
                        image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(path))
                        reg_path = os.path.join(target_path, reg_prefix+"_"+os.path.basename(reg_paths[y]))
                        img = nib.load(image_path_LR)
                        regs = np.loadtxt(reg_path)
                        fmri = img.get_fdata()
                        Y = Brain_Connectome_Task_Download.extract_from_3d_no(volume,fmri)
                        start = 1
                        stop = Y.shape[0]
                        step = 1
                        # detrending
                        t = np.arange(start, stop+step, step)
                        tzd = zscore(np.vstack((t, t**2)), axis=1)
                        XX = np.vstack((np.ones(Y.shape[0]), tzd))
                        B = np.matmul(np.linalg.pinv(XX).T,Y)
                        Yt = Y - np.matmul(XX.T,B) 
                        # regress out head motion regressors
                        B2 = np.matmul(np.linalg.pinv(regs),Yt)
                        Ytm = Yt - np.matmul(regs,B2) 
                        # zscore over axis=0 (time)
                        zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)
                        #save the zscored data
                        ts_path = os.path.join(target_path+"/time_series_100", iid+"_"+os.path.basename(path).split(".")[0]+"_time_series.npy")
                        os.makedirs(os.path.dirname(ts_path), exist_ok=True)
                        np.save(ts_path,zd_Ytm)
                        # os.remove(image_path_LR)
                        # os.remove(reg_path)
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
        tasks = [(iid, BUCKET_NAME, volume) for iid in ids]
        with Pool(self.n_jobs) as pool:
            data_list = pool.map(worker_function, tasks)    
        print("length of data list:", len(data_list))       
        dataset = list(itertools.chain(*data_list))
        
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])  

root = "data/state_100/"
name = "HCPState"
threshold = 5
path_to_data = "data/raw/HCPState"  # store the raw downloaded scans
n_rois = 100
n_jobs = 2 # this script runs in parallel and requires the number of jobs is an input

ACCESS_KEY = ''  # your connectomeDB credentials
SECRET_KEY = ''
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
# this function requires both HCP_behavioral.csv and ids.pkl files under the root directory. Both files have been provided and can be found under the data directory
state_dataset = Brain_Connectome_Task_Download(root, name,n_rois, threshold,path_to_data,n_jobs,s3)

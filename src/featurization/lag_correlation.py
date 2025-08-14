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
import boto3
from pathos.multiprocessing import ProcessingPool as Pool
from nilearn.connectome import ConnectivityMeasure


def worker_function(args):
    iid, behavioral_df, BUCKET_NAME, volume,lag = args
    return Brain_Connectome_Rest_Download.get_data_obj_static(iid, behavioral_df, BUCKET_NAME, volume,lag)



class Brain_Connectome_Rest_Download(InMemoryDataset):
    
    def __init__(self, root,name,n_rois, threshold,path_to_data,n_jobs,s3,lag, transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.graph_threshold,self.target_path,self.n_jobs,self.s3,self.lag = root, name,n_rois,threshold,path_to_data,n_jobs,s3,lag
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']
    
#input: atlas, fmri. output: roi * time series
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
        for i in np.unique(volume): #volume is the atlas
            if i != 0: #create a mask for each roi
    #             print(i)
                bool_roi = np.zeros(volume.shape, dtype=int)
                bool_roi[volume == i] = 1
                bool_roi = bool_roi.astype(bool)
    #             print(bool_roi.shape)
                # extract time-series data for each roi
                roi_ts_mean = []
                for t in range(fmri.shape[-1]):#average fmri signal of each roi over time
                    roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
                subcor_ts.append(np.array(roi_ts_mean))
        Y = np.array(subcor_ts).T # Y=roi x time series
        return Y


    @staticmethod
    def construct_Adj_postive_perc(corr,graph_threshold):
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0],
                                  100 - graph_threshold)
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
        expanded_ts = Brain_Connectome_Rest_Download.expand_time_series(time_series, lag)
        expanded_corr_matrix = Brain_Connectome_Rest_Download.construct_expanded_correlation_matrix(expanded_ts)
        print("expanded_corr_matrix", expanded_corr_matrix.shape)

        return expanded_corr_matrix

    
    @staticmethod
    def get_data_obj_static(iid,behavioral_data,BUCKET_NAME,volume,lag):
        try:
            time_series_file_path = "data/raw/HCPGender/time_series_1000"
            # print("check!",iid)
            time_series_file = os.path.join(time_series_file_path, f"{iid}_time_series.npy")
            
            zd_Ytm = np.load(time_series_file)
            
            threshold = 30
            positive_threshold_value = np.percentile(zd_Ytm[zd_Ytm > 0], 100 - threshold)
            zd_Ytm[zd_Ytm < positive_threshold_value] = 0
            #zd_Ytm[zd_Ytm >= positive_threshold_value] = 1
            
            expanded_corr_matrix = Brain_Connectome_Rest_Download.construct_expanded_lagged_corr(zd_Ytm, lag)
        
            lag_corr = expanded_corr_matrix[0:1000, 1000:2000]
            #lag_corr = expanded_corr_matrix[0:400, 400:800]
            #lag_corr = expanded_corr_matrix[0:100, 100:200]
            np.fill_diagonal(lag_corr, 0)
            
            lag_corr_reverse = expanded_corr_matrix[1000:2000, 0:1000]
            #lag_corr_reverse = expanded_corr_matrix[400:800, 0:400]
            #lag_corr_reverse = expanded_corr_matrix[100:200, 0:100]
            np.fill_diagonal(lag_corr_reverse, 0)

            conn = ConnectivityMeasure(kind='correlation')
            zd_fc = conn.fit_transform([zd_Ytm])[0]
            np.fill_diagonal(zd_fc, 0)
            corr_original = torch.tensor(zd_fc).to(torch.float)
            A = Brain_Connectome_Rest_Download.construct_Adj_postive_perc(corr_original, graph_threshold=5)
            edge_index = A.nonzero().t().to(torch.long)
            
            # Stack the matrices along a new dimension
            concat_matrix = np.concatenate((zd_fc, lag_corr,lag_corr_reverse), axis=1)
            
            corr = torch.tensor(concat_matrix).to(torch.float)
            iid = int(iid)
            gender = behavioral_data.loc[iid,'Gender']
            g = 1 if gender=="M" else 0
            labels = torch.tensor([g,behavioral_data.loc[iid,'AgeClass'],behavioral_data.loc[iid,'ListSort_AgeAdj'],behavioral_data.loc[iid,'PMAT24_A_CR']])
            data = Data(x=corr, edge_index=edge_index, y=labels) 
        except:
            return None
        return data


#         ...
    def process(self):
        path_doc = "data/"
        behavioral_df = pd.read_csv(os.path.join(path_doc,'HCP_behavioral.csv')).set_index('Subject')[['Gender','Age','ListSort_AgeAdj','PMAT24_A_CR']]
        mapping = {'22-25':0, '26-30':1,'31-35':2,'36+':3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)

        dataset = []
        BUCKET_NAME = 'hcp-openaccess'
        
        with open(os.path.join(path_doc,"ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        print(len(ids))
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        lag = self.lag
        #data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj_static)(iid,behavioral_df,BUCKET_NAME,volume) for iid in tqdm(test_ids))
        tasks = [(iid, behavioral_df, BUCKET_NAME, volume,lag) for iid in ids]
        with Pool(self.n_jobs) as pool:
            data_list = pool.map(worker_function, tasks)

        dataset = [x for x in data_list if x is not None]
        # print(len(dataset))
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])


for i in range(3,9):
    root = "data/rs_1000/rs_1000_thre30_"+str(i)+"lag/"
    lag = i
    name = "HCPGender"
    threshold = 5
    path_to_data = "data/raw/HCPGender"  # store the raw downloaded scans
    n_rois = 1000
    n_jobs = 30 # this script runs in parallel and requires the number of jobs is an input

    ACCESS_KEY = ''  # your connectomeDB credentials
    SECRET_KEY = ''
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    # this function requires both HCP_behavioral.csv and ids.pkl files under the root directory. Both files have been provided and can be found under the data directory
    rest_dataset = Brain_Connectome_Rest_Download(root,name,n_rois, threshold,path_to_data,n_jobs,s3,lag)
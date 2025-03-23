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
from connectivity_matrices import KendallConnectivityMeasure
#from nilearn.connectome import ConnectivityMeasure


# This function is called by the parallel processing pool.
# It unpacks the arguments and calls the actual processing method.
def worker_function(args):
    # Unpack the arguments that were prepared for each task
    iid, behavioral_df, BUCKET_NAME, volume = args
    
    # Directly call the static processing method with the unpacked arguments
    return Brain_Connectome_Rest_Download.get_data_obj_static(iid, behavioral_df, BUCKET_NAME, volume)


class Brain_Connectome_Rest_Download(InMemoryDataset):
    
    def __init__(self, root,name,n_rois, threshold,path_to_data,n_jobs,s3, transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.graph_threshold,self.target_path,self.n_jobs,self.s3 = root, name,n_rois,threshold,path_to_data,n_jobs,s3
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
        Y = np.array(subcor_ts).T #Y=roi x time series
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
    def get_data_obj_static(iid,behavioral_data,BUCKET_NAME,volume):
        try:

            time_series_file_path = "data/raw/HCPGender_re/time_series_100"
            print("check!",iid)

            time_series_file = os.path.join(time_series_file_path, f"{iid}_time_series.npy")

            Ytm = np.load(time_series_file)
            #zd_Ytm = np.load(time_series_file)
            zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)
            
            #positive_threshold_value = 1
            threshold = 30
            positive_threshold_value = np.percentile(zd_Ytm[zd_Ytm > 0], 100 - threshold)
            zd_Ytm[zd_Ytm < positive_threshold_value] = 0
            
            conn = KendallConnectivityMeasure(kind='correlation')
            
            zd_fc = conn.fit_transform([zd_Ytm])[0]
            np.fill_diagonal(zd_fc, 0)
            corr = torch.tensor(zd_fc).to(torch.float)
            
            iid = int(iid)
            gender = behavioral_data.loc[iid,'Gender']
            g = 1 if gender=="M" else 0
            labels = torch.tensor([g,behavioral_data.loc[iid,'AgeClass'],behavioral_data.loc[iid,'ListSort_AgeAdj'],behavioral_data.loc[iid,'PMAT24_A_CR']])
            A = Brain_Connectome_Rest_Download.construct_Adj_postive_perc(corr, graph_threshold=5)
            edge_index = A.nonzero().t().to(torch.long)
            data = Data(x=corr, edge_index=edge_index, y=labels) 
        except:
            return None
        return data


#         ...
    def process(self):
        behavioral_df = pd.read_csv(os.path.join("data/",'HCP_behavioral.csv')).set_index('Subject')[['Gender','Age','ListSort_AgeAdj','PMAT24_A_CR']]
        mapping = {'22-25':0, '26-30':1,'31-35':2,'36+':3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)

        dataset = []
        BUCKET_NAME = 'hcp-openaccess'
        
        with open(os.path.join("data/","ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        
        #test_ids = ids[:10]
        
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        #data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj_static)(iid,behavioral_df,BUCKET_NAME,volume) for iid in tqdm(test_ids))
        tasks = [(iid, behavioral_df, BUCKET_NAME, volume) for iid in ids]
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


root = "data/rs_100/rs_100_thre30_kendall/"
name = "HCPGender"
threshold = 5
path_to_data = "data/raw/HCPGender"  # store the raw downloaded scans
n_rois = 100
n_jobs = 30 # this script runs in parellel and requires the number of jobs is an input

ACCESS_KEY = ''  # your connectomeDB credentials
SECRET_KEY = ''
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
# this function requires both HCP_behavioral.csv and ids.pkl files under the root directory. Both files have been provided and can be found under the data directory
rest_dataset = Brain_Connectome_Rest_Download(root,name,n_rois, threshold,path_to_data,n_jobs,s3)
print(rest_dataset)
print(rest_dataset[0])
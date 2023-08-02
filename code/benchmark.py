from training import train_lightgbm, train_xgboost, train, train_minibatch, train_bce_minibatch, train_gaussian_nb
from dstgcn_training import train_dstgcn
import time 
import numpy as np
import pickle
from prettytable import PrettyTable
import statistics
from datetime import datetime
from data import TrafficDataset
from torch.utils.data import DataLoader, Dataset
import torch
import os.path

import sqlite3
from sqlite3 import Error
import time

def save_progress(report_dicts, saved_name='metrics_saved', benchmark_launch_time=''):
    with open(saved_name+'_'+benchmark_launch_time+'.pkl', 'wb') as handle:
        pickle.dump(report_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_results(saved_name='metrics_saved', benchmark_launch_time=''):
    with open(saved_name+'_'+benchmark_launch_time+'.pkl', 'rb') as handle:
        return pickle.load(handle)

class PreLoadedDataset(Dataset):
    def __init__(self, years=['2013', '2014'], months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']):

        self.years = years
        self.months = months
        self.year_months = [(year, month) for year in years for month in months]
        filename_edges = 'loaded_data/edges.pkl'
        self.edges = pickle.load(open(filename_edges, 'rb'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __len__(self):
        return len(self.year_months)
    
    def __getitem__(self, idx):
        year, month = self.year_months[idx]

        filename_X = f'loaded_data/{year}_{month}_X.pkl'
        filename_y = f'loaded_data/{year}_{month}_y.pkl'
        X = pickle.load(open(filename_X, 'rb'))
        y = pickle.load(open(filename_y, 'rb'))
            
        return X.to(self.device), y.to(self.device), self.edges.to(self.device)

def build_dataloaders(train_years, valid_years, train_months, valid_months, class_model):
    train_dataset = class_model(years=train_years, months=train_months)
    valid_dataset = class_model(years=valid_years, months=valid_months)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    return train_dataloader, valid_dataloader

def benchmark(num_epochs=10, seeds=[0,1,2,3,4,5,6,7,8,9]):
    first_time = False
    train_years = [2013 , 2014]
    valid_years = [2013 , 2014]
    train_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    valid_months = ['12']

    for year in train_years + valid_years:
        for month in train_months + valid_months:
            if not os.path.isfile(f'loaded_data/{year}_{month}_X.pkl'):
                first_time = True
    class_model = TrafficDataset if first_time else PreLoadedDataset
    train_dataloader, valid_dataloader = build_dataloaders(
        train_years=train_years, valid_years=valid_years, 
        train_months=train_months, 
        valid_months=valid_months,
        class_model=class_model
    )
    report_dicts = {}
    all_start = time.time()
    benchmark_launch_time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    for seed in seeds:
        np.random.seed(seed)
        print('Training Graph wavenet:')
        start = time.time()
        gwnet_metrics = train(
            model_name='gwnet', num_epochs=num_epochs, save_model=True,
            train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
            model_id='gwnet'
        )
        if 'gwnet' not in report_dicts:
            report_dicts['gwnet'] = []
        report_dicts['gwnet'].append(gwnet_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'Graph Wavenet took: {time.time() - start}') 
    
        print('Training GaussianNB:')
        start = time.time()
        gnb_metrics = train_gaussian_nb(train_dataloader, valid_dataloader)
        if 'gnb' not in report_dicts:
            report_dicts['gnb'] = []
        report_dicts['gnb'].append(gnb_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'GNB took: {time.time() - start}')

        print('Training LightGBM:')
        start = time.time()
        light_gbm_metrics = train_lightgbm(train_dataloader, valid_dataloader)
        if 'lightgbm' not in report_dicts:
            report_dicts['lightgbm'] = []
        report_dicts['lightgbm'].append(light_gbm_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'LightGBM took: {time.time() - start}')
        
        print('Training XGBoost:')
        start = time.time()
        xgboost_metrics = train_xgboost(train_dataloader, valid_dataloader)
        if 'xgboost' not in report_dicts:
            report_dicts['xgboost'] = []
        report_dicts['xgboost'].append(xgboost_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'XGBoost took: {time.time() - start}')

        print('Training Scalable RGNN:')
        start = time.time()
        scalable_rgnn_metrics = train_minibatch(
            model_name='scalable_rgnn', num_epochs=num_epochs, save_model=True,
            train_dataloader = train_dataloader, valid_dataloader = valid_dataloader,
            model_id='scalable_rgnn'
        )
        if 'scalable_rgnn' not in report_dicts:
            report_dicts['scalable_rgnn'] = []
        report_dicts['scalable_rgnn'].append(scalable_rgnn_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'Scalable RGNN took: {time.time() - start}')

        print('Training Lite Scalable RGNN:')
        start = time.time()
        scalable_rgnn_metrics = train_minibatch(
            model_name='lite_scalable_rgnn', num_epochs=num_epochs, save_model=True,
            train_dataloader = train_dataloader, valid_dataloader = valid_dataloader,
            model_id='lite_scalable_rgnn'
        )
        if 'lite_scalable_rgnn' not in report_dicts:
            report_dicts['lite_scalable_rgnn'] = []
        report_dicts['lite_scalable_rgnn'].append(scalable_rgnn_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'Scalable RGNN took: {time.time() - start}')

        # Wrong dimension of dataloader
        print('Training DSTGCN:')
        start = time.time()
        dstgcn_metrics = train_dstgcn(train_dataloader, valid_dataloader, num_epochs=num_epochs)
        if 'dstgcn' not in report_dicts:
           report_dicts['dstgcn'] = []
        report_dicts['dstgcn'].append(dstgcn_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'DSTGCN took: {time.time() - start}')
    
    print()
    print(f'One benchmark pass took: {time.time() - all_start}')
    print()
    return benchmark_launch_time

def print_summary_results(results=None, benchmark_launch_time=''):
    if results is None:
        results = load_results(benchmark_launch_time=benchmark_launch_time)
    t = PrettyTable(['Model', 'Macro Average Recall', '+/-'])
    for model, list_of_runs in results.items():
        avg_res = 0
        metric = [report['macro avg']['recall'] for report in list_of_runs]
        if len(list_of_runs) > 1: std = statistics.stdev(metric) 
        else: std = 0
        t.add_row([model, statistics.mean(metric), std])
    print(t)

if __name__ == '__main__':
    benchmark_launch_time = benchmark(num_epochs=50)
    print_summary_results(benchmark_launch_time=benchmark_launch_time)
    
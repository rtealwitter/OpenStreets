from training import train_lightgbm, train_xgboost, train, train_minibatch, train_bce_minibatch, train_random_forest
from dstgcn_training import train_dstgcn
import time 
import numpy as np
import pickle
from prettytable import PrettyTable
import statistics
from datetime import datetime

def save_progress(report_dicts, saved_name='metrics_saved', benchmark_launch_time=''):
    with open(saved_name+'_'+benchmark_launch_time+'.pkl', 'wb') as handle:
        pickle.dump(report_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_results(saved_name='metrics_saved', benchmark_launch_time=''):
    with open(saved_name+'_'+benchmark_launch_time+'.pkl', 'rb') as handle:
        return pickle.load(handle)

def benchmark(seeds=[3,4,5,6,7,8]):
    report_dicts = {}
    num_epochs = 10
    all_start = time.time()
    benchmark_launch_time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    for seed in seeds:
        np.random.seed(seed)
        print('Training GaussianNB:')
        start = time.time()
        gnb_metrics = train_gaussian_nb()
        if 'gnb' not in report_dicts:
            report_dicts['gnb'] = []
        report_dicts['gnb'].append(gnb_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'GNB took: {time.time() - start}')

        print('Training LightGBM:')
        start = time.time()
        light_gbm_metrics = train_lightgbm()
        if 'lightgbm' not in report_dicts:
            report_dicts['lightgbm'] = []
        report_dicts['lightgbm'].append(light_gbm_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'LightGBM took: {time.time() - start}')
        
        print('Training XGBoost:')
        start = time.time()
        xgboost_metrics = train_xgboost()
        if 'xgboost' not in report_dicts:
            report_dicts['xgboost'] = []
        report_dicts['xgboost'].append(xgboost_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'XGBoost took: {time.time() - start}')

        print('Training Scalable RGNN:')
        start = time.time()
        scalable_rgnn_metrics = train_minibatch(model_name='scalable_rgnn', num_epochs=15, save_model=True)
        if 'scalable_rgnn' not in report_dicts:
            report_dicts['scalable_rgnn'] = []
        report_dicts['scalable_rgnn'].append(scalable_rgnn_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'Scalable RGNN took: {time.time() - start}')

        print('Training Lite Scalable RGNN:')
        start = time.time()
        scalable_rgnn_metrics = train_minibatch(model_name='lite_scalable_rgnn', num_epochs=num_epochs, save_model=True)
        if 'lite_scalable_rgnn' not in report_dicts:
            report_dicts['lite_scalable_rgnn'] = []
        report_dicts['lite_scalable_rgnn'].append(scalable_rgnn_metrics)
        save_progress(report_dicts, benchmark_launch_time=benchmark_launch_time)
        print(f'Scalable RGNN took: {time.time() - start}')

        print('Training DSTGCN:')
        start = time.time()
        dstgcn_metrics = train_dstgcn(num_epochs=num_epochs)
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
    benchmark_launch_time = benchmark()
    print_summary_results(benchmark_launch_time=benchmark_launch_time)
    
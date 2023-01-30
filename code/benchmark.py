from training import train_lightgbm, train_xgboost, train, train_minibatch
from dstgcn_training import train_dstgcn
import time 
import numpy as np

def benchmark(seeds=[1,2,3,4,5]):
    np.random.seed(seeds[0])
    print('Training LightGBM:')
    start = time.time()
    train_lightgbm()
    print(f'LightGBM took: {time.time() - start}')
    
    print('Training XGBoost:')
    start = time.time()
    train_xgboost()
    print(f'XGBoost took: {time.time() - start}')

    print('Training Scalable RGNN:')
    start = time.time()
    train(model_name='scalable_rgnn', num_epochs=10, save_model=True)
    print(f'Scalable RGNN took: {time.time() - start}')

    print('Training DSTGCN:')
    start = time.time()
    train_dstgcn()
    print(f'DSTGCN took: {time.time() - start}')

    print('Training RGNN:')
    start = time.time()
    train(model_name='rgnn', num_epochs=2, save_model=True)
    print(f'RGNN took: {time.time() - start}')

benchmark()
import torch.nn as nn
from data import *
from models import *
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import numpy as np
import xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from gwnet import gwnet

import sqlite3
from sqlite3 import Error
import json
import time

def create_connection():
    conn = None
    loss_conn = None
    try:
        conn = sqlite3.connect('metrics.db')
        loss_conn = sqlite3.connect('losses.db')
        print("Successfully Connected to SQLite")
    except Error as e:
        print(e)
    return conn, loss_conn

def close_connection(conn, loss_conn):
    if (conn):
        conn.close()
    if (loss_conn):
        loss_conn.close()
    
    print("SQLite connection is closed")

def classification_report_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classification_report (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                class TEXT NOT NULL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                support INTEGER,
                model_id TEXT NOT NULL
            )
        """)
        print("Table checked, it exists or has been successfully created.")
    except Error as e:
        print(e)

def loss_table(loss_conn):
    try:
        cursor = loss_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loss_table (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                losses TEXT,
                loss_type TEXT,
                model_id TEXT NOT NULL
            )
        """)
        print("Table checked, it exists or has been successfully created.")
    except Error as e:
        print(e)

def insert_losses(loss_conn, losses, loss_type, run_id, model_id):
    try:
        cursor = loss_conn.cursor()
        cursor.execute("""
            INSERT INTO loss_table (run_id, losses, loss_type, model_id)
            VALUES (?, ?, ?, ?)
        """, (run_id, json.dumps(losses), loss_type, model_id))
    except Error as e:
        print(e)

def insert_report(conn, report, run_id, model_id):
    cursor = conn.cursor()
    
    for class_name, metrics in report.items():
        if class_name in ['accuracy']:
            continue  # Skip as it has a different structure
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = metrics['f1-score']
        support = metrics['support']

        cursor.execute("""
            INSERT INTO classification_report (run_id, class, precision, recall, f1_score, support, model_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, class_name, precision, recall, f1_score, support, model_id))

    conn.commit()

def classification_report_table_to_df(conn):
    try:
        conn = sqlite3.connect('metrics.db')
    except Error as e:
        print(e)

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM classification_report")
    # convert to pandas dataframe
    df = pd.DataFrame(cursor.fetchall(), columns=['id', 'run_id', 'class', 'precision', 'recall', 'f1_score', 'support', 'model_id'])
    return df
    
# Data loaders
def build_dataloaders(train_years, valid_years, train_months, valid_months):
    train_dataset = TrafficDataset(years=train_years, months=train_months)
    valid_dataset = TrafficDataset(years=valid_years, months=valid_months)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    return train_dataloader, valid_dataloader

def initialize_tracking():
    conn, loss_conn = create_connection()
    classification_report_table(conn)
    loss_table(loss_conn)
    return conn, loss_conn

def initialize_training(model_name='recurrent', num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'rgnn':
        # I hid some hyper parameters in the model's initialization step.
        model = RecurrentGCN(node_features = 127).to(device) # Recurrent GCN so we pass temporal information
    elif model_name == 'gnn':
        model = ConvGraphNet(input_dim = 127, output_dim = 1).to(device)
    elif model_name == 'dgcn':
        model = DeeperGCN(num_features = 127, hidden_channels = 64, out_channels=2, num_layers = 3).to(device)
    elif model_name == 'scalable_rgnn':
        # More hidden hyper parameters in the model's initialization step
        # of best setting.
        model = ScalableRecurrentGCN(node_features = 127, hidden_dim_sequence=[1024,512,768,256,128,64,64]).to(device)
    elif model_name == 'lite_scalable_rgnn':
        # More hidden hyper parameters in the model's initialization step
        # of best setting.
        model = ScalableRecurrentGCN(node_features = 127, hidden_dim_sequence=[512,256,128,64,64]).to(device)
        #[256,128,64,32,32] .72
        #[384,192,96,48,48] .74
        #[512,256,128,64,64] .76
    
    elif model_name == 'gwnet':
        model = gwnet(device, num_nodes=19391, in_dim=127, out_dim=2).to(device)
    
    num_updates = 12*num_epochs
    warmup_steps = 2
    
    num_param = sum([p.numel() for p in model.parameters()])
    print(f'There are {num_param} parameters in the model.')

    def warmup(current_step):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps)
        else:                                 
            return max(0.0, float(num_updates - current_step) / float(max(1, num_updates - warmup_steps)))
    
    if model_name == 'scalable_rgnn' or model_name == 'lite_scalable_rgnn':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    elif model_name == 'gnn' or model_name == 'gwnet':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, threshold=1e-3, min_lr=1e-6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    
    # we reweight by the expected number of collisions / non-collisions 
    return model, optimizer, scheduler

def verbose_output(out, y):
    if len(out.shape) == 3:
        pred_labels = out.argmax(axis=2).flatten().detach().numpy()
    elif len(out.shape) == 2:
        pred_labels = out.argmax(axis=1).flatten().detach().numpy()
    true_labels = y.flatten().detach().numpy()
    print(f'The model predicted {pred_labels.sum()} collisions.')
    print(f'There were really {y.sum()} collisions.')
    print(classification_report(true_labels, pred_labels))
    return classification_report(true_labels, pred_labels, output_dict=True)

def train(
    model_name, num_epochs, train_dataloader, valid_dataloader, 
    save_model=True, return_class_report_dict=True, model_id='default'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conn, loss_conn = initialize_tracking()
    model, optimizer, scheduler = initialize_training(model_name=model_name, num_epochs=num_epochs)
    #train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013, 2014], valid_years=[2013, 2014], train_months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'], valid_months=['12'])
    train_losses = []
    valid_losses = []
    best_recall = -1
    report_dict = None
    for epoch in range(num_epochs):
        model.train() # turn on dropout
        for i, (X, y, edges) in enumerate(train_dataloader):
            ratio = y.numel() / y.sum()
            print(ratio)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, ratio+.2]))
            criterion.to(device)
            X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()
            X.to(device)
            y.to(device)
            edges.to(device)
            optimizer.zero_grad()
            if model_name == 'gwnet':
                # Graph wavenets are a bit special
                X = X.unsqueeze(0) # one batch
                X = X.transpose(1,3) # convention
                modified_X = X[:,:,:,:-1] # time series
                modified_y = y[:-1,:] # time series next step
                modified_y = modified_y.flatten()
                out = model(modified_X)
                out = out.squeeze(0).permute(2,1,0).reshape(-1,2)
                loss = criterion(out, modified_y)
                verbose_output(out.cpu(), modified_y.cpu())
            else:
                out = model(X, edges)
                loss = criterion(out.permute(0,2,1), y)
                verbose_output(out.cpu(), y.cpu())
            loss.backward()
            optimizer.step()
            train_losses += [loss.item()]
            print(f'Epoch: {epoch} \t Iteration: {i} \t Train Loss: {train_losses[-1]}')
            
            if model_name == 'gnn':
                scheduler.step(loss)
            else:   
                scheduler.step()
        model.eval() # turn off dropout
        for X, y, edges in valid_dataloader:
            X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()
            X.to(device)
            y.to(device)
            edges.to(device)        
            with torch.no_grad():
                if model_name == 'gwnet':
                    # graph wavenets are a bit special
                    X = X.unsqueeze(0) # one batch
                    X = X.transpose(1,3) # convention
                    modified_X = X[:,:,:,:-1] # time series
                    modified_y = y[:-1,:] # time series next step
                    modified_y = modified_y.flatten()
                    out = model(modified_X)
                    out = out.squeeze(0).permute(2,1,0).reshape(-1,2)
                    loss = criterion(out, modified_y)                    
                    metrics = verbose_output(out.cpu(), modified_y.cpu())
                else:
                    out = model(X, edges)
                    loss = criterion(out.permute(0,2,1), y)
                    metrics = verbose_output(out.cpu(), y.cpu())
                valid_losses += [loss.item()]            
            print(f'Epoch: {epoch} \t Valid Loss: {valid_losses[-1]}')
            
        if metrics['macro avg']['recall'] > best_recall:
            best_recall = metrics['macro avg']['recall']
            report_dict = metrics
            torch.save(model.state_dict(), f'saved_models/best_{model_name}.pt')
    if save_model:
        torch.save(model.state_dict(), f'saved_models/{model_name}.pt')

    if return_class_report_dict:
        print('LOSSES')
        print(valid_losses)
        print(train_losses)
        print()
        run_id = str(time.time())
        insert_report(conn, report_dict, run_id, model_id)
        insert_losses(loss_conn, train_losses, "train", run_id, model_id)
        insert_losses(loss_conn, valid_losses, "valid", run_id, model_id)
        close_connection(conn, loss_conn)
        return report_dict

def train_minibatch(
    model_name, num_epochs, train_dataloader, valid_dataloader, save_model=True, minibatch_size=4, return_class_report_dict=True, model_id='default'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conn, loss_conn = initialize_tracking()
    model, optimizer, scheduler = initialize_training(model_name=model_name, num_epochs=num_epochs)
    #train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013, 2014],
    #                                                       valid_years=[2013, 2014], 
    #                                                       train_months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'], 
    #                                                       valid_months=['12'])
    train_losses = []
    valid_losses = []
    best_recall = -1
    report_dict = None
    for epoch in range(num_epochs):
        model.train() # turn on dropout
        for i, (X, y, edges) in enumerate(train_dataloader):
            ratio = y.numel() / y.sum()
            print(ratio)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, ratio+.2]))
            criterion.to(device)
            X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()
            X.to(device)
            y.to(device)
            edges.to(device)

            minibatch_losses = []
            all_out = []
            for minibatch_x, minibatch_y in zip(torch.split(X, minibatch_size), torch.split(y, minibatch_size)):
                optimizer.zero_grad()
                out = model(minibatch_x, edges)
                all_out.append(out)
                loss = criterion(out.permute(0,2,1), minibatch_y)
                loss.backward()
                optimizer.step()
                minibatch_losses += [loss.item()]
            train_losses += [sum(minibatch_losses)]
            print(f'Epoch: {epoch} \t Iteration: {i} \t Train Loss: {train_losses[-1]}')
            verbose_output(torch.cat(all_out, dim=0).cpu(), y.cpu())
            if model_name == 'gnn':
                scheduler.step(loss)
            else:   
                scheduler.step()
        model.eval() # turn off dropout
        for X, y, edges in valid_dataloader:
            X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()        
            with torch.no_grad():
                out = model(X, edges)
                loss = criterion(out.permute(0,2,1), y)
                valid_losses += [loss.item()]            
            print(f'Epoch: {epoch} \t Valid Loss: {valid_losses[-1]}')
            metrics = verbose_output(out.cpu(), y.cpu())
        if metrics['macro avg']['recall'] > best_recall:
            best_recall = metrics['macro avg']['recall']
            report_dict = metrics
            torch.save(model.state_dict(), f'saved_models/best_{model_name}.pt')
    if save_model:
        torch.save(model.state_dict(), f'saved_models/{model_name}.pt')
    
    if return_class_report_dict:
        print('LOSSES')
        print(valid_losses)
        print(train_losses)
        print()
        run_id = str(time.time())
        insert_report(conn, report_dict, run_id, model_id)
        insert_losses(loss_conn, train_losses, "train", run_id, model_id)
        insert_losses(loss_conn, valid_losses, "valid", run_id, model_id)
        close_connection(conn, loss_conn)
        return report_dict

def train_bce_minibatch(
    model_name, num_epochs, train_dataloader, valid_dataloader,
    save_model=True, minibatch_size=4, return_class_report_dict=True, model_id="default"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conn, loss_conn = initialize_tracking()
    model, optimizer, scheduler = initialize_training(model_name=model_name, num_epochs=num_epochs)
    #train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013, 2014],
    #                                                       valid_years=[2013, 2014], 
    #                                                       train_months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'], 
    #                                                       valid_months=['12'])
    train_losses = []
    valid_losses = []
    best_recall = -1
    report_dict = None
    for epoch in range(num_epochs):
        model.train() # turn on dropout
        for i, (X, y, edges) in enumerate(train_dataloader):
            ratio = y.numel() / y.sum()
            print(ratio)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([ratio+.2]))
            criterion.to(device)
            X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()
            X.to(device)
            y.to(device)
            edges.to(device)

            minibatch_losses = []
            all_out = []
            for minibatch_x, minibatch_y in zip(torch.split(X, minibatch_size), torch.split(y, minibatch_size)):
                optimizer.zero_grad()
                out = model(minibatch_x, edges)
                all_out.append(out)
                loss = criterion(out.permute(0,2,1).squeeze(1), minibatch_y.float())
                loss.backward()
                optimizer.step()
                minibatch_losses += [loss.item()]
            train_losses += [sum(minibatch_losses)]
            print(f'Epoch: {epoch} \t Iteration: {i} \t Train Loss: {train_losses[-1]}')
            verbose_output(torch.cat(all_out, dim=0).cpu(), y.cpu())
            if model_name == 'gnn':
                scheduler.step(loss)
            else:   
                scheduler.step()
        model.eval() # turn off dropout
        for X, y, edges in valid_dataloader:
            X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()        
            with torch.no_grad():
                out = model(X, edges)
                loss = criterion(out.permute(0,2,1).squeeze(1), y.float())
                valid_losses += [loss.item()]            
            print(f'Epoch: {epoch} \t Valid Loss: {valid_losses[-1]}')
            metrics = verbose_output(out.cpu(), y.cpu())
        if metrics['macro avg']['recall'] > best_recall:
            best_recall = metrics['macro avg']['recall']
            report_dict = metrics
            torch.save(model.state_dict(), f'saved_models/best_{model_name}.pt')
    if save_model:
        torch.save(model.state_dict(), f'saved_models/{model_name}.pt')
    
    if return_class_report_dict:
        run_id = str(time.time())
        insert_report(conn, report_dict, run_id, model_id)
        insert_losses(loss_conn, train_losses, "train", run_id, model_id)
        insert_losses(loss_conn, valid_losses, "valid", run_id, model_id)
        close_connection(conn, loss_conn)
        return report_dict


def train_adaboost(train_dataloader, valid_dataloader, num_epochs=10, num_learners=30, verbose=True):
    #train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013], valid_years=[2014])
    X, y, edges = next(iter(train_dataloader))
    X, y, edges = X.squeeze(), y.squeeze(0).float(), edges.squeeze()

    weights = torch.ones_like(y)
    weights = weights / weights.sum()
    num_zeros = (y == 0).sum()
    num_ones = (y == 1).sum()
    class_weights = (torch.ones_like(y) + num_zeros/num_ones * y) / 2
    learners, alphas = [], []
    def warmup(current_step, warmup_steps=2):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps)
        else:                                 
            return max(0.0, float(num_epochs - current_step) / float(max(1, num_epochs - warmup_steps)))

    for _ in range(num_learners):
        weighted_error = 1
        success = True
        while weighted_error > .5:
            model = ConvGraphNet(input_dim=127, output_dim=1, hidden_dim=8, hidden_count=1)
            if len(learners) > 0 and success: model.load_state_dict(learners[-1].state_dict())
            if verbose: print(sum([p.numel() for p in model.parameters()]))
            criterion = nn.BCELoss(reduction='none')
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
            for i in range(num_epochs):
                output = torch.sigmoid(model(X, edges))
                loss_vector = criterion(output, y)
                loss = torch.sum(loss_vector * weights * class_weights)         
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                scheduler.step()
                prediction = torch.round(output).detach()
                weighted_error = ((prediction != y) * weights * class_weights).sum().item()
                num_vals = y.shape[0] * y.shape[1]
                accuracy = (prediction == y).sum() / num_vals 
                if verbose:
                    print('Loss:', np.round(loss.item(), 2))
                    print('Weighted error:', np.round(weighted_error, 2))
                    print('Accuracy:', np.round(accuracy.item(),2))
                    print('Number of 1:', np.round(prediction.sum().item()))
                    print('Number of 0:', np.round((1 - prediction).sum().item()))
            success = False
        if weighted_error < .5:
            alpha = .5 * np.log((1 - weighted_error) / weighted_error)
            learners += [model]
            alphas += [alpha]
            new_multiple = np.exp(- alpha * (2*y - 1) * (2*prediction - 1))
            weights = weights * new_multiple
            weights = weights / weights.sum()
    return learners, alphas

def test_adaboost(learners, alphas, valid_dataloader):
    X, y, edges = next(iter(valid_dataloader))
    combined_pred = torch.zeros_like(y)
    for learner, alpha in zip(learners, alphas):
        print(alpha)
        learner.eval()
        out = torch.sigmoid(learner(X.squeeze(), edges.squeeze()))
        prediction = torch.round(out)
        combined_pred = combined_pred + prediction * alpha / sum(alphas)

    combined_pred = torch.round(combined_pred)

    labels = y.flatten().detach().numpy()
    pred_labels = combined_pred.flatten().detach().numpy()
    print(pred_labels.sum().astype(int))
    print(classification_report(labels, pred_labels))

x_train_expand, y_train_expand, x_valid_expand, y_valid_expand = None, None, None, None
def process_for_feature_only_models(train_dataloader, valid_dataloader, set_global=True):
    global x_train_expand 
    global y_train_expand
    global x_valid_expand
    global y_valid_expand
    if all(x is None for x in [x_train_expand, y_train_expand, x_valid_expand, y_valid_expand]):
        #train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013, 2014],
        #                                                    valid_years=[2013, 2014],
        #                                                    train_months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'], 
        #                                                    valid_months=['12'])

        X_train = []
        Y_train = []
        for i, (X, y, _) in enumerate(train_dataloader):
            X_train.append(X.squeeze()), Y_train.append(y.squeeze())

        X_valid = []
        Y_valid = []
        for i, (X, y, _) in enumerate(valid_dataloader):
            X_valid.append(X.squeeze()), Y_valid.append(y.squeeze())
        
        x_train = torch.cat(X_train).flatten(0,1).cpu().numpy()
        y_train = torch.cat(Y_train).flatten(0,1).cpu().numpy()
        x_valid= torch.cat(X_valid).flatten(0,1).cpu().numpy()
        y_valid = torch.cat(Y_valid).flatten(0,1).cpu().numpy()

        if set_global:
            x_train_expand = x_train
            y_train_expand = y_train
            x_valid_expand = x_valid
            y_valid_expand = y_valid 
    
    return x_train_expand, y_train_expand, x_valid_expand, y_valid_expand 

def train_xgboost(train_dataloader, valid_dataloader, return_class_report_dict=True, model_id="xgboost"):
    x_train_expand, y_train_expand, x_valid_expand, y_valid_expand = process_for_feature_only_models(
        train_dataloader, valid_dataloader
    )
    conn, loss_conn = initialize_tracking()

    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train_expand
    )

    xgboost.set_config(verbosity=2)

    # create model instance
    bst = XGBClassifier(n_estimators=20, max_depth=6, learning_rate=0.3, objective='binary:logistic')
    # fit model
    bst.fit(x_train_expand, y_train_expand, verbose=1, sample_weight=classes_weights)
    # make predictions
    preds = bst.predict(x_valid_expand)

    print(f'XGBoost predicted {preds.sum()} collisions.')
    print(f'There were really {y_valid_expand.sum()} collisions.')
    print(classification_report(y_valid_expand, preds))
    bst.save_model('saved_models/xgb.json')
    if return_class_report_dict:
        run_id = str(time.time())
        report_dict = classification_report(y_valid_expand, preds, output_dict=True)
        insert_report(conn, report_dict, run_id, model_id)
        close_connection(conn, loss_conn)
        return report_dict
    return bst

def train_lightgbm(train_dataloader, valid_dataloader, return_class_report_dict=True, model_id="lightgbm"):
    x_train_expand, y_train_expand, x_valid_expand, y_valid_expand = process_for_feature_only_models(
        train_dataloader, valid_dataloader
    )
    conn, loss_conn = initialize_tracking()
    
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train_expand
    )

    lgbm_model = LGBMClassifier(boosting_type='gbdt', 
                num_leaves=2^10+1, 
                max_depth=-1, 
                learning_rate=0.01, 
                n_estimators=1000, 
                subsample_for_bin=200000, 
                objective=None, 
                class_weight=None, 
                min_split_gain=0.0, 
                min_child_weight=0.001, 
                min_child_samples=20, 
                subsample=1.0, 
                subsample_freq=0, 
                colsample_bytree=1.0, 
                reg_alpha=0.0, 
                reg_lambda=0.0, 
                random_state=None, 
                n_jobs=None, 
                verbose=1000,
                importance_type='split')

    lgbm_model.fit(x_train_expand, y_train_expand, sample_weight=classes_weights, verbose=1000)
    preds = lgbm_model.predict(x_valid_expand)

    print(f'LightGBM predicted {preds.sum()} collisions.')
    print(f'There were really {y_valid_expand.sum()} collisions.')
    print(classification_report(y_valid_expand, preds))
    lgbm_model.booster_.save_model('saved_models/lightgbm.txt')

    if return_class_report_dict:
        run_id = str(time.time())
        report_dict = classification_report(y_valid_expand, preds, output_dict=True)
        insert_report(conn, report_dict, run_id, model_id)
        close_connection(conn, loss_conn)
        return report_dict
    return lgbm_model

def train_gaussian_nb(train_dataloader, valid_dataloader, return_class_report_dict=True, model_id="gaussian_nb"):
    x_train_expand, y_train_expand, x_valid_expand, y_valid_expand = process_for_feature_only_models(
        train_dataloader, valid_dataloader
    )
    conn, loss_conn = initialize_tracking()
    
    # clf = RandomForestClassifier(n_estimators=1000,
    #                              verbose=3,
    #                              n_jobs=-1,
    #                              class_weight='balanced_subsample')
    clf = GaussianNB()
                                 
    clf.fit(x_train_expand, y_train_expand)
    preds = clf.predict(x_valid_expand)
    print(f'GNB predicted {preds.sum()} collisions.')
    print(f'There were really {y_valid_expand.sum()} collisions.')
    print(classification_report(y_valid_expand, preds))

    if return_class_report_dict:
        run_id = str(time.time())
        report_dict = classification_report(y_valid_expand, preds, output_dict=True)
        insert_report(conn, report_dict, run_id, model_id)
        close_connection(conn, loss_conn)
        return report_dict
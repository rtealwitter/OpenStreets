import torch.nn as nn
from data import *
from models import *
from sklearn.metrics import classification_report
import numpy as np


# Data loaders
def build_dataloaders(train_years, valid_years, train_months, valid_months):
    train_dataset = TrafficDataset(years=train_years, months=train_months)
    valid_dataset = TrafficDataset(years=valid_years, months=valid_months)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    return train_dataloader, valid_dataloader

def initialize_training(model_name='recurrent', num_epochs=2):
    # LUCAS
    if model_name == 'rgnn':
        # I hid some hyper parameters in the model's initialization step.
        model = RecurrentGCN(node_features = 127) # Recurrent GCN so we pass temporal information
    elif model_name == 'gnn':
        model = ConvGraphNet(input_dim = 127, output_dim = 2)
    elif model_name == 'dgcn':
        model = DeeperGCN(num_features = 127, hidden_channels = 64, out_channels=2, num_layers = 3)
    
    num_updates = 12*num_epochs
    warmup_steps = 2
    
    num_param = sum([p.numel() for p in model.parameters()])
    print(f'There are {num_param} parameters in the model.')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)
    def warmup(current_step):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps)
        else:                                 
            return max(0.0, float(num_updates - current_step) / float(max(1, num_updates - warmup_steps)))

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

def train(model_name, num_epochs, save_model=True):
    model, optimizer, scheduler = initialize_training(model_name=model_name, num_epochs=num_epochs)
    train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013, 2014], valid_years=[2013, 2014], train_months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'], valid_months=['12'])
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        model.train() # turn on dropout
        for i, (X, y, edges) in enumerate(train_dataloader):
            ratio = y.numel() / y.sum()
            print(ratio)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, ratio+.2]))
            X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()
            optimizer.zero_grad()
            out = model(X, edges)
            loss = criterion(out.permute(0,2,1), y)
            loss.backward()
            optimizer.step()
            train_losses += [loss.item()]
            print(f'Epoch: {epoch} \t Iteration: {i} \t Train Loss: {train_losses[-1]}')
            verbose_output(out, y)
            scheduler.step()
        model.eval() # turn off dropout
        for X, y, edges in valid_dataloader:
            X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()        
            with torch.no_grad():
                out = model(X, edges)
                loss = criterion(out.permute(0,2,1), y)
                valid_losses += [loss.item()]            
            print(f'Epoch: {epoch} \t Valid Loss: {valid_losses[-1]}')
            verbose_output(out, y)
    if save_model:
        torch.save(model.state_dict(), f'saved_models/{model_name}.pt')

def train_adaboost(num_epochs=5, num_learners=30, verbose=True):
    train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013], valid_years=[2014])
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

train(model_name='rgnn', num_epochs=2, save_model=True)
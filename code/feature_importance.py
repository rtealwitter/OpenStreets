# Libraries
import torch
import models
import pickle
from captum.attr import IntegratedGradients

# Files
import data
import training
import benchmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.ScalableRecurrentGCN(node_features = 127, hidden_dim_sequence=[1024,512,768,256,128,64,64]).to(device)

model.load_state_dict(torch.load('saved_models/best_scalable_rgnn.pt', map_location=device))

train_dataloader, valid_dataloader = benchmark.build_dataloaders(
    train_years=[2014], valid_years=[2014], train_months=[12], valid_months=[12], class_model=benchmark.PreLoadedDataset
)

# Get the first batch
X, y, edges = next(iter(valid_dataloader))
# Set the data to track the gradients
X.requires_grad = True
# Get the predictions
def modified_model(X):
    return model(X, edges)

# Use captum to compute the attributions
ig = IntegratedGradients(modified_model)
attr = ig.attribute(X)
attr = attr.detach().numpy()
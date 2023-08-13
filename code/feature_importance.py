# Libraries
import torch
import models
import pickle
from captum.attr import FeaturePermutation, FeatureAblation

# Files
import data
import training
import benchmark

# Load pretrained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.ScalableRecurrentGCN(node_features = 127, hidden_dim_sequence=[1024,512,768,256,128,64,64]).to(device)

model.load_state_dict(torch.load('saved_models/best_scalable_rgnn.pt', map_location=device))

# Load the data
train_dataloader, valid_dataloader = benchmark.build_dataloaders(
    train_years=[2014], valid_years=[2014], train_months=['11'], valid_months=['12'], class_model=benchmark.PreLoadedDataset
)

# Get the first batch
X, y, edges = next(iter(valid_dataloader))
X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()
# Take first day of X
X = X[:2]
y = y[:2]
print(X.shape)
print(y.shape)

X.to(device)
edges.to(device)


ratio = y.numel() / y.sum()
criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, ratio+.2]))
losses = []

# Use captum to compute the attributions
# Make copy of X and permute features at index i
indices = list(range(X.shape[2]))
for index in indices:
    X_i = X.clone()
    permutation = torch.randperm(X_i.shape[1])
    X_i[:,:,index] = X[:,permutation,index]
    preds = model(X_i, edges)

    # Compute the loss
    training.verbose_output(preds.cpu(), y.cpu())
#    loss = criterion(preds.permute(0,2,1), y)
#    print(f'Loss for {index}: {loss.item()}')
#    losses += [loss.item()]

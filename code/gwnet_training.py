import torch
from torch.utils.data import DataLoader, Dataset
from data import TrafficDataset
from gwnet import gwnet
from sklearn.metrics import classification_report
import pickle

# Define loaded_data folder and run below to cache data
first_run = False
if first_run:
    # Run once to set up loaded_data for 2013
    train_dataset = TrafficDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for X,y,edges in train_dataloader:
        print(X.shape)
        print(y.shape)
        print(edges.shape)

class PreLoadedDataset(Dataset):
    def __init__(self, years=['2013'], months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']):
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

def build_dataloaders(train_years, valid_years, train_months, valid_months):
    train_dataset = PreLoadedDataset(years=train_years, months=train_months)
    valid_dataset = PreLoadedDataset(years=valid_years, months=valid_months)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    return train_dataloader, valid_dataloader

# TODO: Rewrite verbose_output for time series graph wavenet input
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

train_dataloader, valid_dataloader = build_dataloaders(
    train_years=[2013], valid_years=[2013], train_months=['01', '02'], valid_months=['12']
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = gwnet(device, num_nodes=19391, in_dim=127, out_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.0001) # hyperparameters from default

num_epochs = 1
for epoch in range(num_epochs):
    for i, (X, y, edges) in enumerate(train_dataloader):
        ratio = y.numel() / y.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, ratio+.2]))
        criterion.to(device)
        X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()
        X = X.unsqueeze(0) # one batch
        X = X.transpose(1,3) # convention
        modified_X = X[:,:,:,:-1] # time series
        modified_y = y[1:,:] # time series next step
        modified_y = modified_y.flatten()
        out = model(modified_X)
        out = out.squeeze(0).permute(2,1,0).reshape(-1,2)
        print('out', out.shape)
        print('y', modified_y.shape)
        loss = criterion(out, modified_y)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        print(f'Epoch: {epoch} \t Iteration: {i} \t Train Loss: {loss.item()}')
        #verbose_output(out.cpu(), y.cpu())


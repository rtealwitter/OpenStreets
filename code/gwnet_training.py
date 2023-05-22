import torch
from data import DataLoader, TrafficDataset
from gwnet import gwnet
from sklearn.metrics import classification_report

def build_dataloaders(train_years, valid_years, train_months, valid_months):
    train_dataset = TrafficDataset(years=train_years, months=train_months)
    valid_dataset = TrafficDataset(years=valid_years, months=valid_months)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    return train_dataloader, valid_dataloader

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

train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013], valid_years=[2013], train_months=['01'], valid_months=['12'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = gwnet(device, num_nodes=19391, in_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.0001) # hyperparameters from default

num_epochs = 1
for epoch in range(num_epochs):
    for i, (X, y, edges) in enumerate(train_dataloader):
        ratio = y.numel() / y.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, ratio+.2]))
        criterion.to(device)
        X, y, edges = X.squeeze(), y.squeeze(), edges.squeeze()
        X.to(device), y.to(device), edges.to(device)
        print(f'X shape: {X.shape}')
        X = torch.nn.functional.pad(X,(1,0,0,0))
        print(f'X shape: {X.shape}')
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        print(f'Epoch: {epoch} \t Iteration: {i} \t Train Loss: {loss.item()}')
        verbose_output(out.cpu(), y.cpu())


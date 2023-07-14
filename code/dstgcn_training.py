from data import TrafficDataset, connect_taxi_to_nodes, get_flows, get_taxi_data
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from dstgcn import DSTGCN
import torch
from data import *
from models import *

def dstgcn_get_X_day(data_constant, weather, flows_day, day):
    # Make a deep copy of the constant link data
    spatial_features = data_constant.copy(deep=True)
    # Add weather data
    weather_ = weather.loc[weather['DATE'] == day].drop(columns=['STATION', 'NAME', 'DATE', 'date', 'year'])
    # Weather is the same for every link (only one weather station)
    external_features = pd.DataFrame(index=spatial_features.index)
    for column_name in weather_: 
        external_features[column_name] = weather_[column_name].values[0]
    # Get flow data on day
    temporal_features = pd.DataFrame.from_dict(flows_day)
    # Make sure the index is the same as the link data
    temporal_features['OBJECTID'] = temporal_features.index
    # Make both indices the same
    temporal_features.set_index('OBJECTID', inplace=True)
    spatial_features.set_index('OBJECTID', inplace=True)
    # Make sure the index is sorted so it connects to labels
    spatial_features.sort_index(inplace=True)
    temporal_features.sort_index(inplace=True)
    return spatial_features, temporal_features, external_features

def dstgcn_get_X(data_constant, weather, flows):
    spatial_features_X = []
    temporal_features_X = []
    external_features_X = []
    for day in flows.keys():
        spatial_features, temporal_features, external_features = dstgcn_get_X_day(data_constant, weather, flows[day], day)
        spatial_features_X += [spatial_features.values]
        temporal_features_X += [temporal_features.values]
        external_features_X += [external_features.values]
    return torch.tensor(np.array(spatial_features_X)).float(), \
           torch.tensor(np.array(temporal_features_X)).float(), \
           torch.tensor(np.array(external_features_X)).float()

class DSTGCNTrafficDataset(TrafficDataset):
    def __init__(self, years=['2013'], months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']):
        super(DSTGCNTrafficDataset, self).__init__(years=years, months=months)
    
    def __len__(self):
        return len(self.year_months)
    
    def __getitem__(self, idx):
        year, month = self.year_months[idx]

        filename_flows = f'flows/flow_{year}_{month}.pickle'
        if os.path.isfile(filename_flows):
            flows = pickle.load(open(filename_flows, 'rb'))
        else:
            # If you're getting throttled, reset router IP address and computer IP address
            taxi = get_taxi_data(year, month)
            # Limit number of trips per month
            taxi = connect_taxi_to_nodes(taxi, 'start', self.nodes)
            taxi = connect_taxi_to_nodes(taxi, 'end', self.nodes)
            # Takes 8 minutes to run on 1 million trips
            print('Calculating flows...', end=' ')
            flows = get_flows(taxi, self.graph, self.links)
            print('complete!')
            pickle.dump(flows, open(filename_flows, 'wb'))

        filename_X = f'dstgcn_loaded_data/{year}_{month}_X.pkl'
        filename_y = f'dstgcn_loaded_data/{year}_{month}_y.pkl'
        # NOTE: Make sure you have a ``loaded_data/'' directory
        if os.path.isfile(filename_X):
            X = pickle.load(open(filename_X, 'rb'))
            y = pickle.load(open(filename_y, 'rb'))
        else:
            X = dstgcn_get_X(self.data_constant, self.weather, flows)
            y = get_y(self.collisions, self.links, flows)

            pickle.dump(X, open(filename_X, 'wb'))
            pickle.dump(y, open(filename_y, 'wb'))
        
        spatial_features, temporal_features, external_features = X
        print(spatial_features.shape, temporal_features.shape, external_features.shape)
        return spatial_features.to(self.device), \
               temporal_features.to(self.device), \
               external_features.to(self.device), \
               y.to(self.device), \
               self.edges

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

def build_dataloaders(train_years, valid_years, train_months, valid_months):
    train_dataset = DSTGCNTrafficDataset(years=train_years, months=train_months)
    valid_dataset = DSTGCNTrafficDataset(years=valid_years, months=valid_months)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    return train_dataloader, valid_dataloader

def train_dstgcn(train_dataloader, valid_dataloader, num_epochs=3,return_class_report_dict=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #train_dataloader, valid_dataloader = build_dataloaders(train_years=[2013, 2014], valid_years=[2013, 2014], train_months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'], valid_months=['12'])
    ## MODEL ##
    model = DSTGCN().to(device)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f'There are {num_param} parameters in the model.')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, threshold=1e-3, min_lr=1e-6)

    train_losses = []
    valid_losses = []
    report_dict = None
    best_recall = -1
    for epoch in range(num_epochs):
        model.train()
        for i, (spatial_features, temporal_features, external_features, y, edges) in enumerate(train_dataloader):
            spatial_features, temporal_features, external_features, y, edges = \
                spatial_features.squeeze(), temporal_features.squeeze(), external_features.squeeze(), y.squeeze(), edges.squeeze()
            spatial_features.to(device), temporal_features.to(device), external_features.to(device), y.to(device), edges.to(device) 
            ratio = y.numel() / y.sum()
            print(ratio)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, ratio+.2]))
            criterion.to(device)
            optimizer.zero_grad()
            out = model(spatial_features, temporal_features, external_features, edges)
            loss = criterion(out.permute(0,2,1), y)
            loss.backward()
            optimizer.step()
            train_losses += [loss.item()]
            print(f'Epoch: {epoch} \t Iteration: {i} \t Train Loss: {train_losses[-1]}')
            verbose_output(out.cpu(), y.cpu())
            scheduler.step(loss)
        model.eval() # turn off dropout
        for i, (spatial_features, temporal_features, external_features, y, edges) in enumerate(valid_dataloader):
            spatial_features, temporal_features, external_features, y, edges = \
                spatial_features.squeeze(), temporal_features.squeeze(), external_features.squeeze(), y.squeeze(), edges.squeeze() 
            spatial_features.to(device), temporal_features.to(device), external_features.to(device), y.to(device), edges.to(device) 
            with torch.no_grad():
                out = model(spatial_features, temporal_features, external_features, edges)
                loss = criterion(out.permute(0,2,1).squeeze(1), y)
                valid_losses += [loss.item()]            
            print(f'Epoch: {epoch} \t Valid Loss: {valid_losses[-1]}')
            metrics = verbose_output(out.cpu(), y.cpu())
            if metrics['macro avg']['recall'] > best_recall:
                best_recall = metrics['macro avg']['recall']
                report_dict = metrics
                torch.save(model.state_dict(), f'saved_models/best_DSTGCN.pt')
    
    torch.save(model.state_dict(), 'saved_models/DSTGCN.pt')
    if return_class_report_dict:
        return report_dict
    
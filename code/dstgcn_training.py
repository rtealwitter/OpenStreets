from .data import TrafficDataset, connect_taxi_to_nodes, get_flows, get_taxi_data
import pandas as pd
import numpy as np

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
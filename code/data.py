import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import pickle
# pip install azureml-opendatasets-runtimeusing
#from azureml.opendatasets import NycTlcYellow
import calendar
from dateutil import parser
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import torch
import os.path
import momepy


# Only need to run this function once
def preprocess_lion():
    # Download data from https://www.dropbox.com/sh/927yoof5wq6ukeo/AAA--Iyb7UUDhfWIF2fncppba?dl=0
    # Put all files into 'data_unwrangled/LION' or change path below
    lion_folder = 'data_unwrangled/LION/'
    # Load all LION data
    links = gpd.read_file(lion_folder+'links.shp')
    # Only consider links in Manhattan
    links = links[links['LBoro']==1]
    # Only consider links that are normal streets
    links = links[links['FeatureTyp']=='0']
    # Only consider constructed links
    links = links[links['Status']=='2']
    # Only consider links that have vehicular traffic
    links = links[links['TrafDir'] != 'P']
    # Make sure there is a speed limit for each link
    links = links[links['POSTED_SPE'].notnull()]
    # Expected time to travel link at posted speed
    links['expected_time'] = links['POSTED_SPE'].astype(int)*links['SHAPE_Leng']
    # Ensure *undirected* graph is connected
    # Note: We could do this for directed graph but maximum size
    # of strongly connected component is 430
    graph = momepy.gdf_to_nx(links, approach="primal", directed=False)
    for component in nx.connected_components(graph):
        if len(component) > 10000:
            graph = graph.subgraph(component)
    # Use resulting links as infrastructure
    _, links = momepy.nx_to_gdf(graph)
    links.drop(columns=['node_start', 'node_end'], inplace=True)
    # Save both links so we can use it to construct directed graph
    links.to_file('data/links.json', driver='GeoJSON')
    # Load nodes
    nodes = gpd.read_file(lion_folder+'nodes.shp')
    # Drop unnecessary columns
    nodes.drop(columns=['OBJECTID_1', 'OBJECTID', 'GLOBALID', 'VIntersect'], inplace=True)
    # Find nodes that are connected to surviving links
    node_IDs = np.union1d(links['NodeIDFrom'], links['NodeIDTo']).astype(int)
    # Select nodes that are connected to surviving links
    selected_nodes = nodes[nodes['NODEID'].isin(node_IDs)]
    # Save to file
    selected_nodes.to_file('data/nodes.json', driver='GeoJSON')

# Only need to run this function once
# Rerun if we change the links data!
def preprocess_dual_graph():
    links = gpd.read_file('data/links.json')
    # Get outgoing edges from each node
    outgoing_edges = {}
    total = 0
    for objectid, from_node, to_node, trafdir in zip(links['OBJECTID'], links['NodeIDFrom'], links['NodeIDTo'], links['TrafDir']):
        if trafdir == 'W' or trafdir == 'T':
            if to_node not in outgoing_edges:
                outgoing_edges[to_node] = []
            outgoing_edges[to_node] += [objectid]
        if trafdir == 'A' or trafdir == 'T':
            if from_node not in outgoing_edges:
                outgoing_edges[from_node] = []
            outgoing_edges[from_node] += [objectid]
    # Build graph
    graph = nx.DiGraph()
    for objectid, from_node, to_node, trafdir in zip(links['OBJECTID'], links['NodeIDFrom'], links['NodeIDTo'], links['TrafDir']):
        graph.add_node(objectid)
        if trafdir == 'W' or trafdir == 'T':
            for outgoing_objectid in outgoing_edges[to_node]:
                graph.add_node(outgoing_objectid)
                graph.add_edge(objectid, outgoing_objectid)
        if trafdir == 'A' or trafdir == 'T':
            for outgoing_objectid in outgoing_edges[from_node]:
                graph.add_node(outgoing_objectid)
                graph.add_edge(objectid, outgoing_objectid)
    # Make sure we have correct number of nodes
    assert len(graph.nodes) == len(links['OBJECTID'].unique())
    pickle.dump(graph, open('data/dual_graph.pkl', 'wb'))
    return graph

def load_filter():
    filename_filter = 'data_unwrangled/2010 Neighborhood Tabulation Areas (NTAs).geojson'
    filter = gpd.read_file(filename_filter)
    filter = filter[filter['boro_name'] == 'Manhattan']
    return filter

def connect_collisions_to_links(collisions):
    links = gpd.read_file('data/links.json')
    links = links[['OBJECTID', 'geometry']]
    collisions.to_crs(links.crs, inplace=True)
    return collisions.sjoin_nearest(links).drop(columns=['index_right'])

# Only need to run this function once for each year
def preprocess_collisions(year=2013):
    filename_collisions = 'data_unwrangled/Motor_Vehicle_Collisions_-_Crashes.csv'
    # Load collisions and drop empty rows
    df = pd.read_csv(filename_collisions, low_memory=False).dropna(subset=['LATITUDE', 'LONGITUDE', 'CRASH DATE'])
    # Drop empty location data
    df = df[df.LONGITUDE != 0] # remove 0,0 coordinates
    # Convert date to datetime
    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
    # Get year
    df['year'] = df['CRASH DATE'].dt.year
    # Convert to geodataframe
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE))
    # Filter to Manhattan
    gdf = gdf.sjoin(load_filter()).drop(columns=['index_right'])
    # Subset to year
    gdf_year = gdf[gdf['year']==year]    
    # Connect collisions to nodes
    gdf_year = connect_collisions_to_links(gdf_year)
    # Save to file
    gdf_year.to_file(f'data/collisions_{year}.json', driver='GeoJSON')

def preprocess_taxi(df):
    # Make sure rides are longer than one minute
    df = df[df['tpepDropoffDateTime'] - df['tpepPickupDateTime'] > np.timedelta64(1, 'm')]
    # Make sure rides are shorter than 12 hours
    df = df[df['tpepDropoffDateTime'] - df['tpepPickupDateTime'] <= np.timedelta64(12, 'h')]
    # Make sure rides are longer than .1 mile
    df = df[df['tripDistance'] > 0.1]
    # Make sure fare is non-zero 
    df = df[df['fareAmount'] > 0.0]
    # Convert to geopandas
    gdf = gpd.GeoDataFrame(df)
    # Reset index ID (there are duplicate indices)
    gdf.reset_index(inplace=True)
    # Create ride ID
    gdf['ride_id'] = gdf.index
    # Make start time date time type
    gdf['start_time'] = pd.to_datetime(gdf['tpepPickupDateTime'])
    # Round start time to day
    gdf['start_day'] = gdf['start_time'].dt.round('d')
    return gdf

def filter_location(type, filter, taxi, make_copy=True):
    # Create a geometry column from the type coordinates
    taxi[f'{type}_geom'] = gpd.points_from_xy(taxi[f'{type}Lon'], taxi[f'{type}Lat'])
    taxi.set_geometry(f'{type}_geom', crs='epsg:4326', inplace=True)
    taxi = taxi.sjoin(filter).drop(columns=['index_right'])
    return taxi

def restrict_start_end(taxi, check_ratio=False):        
    # Load Manhattan objects
    filter_manhattan = load_filter()
    # Restrict to rides that start in Manhattan
    taxi_start = filter_location('start', filter_manhattan, taxi)
    # Restrict to rides that start and end in Manhattan
    taxi_start_end = filter_location('end', filter_manhattan, taxi_start)
    if check_ratio:
        # Check number of rides that start AND end in Manhattan / number of rides that start OR end in Manhattan
        taxi_end = filter_location('end', filter_manhattan, taxi)
        print(len(taxi_start_end)/(len(taxi_start)+len(taxi_end)-len(taxi_start_end))) # About 85%
    return taxi_start_end

def get_taxi_data(year, month, check_ratio=False):
    # Get query for first and last day of month in year
    month_last_day = calendar.monthrange(year=int(year),month=int(month))[1]
    start_date = parser.parse(str(year)+'-'+str(month)+'-01')
    end_date = parser.parse(str(year)+'-'+str(month)+'-'+str(month_last_day))
    #end_date = parser.parse(str(year)+'-'+str(month)+'-02')
    print('Loading taxi data...', end=' ')
    nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date)
    taxi_all = nyc_tlc.to_pandas_dataframe()
    print('complete!')
    print('Preprocessing data...', end=' ')
    taxi = preprocess_taxi(taxi_all)
    print('complete!')
    print('Restricting start and end...', end=' ')
    taxi_start_end = restrict_start_end(taxi, check_ratio)
    print('complete!')

    return taxi_start_end

def get_directed_graph(links):
    # Edges from NodeIDFrom to NodeIDTo for one-way "with" streets and two-way streets
    graph1 = nx.from_pandas_edgelist(
        links[np.logical_or(links['TrafDir'] == 'W', links['TrafDir'] == 'T')],
        source='NodeIDFrom', target='NodeIDTo', edge_attr=True, create_using=nx.DiGraph()
    )
    # Edges from NodeIDTo to NodeIDFrom for one-way "against" streets and two-way streets
    graph2 = nx.from_pandas_edgelist(
        links[np.logical_or(links['TrafDir'] == 'A', links['TrafDir'] == 'T')],
        source='NodeIDTo', target='NodeIDFrom', edge_attr=True, create_using=nx.DiGraph()
    )
    return nx.compose(graph1, graph2)

def connect_taxi_to_nodes(taxi, type_name, nodes):    
    taxi.set_geometry(type_name+'_geom', inplace=True)
    taxi.to_crs(nodes.crs, inplace=True)
    result = taxi.sjoin_nearest(nodes).drop(columns=['index_right'])
    result.rename(columns={'NODEID': type_name+'_NODEID'}, inplace=True)
    return result

# About 8 minutes for one million trips
def get_flows(taxi, graph, links):
    # Initialize dictionary for fast access
    flow_day = {'increasing_order': {}, 'decreasing_order': {}}
    for objectid, trafdir in zip(links['OBJECTID'], links['TrafDir']):
        flow_day['increasing_order'][objectid] = 0
        flow_day['decreasing_order'][objectid] = 0
    flows = {np.datetime_as_string(day, unit='D') : dict(flow_day) for day in taxi['start_day'].unique()}
    # Sort by start node so we can re-use predecessor graph
    taxi_sorted = taxi.sort_values(by=['start_NODEID', 'end_NODEID'])
    previous_source = None
    for source, target, day in zip(taxi_sorted['start_NODEID'], taxi_sorted['end_NODEID'], taxi_sorted['start_day']):
        # Networkx pads node ID with leading zeroes
        source_padded = str(source).zfill(7)
        target_padded = str(target).zfill(7)
        day_pretty = np.datetime_as_string(np.datetime64(day), unit='D')
        # If we haven't already computed the predecessor graph
        if previous_source != source_padded:
            # Compute predecessor graph
            pred, dist = nx.dijkstra_predecessor_and_distance(graph, source=source_padded, weight='expected_time') 
        # We ignore taxi rides that appear infeasible in the directed graph
        if target_padded not in pred:
            continue
        # Follow predecessors to get path
        current, previous = target_padded, None
        while current != source_padded:
            current, previous = pred[current][0], current
            edge_id = (current, previous)
            objectid = graph.edges[edge_id]['OBJECTID']
            if current < previous: # string comparison
                flows[day_pretty]['increasing_order'][objectid] += 1
            else:
                flows[day_pretty]['decreasing_order'][objectid] += 1
        previous_source = source_padded
    return flows

# NOTE: Turns out, the WT-- columns signify whether an extreme 
# weather took place or not. If not, they get a NaN value. So NaNs
# should be replaced with 0. Other NaNs are sparse, so we can fill forward.
def preprocess_weather(years=[2013]):
    # Convert to int because that's how it's stored in the dataframe
    years = [int(year) for year in years]
    df = pd.read_csv('data/weather.csv')
    df['date'] = pd.to_datetime(df.DATE)
    df['year'] = df.date.dt.year
    # Restrict to years we want
    df = df[df.year.isin(years)]
    # If we want more, we can one hot encode the NAN values
    severe_weather_columns = [col for col in df if col.startswith('WT')]
    df[severe_weather_columns] = df[severe_weather_columns].fillna(0.0)
    # For columns missing only a few values, fill forward seems reasonable
    fill_forward_columns = ['AWND','WDF2','WDF5','WSF2','WSF5']
    df[fill_forward_columns] = df[fill_forward_columns].fillna(method='ffill')
    df = df[df.columns[df.isna().sum() == 0]]

    # Normalize weather data
    df_num = df.select_dtypes(include='number')
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df_num)
    df_normalized = pd.DataFrame(np_scaled, columns = df_num.columns)
    df[df_normalized.columns] = df_normalized

    return df


def prepare_links(links):    
    # Remove columns with missing values
    links_modified = links[links.columns[links.isna().sum() == 0]]

    # Remove columns with unnecessary values
    links_drop_columns = ['Street', 'FeatureTyp', 'FaceCode', 'SeqNum', 'StreetCode', 'LGC1', 'BOE_LGC', 'SegmentID', 'LBoro', 'RBoro', 'L_CD', 'R_CD', 'LATOMICPOL', 'RATOMICPOL', 'LCT2020', 'RCT2020', 'LCB2020', 'RCB2020', 'LCT2010', 'RCT2010', 'LCB2010', 'RCB2010', 'LCT2000', 'RCT2000', 'LCB2000', 'RCB2000', 'LCT1990', 'RCT1990', 'LAssmDist', 'LElectDist', 'RAssmDist', 'RElectDist', 'MapFrom', 'MapTo', 'XFrom', 'YFrom', 'XTo', 'YTo', 'ArcCenterX', 'ArcCenterY', 'NodeIDFrom', 'NodeIDTo', 'PhysicalID', 'GenericID', 'LegacyID', 'FromLeft', 'ToLeft', 'FromRight', 'ToRight', 'Join_ID', 'mm_len', 'geometry']
    links_modified = links_modified.drop(columns=links_drop_columns)

    # Add back columns with missing values that are useful
    links_add_columns = ['NonPed', 'BikeLane', 'Snow_Prior', 'Number_Tra', 'Number_Par', 'Number_Tot']
    for column_name in links_add_columns:
        links_modified[column_name] = links[column_name]

    # Convert categorical columns to one hot encoding
    links_categorical_columns = ['SegmentTyp', 'RB_Layer', 'TrafDir', 'NodeLevelF', 'NodeLevelT', 'RW_TYPE', 'Status'] + links_add_columns
    for column_name in links_categorical_columns:
        links_modified = pd.concat([links_modified, pd.get_dummies(links_modified[column_name], prefix=column_name, dummy_na=True)], axis=1)
        links_modified = links_modified.drop(columns=[column_name])
        
    return links_modified.astype(int)

def get_X_day(data_constant, weather, flows_day, day):
    # Make a deep copy of the constant link data
    data = data_constant.copy(deep=True)
    # Add weather data
    weather_day = weather.loc[weather['DATE'] == day].drop(columns=['STATION', 'NAME', 'DATE', 'date', 'year'])
    # Weather is the same for every link (only one weather station)
    for column_name in weather_day: data[column_name] = weather_day[column_name].values[0]
    # Get flow data on day
    flow_day = pd.DataFrame.from_dict(flows_day)
    # Make sure the index is the same as the link data
    flow_day['OBJECTID'] = flow_day.index
    # Make both indices the same
    flow_day.set_index('OBJECTID', inplace=True)
    data.set_index('OBJECTID', inplace=True)
    # Merge the flow data into the link data
    data = data.merge(flow_day, on='OBJECTID')
    # Make sure the index is sorted so it connects to labels
    data.sort_index(inplace=True)
    return data


def get_X(data_constant, weather, flows):
    X = []
    for day in flows.keys():
        data = get_X_day(data_constant, weather, flows[day], day)
        X += [data.values]
        
    return torch.tensor(np.array(X))

def get_y_day(collisions, links, day):
    label = {objectid : 0 for objectid in links.OBJECTID}
    for crash_day, objectid in zip(collisions['CRASH DATE'], collisions['OBJECTID']):
        crash_day_pretty = np.datetime_as_string(np.datetime64(crash_day), unit='D')
        if day == crash_day_pretty: label[objectid] = 1
    label = pd.DataFrame.from_dict(label, orient='index', columns=['crashes'])
    label.sort_index(inplace=True)
    return label

def get_y(collisions, links, flows):
    y = []
    for day in flows.keys():
        label = get_y_day(collisions, links, day)
        y += [label.values]
    return torch.tensor(np.array(y))

class TrafficDataset(Dataset):
    def __init__(self, years=['2013'], months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']):
        # Should take under a minute to load
        self.links = gpd.read_file('data/links.json')        
        self.nodes = gpd.read_file('data/nodes.json')        
        #self.graph = get_directed_graph(self.links)
        #TODO: Use momepy to get directed graph instead of trusting the data
        self.graph = momepy.gdf_to_nx(self.links, directed=True)
        list_of_year_collisions = []
        for year in years:
            list_of_year_collisions.append(gpd.read_file(f'data/collisions_{year}.json'))
        self.collisions = gpd.GeoDataFrame( pd.concat( list_of_year_collisions, ignore_index=True) )
        #print(len(self.collisions))
        # If we change years, different weather features will be returned
        # because we eliminate columns with missing values
        self.weather = preprocess_weather(years)
        self.year_months = [(year, month) for year in years for month in months]
        self.data_constant = prepare_links(self.links)
        dual_graph = pickle.load(open('data/dual_graph.pkl', 'rb'))
        # Relabel so we can plug into GCN
        assert 0 not in dual_graph.nodes # check we're not already relabeled
        mapping = dict(zip(sorted(self.links['OBJECTID']), range(len(self.links))))
        nx.relabel_nodes(dual_graph, mapping, copy=False)
        assert 0 in dual_graph.nodes # check the relabeling worked
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.edges = torch.tensor(np.array(list(dual_graph.edges))).long().to(self.device).T
        filename_edges = 'loaded_data/edges.pkl'
        pickle.dump(self.edges, open(filename_edges, 'wb'))
    
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

        filename_X = f'loaded_data/{year}_{month}_X.pkl'
        filename_y = f'loaded_data/{year}_{month}_y.pkl'
        # NOTE: Make sure you have a ``loaded_data/'' directory
        if os.path.isfile(filename_X):
            X = pickle.load(open(filename_X, 'rb'))
            y = pickle.load(open(filename_y, 'rb'))
        else:
            X = get_X(self.data_constant, self.weather, flows).float()
            y = get_y(self.collisions, self.links, flows)

            pickle.dump(X, open(filename_X, 'wb'))
            pickle.dump(y, open(filename_y, 'wb'))
            
        return X.to(self.device), y.to(self.device), self.edges

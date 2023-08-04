import json
import geopandas as gpd
import numpy as np

links = gpd.read_file('data/links.json')
openstreets = gpd.read_file('data/Open_Streets_Locations.csv')

mask = np.isin(links['SegmentID'], openstreets['segmentidt'])

try:
    print(list(links[mask]['OBJECTID']))
except:
    print('print(list(links[mask].index) objectid) failed')

try:
    print(links[mask].index.tolist())
except:
    print('print(links[mask].index.tolist()) failed')

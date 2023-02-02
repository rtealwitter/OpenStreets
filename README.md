## Recurrent Graph Neural Networks and Deep Q Learning
for Reducing Traffic and Collisions in Road Networks

### Code

All code for the paper appears in the `code` folder.

### Data

The data sources we use (listed below) tend to be quite large. We preprocessed a bunch of them and saved them in the `data` folder.

### Flows

The taxi data set is particularly large. We load each individually and infer routes using Dijkstra's weighted shortest path algorithm. The saved flows appear in the `flows` folder.

### Saved Models

Many of our models took substantial resources to train. We generally saved their weights in the `saved_models` folder.

### Figures

We made one (very pretty) figure of Q values in Manhattan according to our deep Q GNN. You should check it out!

### Data Sources

• [NYC Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)

• [NYC Infrastructure](https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/exjm-f27b)

• [NOAA Daily Weather Data](https://www.ncdc.noaa.gov/cdo-web/datatools)

• [NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

#### Installation notes
If you have issues installing torch-sparse, etc., try running: `pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html --force-reinstall` (you may need to replace torch/cude versions with whatevers in your environment).

#### Build LightGBM for GPU
https://stackoverflow.com/questions/60360750/lightgbm-classifier-with-gpu

### Data Sources

• [NYC Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)

• [NYC Infrastructure](https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/exjm-f27b)

• [NOAA Daily Weather Data](https://www.ncdc.noaa.gov/cdo-web/datatools)

• [Citibike Data](https://s3.amazonaws.com/tripdata/index.html)

• [NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

New idea for CS paper: Analyze car traffic with taxi data and the effect of shutting down road segments on traffic and collisions.

#### Installation notes
If you have issues installing torch-sparse, etc., try running: `pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html --force-reinstall` (you may need to replace torch/cude versions with whatevers in your environment).

#### Build LightGBM for GPU
https://stackoverflow.com/questions/60360750/lightgbm-classifier-with-gpu
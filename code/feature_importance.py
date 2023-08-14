# Libraries
import torch
import models
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np

# Files
import data
import training
import benchmark

# Column names
column_names = ['SegCount', 'Radius', 'StreetWidt', 'StreetWi_1', 'POSTED_SPE', 'SHAPE_Leng', 'expected_time', 'SegmentTyp_B', 'SegmentTyp_C', 'SegmentTyp_E', 'SegmentTyp_F', 'SegmentTyp_G', 'SegmentTyp_R', 'SegmentTyp_S', 'SegmentTyp_T', 'SegmentTyp_U', 'SegmentTyp_nan', 'RB_Layer_B', 'RB_Layer_G', 'RB_Layer_R', 'RB_Layer_nan', 'TrafDir_A', 'TrafDir_T', 'TrafDir_W', 'TrafDir_nan', 'NodeLevelF_*', 'NodeLevelF_D', 'NodeLevelF_E', 'NodeLevelF_I', 'NodeLevelF_M', 'NodeLevelF_O', 'NodeLevelF_Q', 'NodeLevelF_S', 'NodeLevelF_U', 'NodeLevelF_V', 'NodeLevelF_Y', 'NodeLevelF_nan', 'NodeLevelT_*', 'NodeLevelT_D', 'NodeLevelT_E', 'NodeLevelT_I', 'NodeLevelT_M', 'NodeLevelT_O', 'NodeLevelT_Q', 'NodeLevelT_S', 'NodeLevelT_U', 'NodeLevelT_V', 'NodeLevelT_Y', 'NodeLevelT_nan', 'RW_TYPE_1', 'RW_TYPE_12', 'RW_TYPE_13', 'RW_TYPE_2', 'RW_TYPE_3', 'RW_TYPE_4', 'RW_TYPE_7', 'RW_TYPE_9', 'RW_TYPE_nan', 'Status_2', 'Status_nan', 'NonPed_D', 'NonPed_V', 'NonPed_nan', 'BikeLane_1', 'BikeLane_10', 'BikeLane_11', 'BikeLane_2', 'BikeLane_3', 'BikeLane_4', 'BikeLane_5', 'BikeLane_6', 'BikeLane_8', 'BikeLane_9', 'BikeLane_nan', 'Snow_Prior_C', 'Snow_Prior_H', 'Snow_Prior_S', 'Snow_Prior_V', 'Snow_Prior_nan', 'Number_Tra_1', 'Number_Tra_10', 'Number_Tra_2', 'Number_Tra_3', 'Number_Tra_4', 'Number_Tra_5', 'Number_Tra_6', 'Number_Tra_7', 'Number_Tra_8', 'Number_Tra_nan', 'Number_Par_1', 'Number_Par_2', 'Number_Par_4', 'Number_Par_nan', 'Number_Tot_1', 'Number_Tot_10', 'Number_Tot_2', 'Number_Tot_3', 'Number_Tot_4', 'Number_Tot_5', 'Number_Tot_6', 'Number_Tot_7', 'Number_Tot_8', 'Number_Tot_nan', 'AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'WDF2', 'WDF5', 'WSF2', 'WSF5', 'WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT13', 'WT16', 'WT18', 'WT19', 'WT22', 'increasing_order', 'decreasing_order']

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
X = X[0].unsqueeze(0)
y = y[0].unsqueeze(0)

# Put everything on device
X = X.to(device)
y = y.to(device)
edges = edges.to(device)

# Set the data to track the gradients
X.requires_grad = True

def modified_model(X):
    preds = model(X, edges)
    return preds.squeeze()

# Use captum to compute the attributions
ig = IntegratedGradients(modified_model)
attr = ig.attribute(X, target=y.flatten(), n_steps=1)
attr = attr.detach().cpu().numpy().sum(axis=1)
attributes = list(attr[0])

# Sort by importance
indices = np.argsort([-x**2 for x in attributes])

for index in indices:
    print(f'Column Name: {column_names[index]} \t \t Importance: {attributes[index]}')

attribution = dict(zip(column_names, attributes))

# Clean attributions for plotting
attribution_cleaned = {
    'Number of Cars' : attribution['decreasing_order'] + attribution['increasing_order'],
    'Travel Time' : attribution['expected_time'],
    'Street Length' : attribution['SHAPE_Leng'],
    'Street Width' : sum([attribution[x] for x in column_names if 'StreetWi' in x]),
    'Speed Limit' : attribution['POSTED_SPE'],
    'Radius' : attribution['Radius'],
    'Double Leveled' : attribution['SegCount'],
    'Border' : attribution['Status_2'],
    'Bike Lane' : sum([attribution[x] for x in column_names if 'BikeLane' in x ])
}

# Plot the attributions
def visualize_importances(feature_names, importances, title="Average Feature Importances for Predicting Collisions"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    color = ['red' if x > 0 else 'green' for x in importances]
    plt.figure(figsize=(12,6))
    plt.bar(x_pos, importances, color=color, align='center')
    plt.xticks(x_pos, feature_names, rotation=30)
    plt.ylabel('(Directed) Importance')
    plt.title(title)
    plt.savefig('/figures/feature_importances.pdf', bbox_inches='tight')

visualize_importances(
    list(attribution_cleaned.keys()), list(attribution_cleaned.values())
)
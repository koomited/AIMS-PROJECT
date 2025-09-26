#!/usr/bin/env python
# coding: utf-8

#nohup python training_on_hrest0.py > evaluation.log 2>&1 &


import xarray as xr
from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata, rollout
import matplotlib.pyplot as plt

from pathlib import Path

import cdsapi
import numpy as np
from sklearn.metrics import root_mean_squared_error
import gcsfs

from torch.utils.data import Dataset
from aurora import Batch, Metadata
import os
from matplotlib.colors import TwoSlopeNorm


import seaborn as sns



import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter  # <-- Add this import



import sys
sys.path.append(os.path.abspath("../src"))
from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses, create_hrest0_batch


# In[78]:


from evaluation import evaluation
from lora import create_custom_model, full_linear_layer_lora

torch.use_deterministic_algorithms(True)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# # Data

# In[79]:


fs = gcsfs.GCSFileSystem(token="anon")

store = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store, consolidated=True, chunks=None)



start_time, end_time = '2022-01-01', '2022-12-31' 



lat_max = -22.00 
lat_min = -37.75  

lon_min = 15.25   
lon_max = 35.00   
sliced_era5_SA = (
    full_era5
    .sel(
        time=slice(start_time, end_time),
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)  
    )
)

################################"" get hres data
store_hrest0 = fs.get_mapper('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
full_hrest0 = xr.open_zarr(store=store_hrest0, consolidated=True, chunks=None)
sliced_hrest0_sa = full_hrest0.sel(time=slice(start_time, end_time), 
                                   latitude=slice(lat_min, lat_max), 
                                   longitude=slice(lon_min, lon_max))


model_initial = AuroraSmall(
    use_lora=False,  # fine_tuned_Model was not fine-tuned.
)

model_initial.load_state_dict(torch.load('../model/urora-0.25-small-pretrained1.pth'))

fine_tuned_model = AuroraSmall(
    use_lora=False,  # fine_tuned_Model was not fine-tuned.
)
fine_tuned_model = full_linear_layer_lora(fine_tuned_model, lora_r = 16, lora_alpha = 4)
checkpoint = torch.load('../model/training/hrest0/wampln/checkpoint_epoch_18.pth')

fine_tuned_model.load_state_dict(checkpoint['model_state_dict'])
print("Loading fine_tuned_Model from checkpoint")


# In[82]:


results = evaluation(fine_tuned_model, model_initial, sliced_era5_SA, sliced_hrest0_sa)


# In[83]:


counter = results['counter']
surface_rmses_fine_tuned = results['surface_rmses_fine_tuned']
atmospheric_rmses_fine_tuned = results['atmospheric_rmses_fine_tuned']
surface_rmses_non_fine_tuned = results['surface_rmses_non_fine_tuned']
atmospheric_rmses_non_fine_tuned = results['atmospheric_rmses_non_fine_tuned']



relative_surface_rmses = {}


# In[85]:


for surf_var, rmses in surface_rmses_fine_tuned.items():
    
    relative_surface_rmses[surf_var] = (surface_rmses_fine_tuned[surf_var]-surface_rmses_non_fine_tuned[surf_var])/surface_rmses_non_fine_tuned[surf_var]*100
    



relative_atmospheric_rmses = {}


# In[88]:


for atmos_var, rmses in atmospheric_rmses_fine_tuned.items():
    
    relative_atmospheric_rmses[atmos_var] = (atmospheric_rmses_fine_tuned[atmos_var]-atmospheric_rmses_non_fine_tuned[atmos_var])/atmospheric_rmses_non_fine_tuned[atmos_var]*100
    


surface_variables_names = ["2m temperature", "10m wind speed u", "10m wind speed v", "Mean sea level pressure"]
atmospheric_variables_names = ["Geopotential", "Specific humidity", "Temperature", "Wind speed u", "Wind speed v"]

# num_atmospheric = len(atmospheric_rmses_fine_tuned)
# num_surface = len(surface_rmses_fine_tuned)
n_cols = 5

save_path = "../report/evaluation/wampln"




#
# Compute global vmin and vmax for color consistency
all_values = np.concatenate(
    [relative_atmospheric_rmses[var].flatten() for var in atmospheric_rmses_fine_tuned] +
    [relative_surface_rmses[var].flatten() for var in surface_rmses_fine_tuned]
)
vmin, vmax = np.min(all_values), np.max(all_values)
abs_max = 60#max(abs(vmin), abs(vmax))
# Create a norm centered at 0
norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# Create the figure and gridspec layout with space for vertical colorbar
fig = plt.figure(figsize=(25, 6), dpi=300)
n_cols = 5
gs = GridSpec(2, n_cols, height_ratios=[1, 0.1], figure=fig)

from matplotlib.ticker import FuncFormatter

# Set global font size defaults (optional)
plt.rcParams.update({'font.size': 22})

# Define custom font sizes
label_fontsize = 22
tick_fontsize = 20
title_fontsize = 24.5

for i, variable in enumerate(atmospheric_rmses_fine_tuned):
    ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
    sns.heatmap(relative_atmospheric_rmses[variable], cmap="RdBu_r", cbar=False, norm=norm, ax=ax)

    ax.set_xlabel("Lead Time (Hours)", fontsize=label_fontsize)

    if i % n_cols == 0:
        ax.set_ylabel("Pressure levels (hPa)", fontsize=label_fontsize)
        ax.set_yticks(np.arange(0.5, 13, 1))
        full_labels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        cleaned_labels = [label if idx % 2 == 0 else "" for idx, label in enumerate(full_labels)]
        ax.set_yticklabels(cleaned_labels, fontsize=tick_fontsize, rotation=0)  # Horizontal ticks
    else:
        ax.set_yticks([])

    ax.set_title(f"{atmospheric_variables_names[i]}", fontsize=title_fontsize)
    ax.set_xticks(np.arange(0.5, 8.5, 1))
    ax.set_xticklabels(np.arange(6, 48+6, 6), fontsize=tick_fontsize)

    # Set tick label font size
    ax.tick_params(axis='both', labelsize=tick_fontsize)

# Plot surface heatmaps
for j, variable in enumerate(surface_rmses_fine_tuned):
    ax = fig.add_subplot(gs[1, j])
    sns.heatmap(relative_surface_rmses[variable].reshape(1, -1), cmap="RdBu_r", cbar=False, norm=norm, ax=ax)

    ax.set_xlabel("Lead Time (Hours)", fontsize=label_fontsize)
    ax.set_ylabel("")
    ax.set_title(f"{surface_variables_names[j]}", fontsize=title_fontsize)
    ax.set_yticks([])

    ax.set_xticks(np.arange(0.5, 8.5, 1))
    ax.set_xticklabels(np.arange(6, 48+6, 6), fontsize=tick_fontsize)

    # Set tick label font size
    ax.tick_params(axis='both', labelsize=tick_fontsize)

# Add vertical colorbar on the right side
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cbar_ax)

# Format colorbar ticks with +/- signs and percent
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}%"))
cbar.ax.tick_params(labelsize=tick_fontsize)

# Final layout and saving
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(f"{save_path}/scorecard.pdf", bbox_inches="tight")
plt.savefig(f"{save_path}/scorecard.png", bbox_inches="tight")
plt.savefig(f"{save_path}/scorecard.svg", bbox_inches="tight")

plt.show()
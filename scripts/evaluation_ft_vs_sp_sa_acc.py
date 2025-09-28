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

# Load 6-hourly climatology
clim = xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr")


clim_sa = (
    clim
    .sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)  
    )
)

results = evaluation(fine_tuned_model, model_initial, sliced_era5_SA, sliced_hrest0_sa, clim_sa)


# In[83]:


counter = results['counter']
surface_acc_fine_tuned = results['surface_acc_fine_tuned']
atmospheric_acc_fine_tuned = results['atmospheric_acc_fine_tuned']
surface_acc_non_fine_tuned = results['surface_acc_non_fine_tuned']
atmospheric_acc_non_fine_tuned = results['atmospheric_acc_non_fine_tuned']

surface_acc_fine_tuned = {var:values/counter for var, values in surface_acc_fine_tuned.items()}
atmospheric_acc_fine_tuned = {var:values/counter for var, values in atmospheric_acc_fine_tuned.items()}
surface_acc_non_fine_tuned = {var:values/counter for var, values in surface_acc_non_fine_tuned.items()}
atmospheric_acc_non_fine_tuned = {var:values/counter for var, values in atmospheric_acc_non_fine_tuned.items()}




lead_time = [6, 12, 18, 24, 30, 36, 42, 48]

SELECTED_ATMOS_LEVELS = {7:"500 hPa", 9: "700 hPa",  10:"850 hPa"}

# selecte the require data for plotting
fine_tune_atmos_acc={}
# z_500

fine_tune_atmos_acc["Geopotential at 500hPa"] = atmospheric_acc_fine_tuned['z'][7]
# t 850
fine_tune_atmos_acc["Temperature at 850hPa"] = atmospheric_acc_fine_tuned['t'][10]
# q 700
fine_tune_atmos_acc["Specific humidity at 700hPa"] = atmospheric_acc_fine_tuned['q'][9]
# u 850
fine_tune_atmos_acc["Eastward wind speed at 850hPa"] = atmospheric_acc_fine_tuned['u'][10]
# v 850
fine_tune_atmos_acc["Southward wind speed at 850hPa"] = atmospheric_acc_fine_tuned['v'][10]

non_fine_tune_atmos_acc = {}
# z_500
non_fine_tune_atmos_acc["Geopotential at 500hPa"] = atmospheric_acc_non_fine_tuned['z'][7]
# t 850
non_fine_tune_atmos_acc["Temperature at 850hPa"] = atmospheric_acc_non_fine_tuned['t'][10]
# q 700
non_fine_tune_atmos_acc["Specific humidity at 700hPa"] = atmospheric_acc_non_fine_tuned['q'][9]
# u 850
non_fine_tune_atmos_acc["Eastward wind speed at 850hPa"] = atmospheric_acc_non_fine_tuned['u'][10]
# v 850
non_fine_tune_atmos_acc["Southward wind speed at 850hPa"] = atmospheric_acc_non_fine_tuned['v'][10]





    
 # --- Global font settings ---
plt.rcParams.update({'font.size': 22})

# Define custom font sizes
label_fontsize = 22
tick_fontsize = 20
title_fontsize = 24.5

# --- Data setup ---
num_plots = len(fine_tune_atmos_acc)
num_plots_per_rows = 5
num_rows = 1
variables = list(fine_tune_atmos_acc.keys())

saving_path = "../report/evaluation/acc_grid/DLI"

# --- Figure and subplots ---
fig, axs = plt.subplots(num_rows, num_plots_per_rows, dpi=300, figsize=(40, 8))
axs = axs.ravel()

# Store handles and labels from the first plot for global legend
handles, labels = None, None

# --- Plot each variable ---
for i, ax in enumerate(axs[:num_plots]):
    line1, = ax.plot(lead_time, fine_tune_atmos_acc[variables[i]], label="Fine-tuned AuroraSmall", c="brown")
    line2, = ax.plot(lead_time, non_fine_tune_atmos_acc[variables[i]], label="Pretrained AuroraSmall", c="teal")
    
    ax.set_title(variables[i], fontsize=title_fontsize+2)
    ax.tick_params(axis='both', labelsize=tick_fontsize+2)
    ax.grid(True)

    # Capture legend handles/labels once
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# Turn off unused axes
for ax in axs[num_plots:]:
    ax.axis('off')

# --- Shared axis labels ---
fig.supxlabel("Lead Time (Hours)", x=0.5, y=0.05, fontsize=label_fontsize+4)
fig.supylabel("ACC", x=0.01, y=0.5, fontsize=label_fontsize)

# --- Shared legend ---
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, -0.07),  # Positioned below x-label
    frameon=False,
    fontsize=24
)

# --- Layout and saving ---
plt.tight_layout(rect=[0, 0.05, 1, 1])  
plt.savefig(f"{saving_path}/acc_fit_vs_pre_sa.pdf", bbox_inches="tight")
plt.savefig(f"{saving_path}/acc_fit_vs_pre_sa.png", bbox_inches="tight")
plt.savefig(f"{saving_path}/acc_fit_vs_pre_sa.svg", bbox_inches="tight")
plt.close()

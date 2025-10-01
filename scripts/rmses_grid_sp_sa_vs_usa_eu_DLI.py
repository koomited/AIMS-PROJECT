

import xarray as xr
from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata, rollout, Aurora
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


from evaluation_three_regions import evaluation_between_regions
from lora import create_custom_model, full_linear_layer_lora

torch.use_deterministic_algorithms(True)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 0.8,
    'grid.linewidth': 1.2,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.8,
    'figure.dpi': 150,
})


fs = gcsfs.GCSFileSystem(token="anon")

store = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store, consolidated=True, chunks=None)



start_time, end_time = '2022-01-01', '2022-12-31' 

  
sliced_era5 = (
    full_era5
    .sel(
        time=slice(start_time, end_time) 
    )
)

store_hrest0 = fs.get_mapper('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
full_hrest0 = xr.open_zarr(store=store_hrest0, consolidated=True, chunks=None)
sliced_hrest0 = full_hrest0.sel(time=slice(start_time, end_time))
#------------------------------1--------------------------------------
# model = AuroraSmall(
#     use_lora=False,  # fine_tuned_Model was not fine-tuned.
# )

# model.load_state_dict(torch.load('../model/aurora-0.25-small-pretrained.pth'))

model = Aurora(
    use_lora=False,  # fine_tuned_Model was not fine-tuned.
)

model.load_state_dict(torch.load('../model/aurora-0.25-pretrained_big.pth'))



#-----------------------------2----------------------------------------
# model = AuroraSmall(
#     use_lora=False,  # fine_tuned_Model was not fine-tuned.
# )
# model = full_linear_layer_lora(model, lora_r = 16, lora_alpha = 4)
# checkpoint = torch.load('../model/training/hrest0/wampln/checkpoint_epoch_4.pth')

# model.load_state_dict(checkpoint['model_state_dict'])


USA_REGION = (
      20.00,  # Southern boundary (approximate)
         54.99,  # Northern boundary (approximate)
         230,  # Western boundary (approximate)
         299.99  # Eastern boundary (approximate)
)

SA_REGION = (
      -37.75,  # Southern boundary (approximate)
         -22.00,  # Northern boundary (approximate)
         15.25,   # Western boundary (approximate)
         35.00,  # Eastern boundary (approximate)
)

EU_REGION = (
      35.00,  # Southern boundary (approximate)
         69.9,  # Northern boundary (approximate)
         -10.00,  # Western boundary (approximate)
         39.9  # Eastern boundary (approximate)
)


results = evaluation_between_regions(
                        model,
                        era5_data=sliced_era5, 
                        hres_data=sliced_hrest0,
                        target_region=SA_REGION,
                        base_region= USA_REGION,
                        third_region= EU_REGION,
               )



# eu_region_surface_rmses = results['third_region_surface_rmses']
eu_region_atmospheric_rmses = results['third_region_atmospheric_rmses']

# target_region_surface_rmses = results['target_region_surface_rmses']
target_region_atmospheric_rmses = results['target_region_atmospheric_rmses']
# base_region_surface_rmses = results['base_region_surface_rmses']
base_region_atmospheric_rmses = results['base_region_atmospheric_rmses']


# Selected variables names for atmospheric levels
# variables = list(target_region_atmospheric_rmses.keys())
# print("Selected variables:", variables)
# selected_variables = [ "z", "t", "q" "u", "v"]

# selected_variables_names= ["Geopotential", "Temperature", "Specific Humidity", "Eastward wind speed", "Southward wind speed"]

# selected_variables = {var: name for var, name in zip(variables, selected_variables_names) if var in selected_variables}


lead_time = [6, 12, 18, 24, 30, 36, 42, 48]

SELECTED_ATMOS_LEVELS = {7:"500 hPa", 9: "700 hPa",  10:"850 hPa"}

# selecte the require data for plotting
target_region_atmos_rmses={}
# z_500

target_region_atmos_rmses["Geopotential at 500hPa"] = target_region_atmospheric_rmses['z'][7]
# t 850
target_region_atmos_rmses["Temperature at 850hPa"] = target_region_atmospheric_rmses['t'][10]
# q 700
target_region_atmos_rmses["Specific humidity at 700hPa"] = target_region_atmospheric_rmses['q'][9]
# u 850
target_region_atmos_rmses["Eastward wind speed at 850hPa"] = target_region_atmospheric_rmses['u'][10]
# v 850
target_region_atmos_rmses["Southward wind speed at 850hPa"] = target_region_atmospheric_rmses['v'][10]

eu_region_atmos_rmses = {}
# z_500
eu_region_atmos_rmses["Geopotential at 500hPa"] = eu_region_atmospheric_rmses['z'][7]
# t 850
eu_region_atmos_rmses["Temperature at 850hPa"] = eu_region_atmospheric_rmses['t'][10]
# q 700
eu_region_atmos_rmses["Specific humidity at 700hPa"] = eu_region_atmospheric_rmses['q'][9]
# u 850
eu_region_atmos_rmses["Eastward wind speed at 850hPa"] = eu_region_atmospheric_rmses['u'][10]
# v 850
eu_region_atmos_rmses["Southward wind speed at 850hPa"] = eu_region_atmospheric_rmses['v'][10]

base_region_atmos_rmses = {}
# z_500
base_region_atmos_rmses["Geopotential at 500hPa"] = base_region_atmospheric_rmses['z'][7]
# t 850
base_region_atmos_rmses["Temperature at 850hPa"] = base_region_atmospheric_rmses['t'][10]
# q 700
base_region_atmos_rmses["Specific humidity at 700hPa"] = base_region_atmospheric_rmses['q'][9]
# u 850
base_region_atmos_rmses["Eastward wind speed at 850hPa"] = base_region_atmospheric_rmses['u'][10]
# v 850
base_region_atmos_rmses["Southward wind speed at 850hPa"] = base_region_atmospheric_rmses['v'][10]



    
 # --- Global font settings ---
plt.rcParams.update({'font.size': 22})

# Define custom font sizes
label_fontsize = 22
tick_fontsize = 20
title_fontsize = 24.5

# --- Data setup ---
num_plots = len(target_region_atmos_rmses)
num_plots_per_rows = 5
num_rows = 1
variables = list(target_region_atmos_rmses.keys())

saving_path = "../report/evaluation/rmses_grid/pretrained_small/DLI/test"


# --- Figure and subplots ---
fig, axs = plt.subplots(num_rows, num_plots_per_rows, dpi=300, figsize=(40, 8))
axs = axs.ravel()

# Store handles and labels from the first plot for global legend
handles, labels = None, None

# --- Plot each variable ---
for i, ax in enumerate(axs[:num_plots]):
    line1, = ax.plot(lead_time, target_region_atmos_rmses[variables[i]], label="South Africa", c="brown")
    line2, = ax.plot(lead_time, base_region_atmos_rmses[variables[i]], label="USA", c="teal")
    line3, = ax.plot(lead_time, eu_region_atmos_rmses[variables[i]], label="Europe", c="navy")
    
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
fig.supylabel("RMSE", x=0.01, y=0.5, fontsize=label_fontsize)

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
plt.savefig(f"{saving_path}/sa_vs_usa_vs_eu.pdf", bbox_inches="tight")
plt.savefig(f"{saving_path}/sa_vs_usa_vs_eu.png", bbox_inches="tight")
plt.savefig(f"{saving_path}/sa_vs_usa_vs_eu.svg", bbox_inches="tight")
plt.close()

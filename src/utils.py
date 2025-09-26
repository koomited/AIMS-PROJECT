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
from functools import partial
from lora import LinearWithLoRA
from lora import create_custom_model

# Variables
ERA5_SURFACE_VARIABLES = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']

ERA5_ATMOSPHERIC_VARIABLES = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]

ERA5_STATIC_VARIABLES = ["land_sea_mask", "soil_type", "geopotential_at_surface"]



def min_max_scaled(data):
    data= np.array(data)
    
    data = (data-data.min())/(data.max()-data.min())
    return data

def get_surface_feature_target_data(feature_sliced_data, target_sliced_data, 
                                    surface_variables=ERA5_SURFACE_VARIABLES):
    # Select surface variables
    feature_surf_vars_ds = feature_sliced_data[surface_variables]
    target_surf_vars_ds = target_sliced_data[surface_variables]
    
    return feature_surf_vars_ds, target_surf_vars_ds


def get_atmos_feature_target_data(feature_sliced_data, target_sliced_data,
                                  atmos_variables=ERA5_ATMOSPHERIC_VARIABLES):
    # Select surface variables
    feature_atmos_vars_ds = feature_sliced_data[atmos_variables]
    target_atmos_vars_ds = target_sliced_data[atmos_variables]
    
    return feature_atmos_vars_ds, target_atmos_vars_ds

def get_static_feature_target_data(feature_sliced_data, target_sliced_data,
                                   static_variables=ERA5_STATIC_VARIABLES):
    # Select surface variables
    feature_static_vars_ds = feature_sliced_data[static_variables]
    target_static_vars_ds = target_sliced_data[static_variables]
    
    return feature_static_vars_ds, target_static_vars_ds




def create_batch(surf_vars_ds, atmos_vars_ds, static_vars_ds, i=1, target=True):

    surf_vars = {
        "2t": torch.from_numpy(surf_vars_ds["2m_temperature"].values[[i - 1, i]][None]),
        "10u": torch.from_numpy(surf_vars_ds["10m_u_component_of_wind"].values[[i - 1, i]][None]),
        "10v": torch.from_numpy(surf_vars_ds["10m_v_component_of_wind"].values[[i - 1, i]][None]),
        "msl": torch.from_numpy(surf_vars_ds["mean_sea_level_pressure"].values[[i - 1, i]][None]),
    }

    static_vars = {
        "z": torch.from_numpy(static_vars_ds["geopotential_at_surface"].values),
        "slt": torch.from_numpy(static_vars_ds["soil_type"].values),
        "lsm": torch.from_numpy(static_vars_ds["land_sea_mask"].values),
    }

    atmos_vars = {
        "t": torch.from_numpy(atmos_vars_ds["temperature"].values[[i - 1, i]][None]),
        "u": torch.from_numpy(atmos_vars_ds["u_component_of_wind"].values[[i - 1, i]][None]),
        "v": torch.from_numpy(atmos_vars_ds["v_component_of_wind"].values[[i - 1, i]][None]),
        "q": torch.from_numpy(atmos_vars_ds["specific_humidity"].values[[i - 1, i]][None]),
        "z": torch.from_numpy(atmos_vars_ds["geopotential"].values[[i - 1, i]][None]),
    }

    metadata=Metadata(
    lat=torch.from_numpy(surf_vars_ds.latitude.values),
    lon=torch.from_numpy(surf_vars_ds.longitude.values),
    time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[i],),
    atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values)
    )


    return Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)
####################################################################################################################""
# model
# model = AuroraSmall()
# model = AuroraSmall(
#     use_lora=False,  # Model was not fine-tuned.
#     autocast=True,  # Use AMP.
# )
# model = create_custom_model(model, lora_r = 8, lora_alpha = 16)

# model.load_state_dict(torch.load('../model/aurora.pth'))
# model.load_state_dict(torch.load('../model/best_models/best_model.pth'))
####################################################################################################################""


def predict_fn(model=None, batch=None, rollout_nums=2, device=None):
    model = model.to(device)
    model.eval()
    
    # batch = batch.to("cuda")
    with torch.inference_mode():
        preds = [pred for pred in rollout(model, batch, steps=rollout_nums)]
    return preds

def predict_train_fn(model=None, batch=None, rollout_nums=8, device="cuda"):
    model = model.to(device)
    model.train()
    preds = [pred for pred in rollout(model, batch, steps=rollout_nums)]
    return preds


# get weight for RMSE

def rmse_weights(latitudes, longitudes, 
                 R=6371.0, device="cuda"):
    """
    Compute area weights for RMSE calculation over a Europe grid.

    Parameters:
        latitudes (array-like): 1D array of latitudes (degrees).
        longitudes (array-like): 1D array of longitudes (degrees).
        R (float): Earth's radius in km (default: 6371.0).
        device (str): 'cpu' or 'cuda' for GPU acceleration.

    Returns:
        torch.Tensor: Area weights normalized to sum to 1.
    """
    # Convert lat/lon to radians
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    # Compute latitude and longitude differences
    dlat = np.abs(np.diff(lat_rad).mean())  # Average latitude difference
    dlon = np.abs(np.diff(lon_rad).mean())  # Average longitude difference

    # Calculate area weights
    areas = R**2 * dlon * np.abs(np.sin(lat_rad + dlat / 2) - np.sin(lat_rad - dlat / 2))

    # Expand areas to create a 2D area weight grid
    area_grid = np.outer(areas, np.ones(len(longitudes)))

  

    # Convert to PyTorch tensor
    return torch.tensor(area_grid, dtype=torch.float32, device=device)


def custom_rmse(actual, prediction, 
                weights, type="sum"):
    """
    Compute the weighted RMSE (Root Mean Square Error).

    Parameters:
        actual (torch.Tensor): Ground truth values.
        prediction (torch.Tensor): Predicted values.
        weights (torch.Tensor): Area weights (normalized).

    Returns:
        torch.Tensor: Weighted RMSE.
    """
    device = actual.device  # Get the device of `actual`
    prediction = prediction.to(device)
    weights = weights.to(device)


    # Compute weighted squared error
    if type=="sum":
        # print(actual.shape, prediction.shape, weights.shape)
        squared_error = ((actual - prediction) ** 2) * weights
        rmse = torch.sqrt(squared_error.sum()/weights.sum())
    else:
        squared_error = ((actual - prediction) ** 2*weights).mean() 
        rmse = torch.sqrt(squared_error/weights.mean())

    # Compute and return the sum of the squared errors (RMSE without sqrt)
    return rmse

def rmse_fn(predictions=None, 
            target_batch=None, var_name=None,
            weigths=None, var_type="surface",
            atmos_level_idx=0, area="world",
            device="cuda"):
    two_steps_rmse = []
    pred_dates = []
    for i in range(len(predictions)):
        pred = predictions[i]
        if var_type=="surface":
            prediction = pred.surf_vars[var_name][0, 0]#.numpy()
            if area=="world":
                # actual = target_batch.surf_vars[var_name].squeeze()[i,:,:][1:, :]
                actual = target_batch.surf_vars[var_name].squeeze()[i,:,:]
                
                
            else:
                actual = target_batch.surf_vars[var_name].squeeze()[i,:,:]
            # actual = target_batch.surf_vars[var_name][0, 0].numpy()
            actual = actual.to(device)
            # print(f"prediction {prediction}")
            # print(f"actual {actual}")
            
            # rmse = root_mean_squared_error(actual.flatten(), prediction.flatten())
            rmse_ = custom_rmse(actual, prediction, weigths)
            # print(rmse1)
            two_steps_rmse.append(rmse_.item())
            pred_dates.append(pred.metadata.time[0])
        # Atmospherique variable
        elif var_type=="atmosphere":
            prediction = pred.atmos_vars[var_name].squeeze()[atmos_level_idx,:,:].squeeze()
            # actual = target_batch.atmos_vars[var_name].squeeze()[i,:,:][1:, :]
            if area=="world":
                # actual = target_batch.atmos_vars[var_name].squeeze()[i,atmos_level_idx,:,:][:-1,:]
                actual = target_batch.atmos_vars[var_name].squeeze()[i,atmos_level_idx,:,:]
            else:
                actual = target_batch.atmos_vars[var_name].squeeze()[i,atmos_level_idx,:,:]
            
            # rmse = root_mean_squared_error(actual.flatten(), prediction.flatten())
            rmse_ = custom_rmse(actual, prediction, weigths)
            two_steps_rmse.append(rmse_.item())
            pred_dates.append(pred.metadata.time[0])
    return two_steps_rmse, pred_dates



def plot_rmses(variable, rmses_world, rmses_sa, 
               figsize=(12, 8), fontsize=35,
               date_ranges=None, 
               title=None,
               save_path="../report/rmses_world_SA",
               place = None,
               atmos_level=None):

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Extract dates
    date_times_6_hours = [date1 for date1, date2 in date_ranges]
    date_times_12_hours = [date2 for date1, date2 in date_ranges]
    formatted_dates_6_hours = [dt.strftime('%Y-%m-%d (%H:%M)') for dt in date_times_6_hours]
    formatted_dates_12_hours = [dt.strftime('%Y-%m-%d (%H:%M)') for dt in date_times_12_hours]

    # Convert x-axis to indices
    x_indices = np.arange(len(formatted_dates_6_hours))

    # Select a subset of dates for x-axis labels
    num_ticks = min(6, len(formatted_dates_6_hours))
    tick_positions = np.linspace(0, len(formatted_dates_6_hours) - 1, num_ticks, dtype=int)

    # Plot RMSEs
    ax.plot(x_indices, np.array(rmses_world)[:, 0], label=f"{place} RMSE (6h Forecast)", color="blue", linestyle="-", linewidth=2)
    ax.plot(x_indices, np.array(rmses_sa)[:, 0], label="South Africa RMSE (6h Forecast)", color="orange", linestyle="-", linewidth=2)
    ax.plot(x_indices, np.array(rmses_world)[:, 1], label=f"{place} RMSE (12h Forecast)", color="blue", linestyle="--", linewidth=2)
    ax.plot(x_indices, np.array(rmses_sa)[:, 1], label="South Africa RMSE (12h Forecast)", color="orange", linestyle="--", linewidth=2)

    # Set selected x-ticks and make horizontal
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([formatted_dates_12_hours[i] for i in tick_positions], fontsize=fontsize-4, rotation=30, ha='right')
    

    # Legend
    ax.legend(title="Forecast Horizon", title_fontsize=fontsize-2, fontsize=fontsize-4,
              bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    # Axis labels and title
    ax.set_xlabel("Forecast Date", fontsize=fontsize-2)
    ax.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=fontsize-2)
    # ax.set_title(title, fontsize=fontsize, pad=20)

    # Save plots
    if atmos_level:
        plt.savefig(f"{save_path}/rmse-{variable}-{atmos_level}.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}/rmse-{variable}-{atmos_level}.png", bbox_inches="tight")
        plt.savefig(f"{save_path}/rmse-{variable}-{atmos_level}.svg", bbox_inches="tight")
    else:
        plt.savefig(f"{save_path}/rmse-{variable}.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}/rmse-{variable}.png", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}/rmse-{variable}.svg", bbox_inches="tight")

    plt.show()


###################################### HRES T0 part##########################################


def _prepare(x: np.ndarray, i = 1) -> torch.Tensor:
    """Prepare a variable.

    This does the following things:
    * Select time indices `i` and `i - 1`.
    * Insert an empty batch dimension with `[None]`.
    * Flip along the latitude axis to ensure that the latitudes are decreasing.
    * Copy the data, because the data must be contiguous when converting to PyTorch.
    * Convert to PyTorch.
    """
    return torch.from_numpy(x[[i - 1, i]][None][..., ::-1, :].copy())

def create_hrest0_batch(surf_vars_ds, atmos_vars_ds, static_vars_ds, i=1):
    batch = Batch(
    surf_vars={
        "2t": _prepare(surf_vars_ds["2m_temperature"].values, i),
        "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values, i),
        "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values, i),
        "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values, i),
    },
    static_vars = {
            "z": torch.from_numpy(static_vars_ds["geopotential_at_surface"].values),
            "slt": torch.from_numpy(static_vars_ds["soil_type"].values),
            "lsm": torch.from_numpy(static_vars_ds["land_sea_mask"].values),
        },
    atmos_vars={
        "t": _prepare(atmos_vars_ds["temperature"].values,i),
        "u": _prepare(atmos_vars_ds["u_component_of_wind"].values, i),
        "v": _prepare(atmos_vars_ds["v_component_of_wind"].values, i),
        "q": _prepare(atmos_vars_ds["specific_humidity"].values, i),
        "z": _prepare(atmos_vars_ds["geopotential"].values, i),
    },
    metadata=Metadata(
        # Flip the latitudes! We need to copy because converting to PyTorch, because the
        # data must be contiguous.
        lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element.
        time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[i],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
    ),
)

    return batch
    
    
import pandas as pd

def select_climatology(clim_ds, target_time, var):
    """
    Selects the 6-hourly climatology for a given target timestamp.
    
    Parameters:
        clim_ds : xarray.Dataset
            Climatology dataset with dims (dayofyear, hour, lat, lon, level)
        target_time : np.datetime64 or pd.Timestamp
            Timestamp of the forecast target
    
    Returns:
        xarray.DataArray
            Climatology at the same lat/lon/level as the target
    """
    # Convert target time to pandas timestamp
    ts = pd.Timestamp(target_time)
    
    # Day of year (1-366)
    day_of_year = ts.dayofyear
    
    # Hour (0, 6, 12, 18)
    hour = (ts.hour // 6) * 6
    
    # Select from climatology
    clim_selected = clim_ds.sel(dayofyear=day_of_year, hour=hour)
    
    return clim_selected[var].values

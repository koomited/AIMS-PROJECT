import xarray as xr
import torch
import os
import logging
import sys
import torch.optim as optim
from datetime import datetime
import numpy as np

from aurora import AuroraSmall, Batch, Metadata, rollout
from torch.utils.data import Dataset
from loss import AuroraLoss
from utils import (
    get_surface_feature_target_data,
    get_atmos_feature_target_data,
    get_static_feature_target_data,
    create_batch,
    create_hrest0_batch,
    predict_train_fn,
    predict_fn, select_climatology
)
from evaluation_metric import evaluation_rmse, evaluation_acc
log_file = "big_modelvs_finetuned_model.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


VARIABLE_CORRESPONDANCY = {
    # Surface variables
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",

    # Atmospheric variables
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind"
}

#
SURFACE_WEIGHTS = {"2t":[], "10u":[], "10v":[], "msl":[]} 

SURFACE_VARIABLES =["2t", "10u", "10v", "msl"] 


ATMOSPHERIC_VARIABLES = ["z", "q", "t", "u", "v"]


surface_rmses_fine_tuned = {var:np.zeros(8) for var in SURFACE_VARIABLES}
surface_acc_fine_tuned = {var:np.zeros(8) for var in SURFACE_VARIABLES}
surface_rmses_non_fine_tuned = {var:np.zeros(8) for var in SURFACE_VARIABLES}
surface_acc_non_fine_tuned = {var:np.zeros(8) for var in SURFACE_VARIABLES}

atmospheric_rmses_fine_tuned  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}
atmospheric_acc_fine_tuned  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}
atmospheric_rmses_non_fine_tuned  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}
atmospheric_acc_non_fine_tuned  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}


def evaluation_rmse_acc(model_fine_tuned,
               model_non_fine_tuned,
               era5_data=None, 
               hres_data=None,
               climatology=None,
             rollouts_num=8,
             device = "cuda"):
    
    model_fine_tuned.to(device)
    model_non_fine_tuned.to(device)
    
    selected_times = hres_data.time.values  # Extract time values as NumPy array (faster access)
    num_samples = len(selected_times) - rollouts_num - 2
    loss_list = []

    model_fine_tuned.to(device)  # Move model to GPU if available
    model_non_fine_tuned.to(device)  # Move model to GPU if available

    counter = 0

    for i in range(num_samples):
        counter+=1
        t0, t1, t2, t3 = selected_times[i], selected_times[i+1], selected_times[i+2], selected_times[i+10]

        # Load required time slices once 
        sa_feature_hrest0_data = hres_data.sel(time=slice(t0, t1))
        sa_feature_era5_data = era5_data.sel(time=slice(t0, t1))
        sa_target_era5_data = era5_data.sel(time=slice(t0, t1))
        
        sa_targets_hrest0_data = hres_data.sel(time=slice(t2, t3))
        

        for i in range(rollouts_num):
            # get target time
            target_times = sa_targets_hrest0_data.time
            # get only two data for target
            lead_time_target = sa_targets_hrest0_data.sel(time=slice(target_times[i], target_times[i+1]))
            ######################### climatology data
            
            target_time = lead_time_target.time.values[0]
            
            # Extract features and targets
            sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_hrest0_data, lead_time_target)
            sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_hrest0_data, lead_time_target)
            sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_era5_data, sa_target_era5_data)

            
            # Create input and target batches
            input_batch = create_hrest0_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data).to(device)
            target_batch = create_hrest0_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data).to(device)

            # prediction from the fine tuned and non fine tuned model
            predictions_fine_tuned_model =  predict_fn(model=model_fine_tuned, batch=input_batch, rollout_nums=rollouts_num, device=device)
            predictions_non_fine_tuned_model =  predict_fn(model=model_non_fine_tuned, batch=input_batch, rollout_nums=rollouts_num, device=device)
            
            # Compuete rmse for all lead times
            fine_tuned_prediction = predictions_fine_tuned_model[i]
            non_fine_tuned_prediction = predictions_non_fine_tuned_model[i]
            
            for surf_var in SURFACE_VARIABLES:
                climatology_data = select_climatology(climatology, target_time,
                                                      VARIABLE_CORRESPONDANCY[surf_var])
                
                # for fine tuned model
                pred_tensor_ft = fine_tuned_prediction.surf_vars[surf_var].squeeze()
                pred_tensor_ft = pred_tensor_ft.to(device)
                
                
                # for non fine tuned
                pred_tensor_nft = non_fine_tuned_prediction.surf_vars[surf_var].squeeze()
                pred_tensor_nft = pred_tensor_nft.to(device)
                
                target_tensor = target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                # print(target_batch.surf_vars[surf_var].squeeze().shape)
                target_tensor = target_tensor.to(device)
                
                
              
                
                
                
                
                # Rmses
                surface_rmses_fine_tuned[surf_var][i]+= evaluation_rmse(
                                                    target_tensor, 
                                                    pred_tensor_ft, 
                                                    torch.tensor(hres_data.latitude.values),
                                                    torch.tensor(hres_data.longitude.values)  ,
                                                     device = device 
                                                ) 
                surface_rmses_non_fine_tuned[surf_var][i]+= evaluation_rmse(
                                                    target_tensor, 
                                                    pred_tensor_nft, 
                                                    torch.tensor(hres_data.latitude.values),
                                                    torch.tensor(hres_data.longitude.values),
                                                    device = device  
                                                ) 
                # ACC
                surface_acc_fine_tuned[surf_var][i]+=evaluation_acc(target_tensor, 
                                    pred_tensor_ft, 
                                    climatology_data,
                                    torch.tensor(hres_data.latitude.values),
                                    torch.tensor(hres_data.longitude.values) ).item()
                            
                surface_acc_non_fine_tuned[surf_var][i]+=evaluation_acc(target_tensor, 
                                    pred_tensor_nft, 
                                    climatology_data,
                                    torch.tensor(hres_data.latitude.values),
                                    torch.tensor(hres_data.longitude.values) ).item()
                
                
            ## Atmospheric
            atmos_levels_num = atmospheric_rmses_fine_tuned["z"].shape[0]
            for c in range(atmos_levels_num):
                for atmos_var in ATMOSPHERIC_VARIABLES:
                    climatology_data = select_climatology(climatology, target_time,
                                                      VARIABLE_CORRESPONDANCY[atmos_var])[c,:,:]
                    #fine tuned model
                    pred_tensor_ft = fine_tuned_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    pred_tensor_ft = pred_tensor_ft.to(device)
                    
                    # Non fine tuned model
                    pred_tensor_nft = non_fine_tuned_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    pred_tensor_nft = pred_tensor_nft.to(device)
                    
                    
                    target_tensor = target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    # print(target_batch.atmos_vars[atmos_var].squeeze().shape)
                    target_tensor = target_tensor.to(device)
                    
                    
                    
                    #Atmospheric rmses
                    atmospheric_rmses_fine_tuned[atmos_var][c, i] += evaluation_rmse(
                                                    target_tensor, 
                                                    pred_tensor_ft, 
                                                    torch.tensor(hres_data.latitude.values),
                                                    torch.tensor(hres_data.longitude.values),
                                                    device = device  
                                                ) 
                    atmospheric_rmses_non_fine_tuned[atmos_var][c, i]+= evaluation_rmse(
                                                    target_tensor, 
                                                    pred_tensor_nft, 
                                                    torch.tensor(hres_data.latitude.values),
                                                    torch.tensor(hres_data.longitude.values),
                                                    device = device   
                                                ) 
                    # ACC
                    atmospheric_acc_fine_tuned[atmos_var][c, i] += evaluation_acc(target_tensor, 
                                    pred_tensor_ft, 
                                    climatology_data,
                                    torch.tensor(hres_data.latitude.values),
                                    torch.tensor(hres_data.longitude.values) ).item()
                    atmospheric_acc_non_fine_tuned[atmos_var][c, i] += evaluation_acc(target_tensor, 
                                    pred_tensor_nft, 
                                    climatology_data,
                                    torch.tensor(hres_data.latitude.values),
                                    torch.tensor(hres_data.longitude.values) ).item()
        if not counter%10:
            logger.info(f"Iteration {counter} done")
    logger.info("Evaluation completed")
    
    return {
        'counter': counter,
        'surface_rmses_fine_tuned': surface_rmses_fine_tuned,
        'surface_acc_fine_tuned': surface_acc_fine_tuned,
        'atmospheric_rmses_fine_tuned': atmospheric_rmses_fine_tuned,
        'atmospheric_acc_fine_tuned': atmospheric_acc_fine_tuned,
        'surface_rmses_non_fine_tuned': surface_rmses_non_fine_tuned,
        'surface_acc_non_fine_tuned': surface_acc_non_fine_tuned,
        'atmospheric_rmses_non_fine_tuned': atmospheric_rmses_non_fine_tuned,
        'atmospheric_acc_non_fine_tuned': atmospheric_acc_non_fine_tuned
    }




def evaluation(model_fine_tuned,
               model_non_fine_tuned,
               era5_data=None, 
               hres_data=None,
             rollouts_num=8,
             device = "cuda"):
    
    model_fine_tuned.to(device)
    model_non_fine_tuned.to(device)
    
    selected_times = hres_data.time.values  # Extract time values as NumPy array (faster access)
    num_samples = len(selected_times) - rollouts_num - 2
    loss_list = []
    print(hres_data.latitude.values)

    model_fine_tuned.to(device)  # Move model to GPU if available
    model_non_fine_tuned.to(device)  # Move model to GPU if available

    counter = 0

    for i in range(num_samples):
        counter+=1
        t0, t1, t2, t3 = selected_times[i], selected_times[i+1], selected_times[i+2], selected_times[i+10]

        # Load required time slices once 
        sa_feature_hrest0_data = hres_data.sel(time=slice(t0, t1))
        sa_feature_era5_data = era5_data.sel(time=slice(t0, t1))
        sa_target_era5_data = era5_data.sel(time=slice(t0, t1))
        
        sa_targets_hrest0_data = hres_data.sel(time=slice(t2, t3))
        

        for i in range(rollouts_num):
            # get target time
            target_times = sa_targets_hrest0_data.time
            # get only two data for target
            lead_time_target = sa_targets_hrest0_data.sel(time=slice(target_times[i], target_times[i+1]))
            ######################### climatology data
            
            target_time = lead_time_target.time.values[0]
            
            # Extract features and targets
            sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_hrest0_data, lead_time_target)
            sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_hrest0_data, lead_time_target)
            sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_era5_data, sa_target_era5_data)

            
            # Create input and target batches
            input_batch = create_hrest0_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data).to(device)
            target_batch = create_hrest0_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data).to(device)

            # prediction from the fine tuned and non fine tuned model
            predictions_fine_tuned_model =  predict_fn(model=model_fine_tuned, batch=input_batch, rollout_nums=rollouts_num, device=device)
            predictions_non_fine_tuned_model =  predict_fn(model=model_non_fine_tuned, batch=input_batch, rollout_nums=rollouts_num, device=device)
            
            # Compuete rmse for all lead times
            fine_tuned_prediction = predictions_fine_tuned_model[i]
            non_fine_tuned_prediction = predictions_non_fine_tuned_model[i]
            
            for surf_var in SURFACE_VARIABLES:
               
                # for fine tuned model
                pred_tensor_ft = fine_tuned_prediction.surf_vars[surf_var].squeeze()
                pred_tensor_ft = pred_tensor_ft.to(device)
                
                
                # for non fine tuned
                pred_tensor_nft = non_fine_tuned_prediction.surf_vars[surf_var].squeeze()
                pred_tensor_nft = pred_tensor_nft.to(device)
                
                target_tensor = target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                # print(target_batch.surf_vars[surf_var].squeeze().shape)
                target_tensor = target_tensor.to(device)
                
                
              
                
                
                
                
                # Rmses
                surface_rmses_fine_tuned[surf_var][i]+= evaluation_rmse(
                                                    target_tensor, 
                                                    pred_tensor_ft, 
                                                    torch.tensor(hres_data.latitude.values),
                                                    torch.tensor(hres_data.longitude.values)  ,
                                                     device = device 
                                                ) 
                surface_rmses_non_fine_tuned[surf_var][i]+= evaluation_rmse(
                                                    target_tensor, 
                                                    pred_tensor_nft, 
                                                    torch.tensor(hres_data.latitude.values),
                                                    torch.tensor(hres_data.longitude.values),
                                                    device = device  
                                                ) 

            ## Atmospheric
            atmos_levels_num = atmospheric_rmses_fine_tuned["z"].shape[0]
            for c in range(atmos_levels_num):
                for atmos_var in ATMOSPHERIC_VARIABLES:

                    #fine tuned model
                    pred_tensor_ft = fine_tuned_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    pred_tensor_ft = pred_tensor_ft.to(device)
                    
                    # Non fine tuned model
                    pred_tensor_nft = non_fine_tuned_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    pred_tensor_nft = pred_tensor_nft.to(device)
                    
                    
                    target_tensor = target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    # print(target_batch.atmos_vars[atmos_var].squeeze().shape)
                    target_tensor = target_tensor.to(device)
                    
                    
                    
                    #Atmospheric rmses
                    atmospheric_rmses_fine_tuned[atmos_var][c, i] += evaluation_rmse(
                                                    target_tensor, 
                                                    pred_tensor_ft, 
                                                    torch.tensor(hres_data.latitude.values),
                                                    torch.tensor(hres_data.longitude.values),
                                                    device = device  
                                                ) 
                    atmospheric_rmses_non_fine_tuned[atmos_var][c, i]+= evaluation_rmse(
                                                    target_tensor, 
                                                    pred_tensor_nft, 
                                                    torch.tensor(hres_data.latitude.values),
                                                    torch.tensor(hres_data.longitude.values),
                                                    device = device   
                                                ) 
                
        if not counter%10:
            logger.info(f"Iteration {counter} done")
    logger.info("Evaluation completed")
    
    return {
        'counter': counter,
        'surface_rmses_fine_tuned': surface_rmses_fine_tuned,
        'atmospheric_rmses_fine_tuned': atmospheric_rmses_fine_tuned,
        'surface_rmses_non_fine_tuned': surface_rmses_non_fine_tuned,
        'atmospheric_rmses_non_fine_tuned': atmospheric_rmses_non_fine_tuned,
    }

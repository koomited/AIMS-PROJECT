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
log_file = "evaluation_run_wampln_smalftvs_pretrained_sa.log"
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





def evaluation_between_regions_acc(
            model,
            era5_data=None, 
            hres_data=None,
            target_region=None,
            third_region=None,
            base_region= None,
            climatology_tr=None,
            climatology_br=None,
            climatology_third=None,
            rollouts_num=8,
            device = "cuda"):
    
    tr_surface_acc = {var:np.zeros(8) for var in SURFACE_VARIABLES}
    br_surface_acc = {var:np.zeros(8) for var in SURFACE_VARIABLES}
    third_surface_acc = {var:np.zeros(8) for var in SURFACE_VARIABLES}

    tr_atmospheric_acc = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}
    br_atmospheric_acc = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}
    third_atmospheric_acc = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}



    selected_times = hres_data.time.values  # Extract time values as NumPy array (faster access)
    num_samples = len(selected_times) - rollouts_num - 2


    model.to(device)  # Move model to GPU if available

    counter = 0
    # regions
    
    target_region_lat_min ,target_region_lat_max , target_region_lon_min, target_region_lon_max = target_region
    base_region_lat_min ,base_region_lat_max , base_region_lon_min, base_region_lon_max = base_region
    third_region_lat_min ,third_region_lat_max , third_region_lon_min, third_region_lon_max = third_region
    
 
    counter_tr = 0
    counter_br = 0
    counter_third = 0
    for i in range(num_samples):
        counter+=1
        t0, t1, t2, t3 = selected_times[i], selected_times[i+1], selected_times[i+2], selected_times[i+10]

        # target region data
        tr_feature_hres_data = hres_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_min, target_region_lat_max),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_target_hres_data = hres_data.sel(time=slice(t2, t3),
                              latitude=slice(target_region_lat_min, target_region_lat_max),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_feature_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_max, target_region_lat_min),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_target_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_max, target_region_lat_min),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        
        # get data base region
        
        br_feature_hres_data = hres_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_min, base_region_lat_max),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        br_target_hres_data = hres_data.sel(time=slice(t2, t3),
                              latitude=slice(base_region_lat_min, base_region_lat_max),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        br_feature_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_max, base_region_lat_min),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        br_target_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_max, base_region_lat_min),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        # third region
        third_feature_hres_data = hres_data.sel(time=slice(t0, t1),
                              latitude=slice(third_region_lat_min, third_region_lat_max),
                               longitude=slice(third_region_lon_min, third_region_lon_max))
        third_target_hres_data = hres_data.sel(time=slice(t2, t3),
                              latitude=slice(third_region_lat_min, third_region_lat_max),
                               longitude=slice(third_region_lon_min, third_region_lon_max))
        
        third_feature_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(third_region_lat_max, third_region_lat_min),
                               longitude=slice(third_region_lon_min, third_region_lon_max))
        
        third_target_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(third_region_lat_max, third_region_lat_min),
                               longitude=slice(third_region_lon_min, third_region_lon_max))
        
        
        
        target_times = tr_target_hres_data.time

        for i in range(rollouts_num):
            # get target time
            # get only two data for target
            # Lead time target
            tr_lead_time_target = tr_target_hres_data.sel(time=slice(target_times[i], target_times[i+1]))
            # Lead time target
            br_lead_time_target = br_target_hres_data.sel(time=slice(target_times[i], target_times[i+1]))
            third_lead_time_target = third_target_hres_data.sel(time=slice(target_times[i], target_times[i+1]))
            
            ## ACC
            target_time = tr_lead_time_target.time.values[0]
            
            
            # Extract features and targets for target region
            tr_feature_surface_data, tr_target_surface_data = get_surface_feature_target_data(tr_feature_hres_data, tr_lead_time_target)
            tr_feature_atmos_data, tr_target_atmos_data = get_atmos_feature_target_data(tr_feature_hres_data, tr_lead_time_target)
            tr_feature_static_data, tr_target_static_data = get_static_feature_target_data(tr_feature_era_data, tr_target_era_data)

            # Extract features and targets for base region
            br_feature_surface_data, br_target_surface_data = get_surface_feature_target_data(br_feature_hres_data, br_lead_time_target)
            br_feature_atmos_data, br_target_atmos_data = get_atmos_feature_target_data(br_feature_hres_data, br_lead_time_target)
            br_feature_static_data, br_target_static_data = get_static_feature_target_data(br_feature_era_data, br_target_era_data)

            # Extract features and targets for third region
            third_feature_surface_data, third_target_surface_data = get_surface_feature_target_data(third_feature_hres_data, third_lead_time_target)
            third_feature_atmos_data, third_target_atmos_data = get_atmos_feature_target_data(third_feature_hres_data, third_lead_time_target)
            third_feature_static_data, third_target_static_data = get_static_feature_target_data(third_feature_era_data, third_target_era_data)

            
            # Create input and target batches for target regio
            
            tr_input_batch = create_hrest0_batch(tr_feature_surface_data, tr_feature_atmos_data, tr_feature_static_data).to(device)
            tr_target_batch = create_hrest0_batch(tr_target_surface_data, tr_target_atmos_data, tr_target_static_data).to(device)

            # Create input and target batches for target regio
            br_input_batch = create_hrest0_batch(br_feature_surface_data, br_feature_atmos_data, br_feature_static_data).to(device)
            br_target_batch = create_hrest0_batch(br_target_surface_data, br_target_atmos_data, br_target_static_data).to(device)

            # Create input and target batches for third regio
            third_input_batch = create_hrest0_batch(third_feature_surface_data, third_feature_atmos_data, third_feature_static_data).to(device)
            third_target_batch = create_hrest0_batch(third_target_surface_data, third_target_atmos_data, third_target_static_data).to(device)

            
            # predictions
            tr_predictions =  predict_fn(model=model, batch=tr_input_batch, rollout_nums=rollouts_num, device=device)
            br_predictions =  predict_fn(model=model, batch=br_input_batch, rollout_nums=rollouts_num, device=device)
            third_predictions =  predict_fn(model=model, batch=third_input_batch, rollout_nums=rollouts_num, device=device)
            
            # Compuete rmse for all lead times
            tr_prediction = tr_predictions[i]
            br_prediction = br_predictions[i]
            third_prediction = third_predictions[i]
            
            for surf_var in SURFACE_VARIABLES:
                climatology_data = select_climatology(climatology_tr, target_time,
                                                      VARIABLE_CORRESPONDANCY[surf_var])
                # for target region
                tr_pred_tensor = tr_prediction.surf_vars[surf_var].squeeze()
                tr_pred_tensor = tr_pred_tensor.to(device)
                
                # for target region
                br_pred_tensor = br_prediction.surf_vars[surf_var].squeeze()
                br_pred_tensor = br_pred_tensor.to(device)
                
                # for third region
                third_pred_tensor = third_prediction.surf_vars[surf_var].squeeze()
                third_pred_tensor = third_pred_tensor.to(device)
              
                # target tensors
                tr_target_tensor = tr_target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                tr_target_tensor = tr_target_tensor.to(device)
                
                br_target_tensor = br_target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                br_target_tensor = br_target_tensor.to(device)
                
                third_target_tensor = third_target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                third_target_tensor = third_target_tensor.to(device)
                ######### Target region
               
                    
                 # ACC
                acc_tr=evaluation_acc(tr_target_tensor, 
                                    tr_pred_tensor, 
                                    climatology_data,
                                    torch.tensor(tr_feature_hres_data.latitude.values),
                                    torch.tensor(tr_feature_hres_data.longitude.values)  ).item()
                if acc_tr is not None and not np.isnan(acc_tr):
                    counter_tr+=1
                    
                    tr_surface_acc[surf_var][i]+= acc_tr
                
                ######### Base region
                
                climatology_data = select_climatology(climatology_br, target_time,
                                                      VARIABLE_CORRESPONDANCY[surf_var])
                
                
                acc_br = evaluation_acc(br_target_tensor, 
                                    br_pred_tensor, 
                                    climatology_data,
                                    torch.tensor(br_feature_hres_data.latitude.values),
                                    torch.tensor(br_feature_hres_data.longitude.values)  ).item()
                if acc_br is not None and not np.isnan(acc_br):
                    counter_br+=1
                    
                    br_surface_acc[surf_var][i]+= acc_br
                
                
                ## Third region
                
                climatology_data = select_climatology(climatology_third, target_time,
                                                      VARIABLE_CORRESPONDANCY[surf_var])
                
                acc_third = evaluation_acc(third_target_tensor, 
                                    third_pred_tensor, 
                                    climatology_data,
                                    torch.tensor(third_feature_hres_data.latitude.values),
                                    torch.tensor(third_feature_hres_data.longitude.values)   ).item()
                if acc_third is not None and not np.isnan(acc_third):
                    counter_third+=1
                    
                    third_surface_acc[surf_var][i]+= acc_third
            ## Atmospheric
            atmos_levels_num = tr_atmospheric_acc["z"].shape[0]
            for c in range(atmos_levels_num):
                for atmos_var in ATMOSPHERIC_VARIABLES:
                    climatology_data = select_climatology(climatology_tr, target_time,
                                                      VARIABLE_CORRESPONDANCY[atmos_var])[c,:,:]
                    #target region
                    tr_pred_tensor = tr_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    tr_pred_tensor = tr_pred_tensor.to(device)
                    
                    #target region
                    br_pred_tensor = br_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    br_pred_tensor = br_pred_tensor.to(device)
                    
                     #third region
                    third_pred_tensor = third_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    third_pred_tensor = third_pred_tensor.to(device)
                    
                    
                    tr_target_tensor = tr_target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    tr_target_tensor = tr_target_tensor.to(device)
                    
                    br_target_tensor = br_target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    br_target_tensor = br_target_tensor.to(device)
                    
                    third_target_tensor = third_target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    third_target_tensor = third_target_tensor.to(device)
                    
                    
                    #Atmospheric rms
                    ## TARGET REGION
                   
                    
                     # ACC
                    acc_tr=evaluation_acc(tr_target_tensor, 
                                        tr_pred_tensor, 
                                        climatology_data,
                                        torch.tensor(tr_feature_hres_data.latitude.values),
                                        torch.tensor(tr_feature_hres_data.longitude.values)  ).item()
                    if acc_tr is not None and not np.isnan(acc_tr):
                        tr_atmospheric_acc[atmos_var][c, i] += acc_tr
                        
                    
                    ## BASE REGION
                    climatology_data = select_climatology(climatology_br, target_time,
                                                      VARIABLE_CORRESPONDANCY[atmos_var])[c,:,:]
                    
                    
                    acc_br=evaluation_acc(br_target_tensor, 
                                        br_pred_tensor, 
                                        climatology_data,
                                        torch.tensor(br_feature_hres_data.latitude.values),
                                        torch.tensor(br_feature_hres_data.longitude.values) ).item()
                    if acc_br is not None and not np.isnan(acc_br):
                        br_atmospheric_acc[atmos_var][c, i] += acc_br
                    
                    ## THIRD REGION
                    climatology_data = select_climatology(climatology_third, target_time,
                                                      VARIABLE_CORRESPONDANCY[atmos_var])[c,:,:]
                    
                    
                    
                    acc_third=evaluation_acc(third_target_tensor, 
                                        third_pred_tensor, 
                                        climatology_data,
                                        torch.tensor(third_feature_hres_data.latitude.values),
                                        torch.tensor(third_feature_hres_data.longitude.values) ).item()
                    if acc_third is not None and not np.isnan(acc_third):
                        third_atmospheric_acc[atmos_var][c, i] += acc_third
                 
        if not counter%10:
            logger.info(f"Iteration {counter} done")
    logger.info("Evaluation completed")
    tr_surface_acc = {var:values/counter_tr for var, values in tr_surface_acc.items()}
    tr_atmospheric_acc = {var:values/counter_tr for var, values in tr_atmospheric_acc.items()}
    br_surface_acc = {var:values/counter_br for var, values in br_surface_acc.items()}
    br_atmospheric_acc = {var:values/counter_br for var, values in br_atmospheric_acc.items()}
    third_surface_acc = {var:values/counter_third for var, values in third_surface_acc.items()}
    third_atmospheric_acc = {var:values/counter_third for var, values in third_atmospheric_acc.items()}
    
    return {
        'counter': counter,
        'target_region_surface_acc': tr_surface_acc,
        'target_region_atmospheric_acc': tr_atmospheric_acc,
        'base_region_surface_acc': br_surface_acc,
        'base_region_atmospheric_acc': br_atmospheric_acc,
        'third_region_surface_acc': third_surface_acc,
        'third_region_atmospheric_acc': third_atmospheric_acc
    }



def evaluation_between_regions(
            model,
            era5_data=None, 
            hres_data=None,
            target_region=None,
            third_region=None,
            base_region= None,
            rollouts_num=8,
            device = "cuda"):
    
    tr_surface_rmses = {var:np.zeros(8) for var in SURFACE_VARIABLES}
    br_surface_rmses = {var:np.zeros(8) for var in SURFACE_VARIABLES}
    third_surface_rmses = {var:np.zeros(8) for var in SURFACE_VARIABLES}

    tr_atmospheric_rmses  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}
    br_atmospheric_rmses  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}
    third_atmospheric_rmses  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}



    selected_times = hres_data.time.values  # Extract time values as NumPy array (faster access)
    num_samples = len(selected_times) - rollouts_num - 2


    model.to(device)  # Move model to GPU if available

    counter = 0
    # regions
    
    target_region_lat_min ,target_region_lat_max , target_region_lon_min, target_region_lon_max = target_region
    base_region_lat_min ,base_region_lat_max , base_region_lon_min, base_region_lon_max = base_region
    third_region_lat_min ,third_region_lat_max , third_region_lon_min, third_region_lon_max = third_region
    
 
    counter_tr = 0
    counter_br = 0
    counter_third = 0
    for i in range(num_samples):
        counter+=1
        t0, t1, t2, t3 = selected_times[i], selected_times[i+1], selected_times[i+2], selected_times[i+10]

        # target region data
        tr_feature_hres_data = hres_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_min, target_region_lat_max),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_target_hres_data = hres_data.sel(time=slice(t2, t3),
                              latitude=slice(target_region_lat_min, target_region_lat_max),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_feature_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_max, target_region_lat_min),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_target_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_max, target_region_lat_min),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        
        # get data base region
        
        br_feature_hres_data = hres_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_min, base_region_lat_max),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        br_target_hres_data = hres_data.sel(time=slice(t2, t3),
                              latitude=slice(base_region_lat_min, base_region_lat_max),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        br_feature_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_max, base_region_lat_min),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        br_target_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_max, base_region_lat_min),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        # third region
        third_feature_hres_data = hres_data.sel(time=slice(t0, t1),
                              latitude=slice(third_region_lat_min, third_region_lat_max),
                               longitude=slice(third_region_lon_min, third_region_lon_max))
        third_target_hres_data = hres_data.sel(time=slice(t2, t3),
                              latitude=slice(third_region_lat_min, third_region_lat_max),
                               longitude=slice(third_region_lon_min, third_region_lon_max))
        
        third_feature_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(third_region_lat_max, third_region_lat_min),
                               longitude=slice(third_region_lon_min, third_region_lon_max))
        
        third_target_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(third_region_lat_max, third_region_lat_min),
                               longitude=slice(third_region_lon_min, third_region_lon_max))
        
        
        
        target_times = tr_target_hres_data.time

        for r in range(rollouts_num):
            # get target time
            # get only two data for target
            # Lead time target
            tr_lead_time_target = tr_target_hres_data.sel(time=slice(target_times[r], target_times[r+1]))
            # Lead time target
            br_lead_time_target = br_target_hres_data.sel(time=slice(target_times[r], target_times[r+1]))
            third_lead_time_target = third_target_hres_data.sel(time=slice(target_times[r], target_times[r+1]))
            
            ## ACC
            target_time = tr_lead_time_target.time.values[0]
            
            
            # Extract features and targets for target region
            tr_feature_surface_data, tr_target_surface_data = get_surface_feature_target_data(tr_feature_hres_data, tr_lead_time_target)
            tr_feature_atmos_data, tr_target_atmos_data = get_atmos_feature_target_data(tr_feature_hres_data, tr_lead_time_target)
            tr_feature_static_data, tr_target_static_data = get_static_feature_target_data(tr_feature_era_data, tr_target_era_data)
            
            
            #####################"DEBUG"
 
            # Extract features and targets for base region
            br_feature_surface_data, br_target_surface_data = get_surface_feature_target_data(br_feature_hres_data, br_lead_time_target)
            br_feature_atmos_data, br_target_atmos_data = get_atmos_feature_target_data(br_feature_hres_data, br_lead_time_target)
            br_feature_static_data, br_target_static_data = get_static_feature_target_data(br_feature_era_data, br_target_era_data)

            # Extract features and targets for third region
            third_feature_surface_data, third_target_surface_data = get_surface_feature_target_data(third_feature_hres_data, third_lead_time_target)
            third_feature_atmos_data, third_target_atmos_data = get_atmos_feature_target_data(third_feature_hres_data, third_lead_time_target)
            third_feature_static_data, third_target_static_data = get_static_feature_target_data(third_feature_era_data, third_target_era_data)

            
            # Create input and target batches for target regio
            
            tr_input_batch = create_hrest0_batch(tr_feature_surface_data, tr_feature_atmos_data, tr_feature_static_data).to(device)
            tr_target_batch = create_hrest0_batch(tr_target_surface_data, tr_target_atmos_data, tr_target_static_data).to(device)

            # Create input and target batches for target regio
            br_input_batch = create_hrest0_batch(br_feature_surface_data, br_feature_atmos_data, br_feature_static_data).to(device)
            br_target_batch = create_hrest0_batch(br_target_surface_data, br_target_atmos_data, br_target_static_data).to(device)

            # Create input and target batches for third regio
            third_input_batch = create_hrest0_batch(third_feature_surface_data, third_feature_atmos_data, third_feature_static_data).to(device)
            third_target_batch = create_hrest0_batch(third_target_surface_data, third_target_atmos_data, third_target_static_data).to(device)

            
            # predictions
            tr_predictions =  predict_fn(model=model, batch=tr_input_batch, rollout_nums=r+1, device=device)
            br_predictions =  predict_fn(model=model, batch=br_input_batch, rollout_nums=r+1, device=device)
            third_predictions =  predict_fn(model=model, batch=third_input_batch, rollout_nums=r+1, device=device)
            
            # Compuete rmse for all lead times
            tr_prediction = tr_predictions[r]
            br_prediction = br_predictions[r]
            third_prediction = third_predictions[r]
            
            for surf_var in SURFACE_VARIABLES:
                # for target region
                tr_pred_tensor = tr_prediction.surf_vars[surf_var].squeeze()
                tr_pred_tensor = tr_pred_tensor.to(device)
                
                # for target region
                br_pred_tensor = br_prediction.surf_vars[surf_var].squeeze()
                br_pred_tensor = br_pred_tensor.to(device)
                
                # for third region
                third_pred_tensor = third_prediction.surf_vars[surf_var].squeeze()
                third_pred_tensor = third_pred_tensor.to(device)
              
                # target tensors
                tr_target_tensor = tr_target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                tr_target_tensor = tr_target_tensor.to(device)
                
                br_target_tensor = br_target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                br_target_tensor = br_target_tensor.to(device)
                
                third_target_tensor = third_target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                third_target_tensor = third_target_tensor.to(device)
                ######### Target region
                # Rmses
                rmse_tr = evaluation_rmse(
                                            tr_target_tensor, 
                                            tr_pred_tensor, 
                                            torch.tensor(tr_feature_hres_data.latitude.values),
                                            torch.tensor(tr_feature_hres_data.longitude.values)   
                                        ) 
                if rmse_tr is not None and not np.isnan(rmse_tr.cpu()):
                    counter_tr+=1
                    tr_surface_rmses[surf_var][r]+= rmse_tr
                    
                
                ######### Base region
                
                rmse_br = evaluation_rmse(
                                                br_target_tensor, 
                                                br_pred_tensor, 
                                                torch.tensor(br_feature_hres_data.latitude.values),
                                                torch.tensor(br_feature_hres_data.longitude.values)   
                                            ) 
                if rmse_br is not None and not np.isnan(rmse_br.cpu()):
                    counter_br+=1
                
                    br_surface_rmses[surf_var][r]+= rmse_br
                
                
                
                ## Third region

                rmse_third =  evaluation_rmse(
                                            third_target_tensor, 
                                            third_pred_tensor, 
                                            torch.tensor(third_feature_hres_data.latitude.values),
                                            torch.tensor(third_feature_hres_data.longitude.values)   
                                        ) 
                if rmse_third is not None and not np.isnan(rmse_third.cpu()):
                    counter_third+=1
                
                    third_surface_rmses[surf_var][r]+= rmse_third
               
            ## Atmospheric
            atmos_levels_num = tr_atmospheric_rmses["z"].shape[0]
            for c in range(atmos_levels_num):
                for atmos_var in ATMOSPHERIC_VARIABLES:
                    
                    #target region
                    tr_pred_tensor = tr_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    tr_pred_tensor = tr_pred_tensor.to(device)
                    
                    #target region
                    br_pred_tensor = br_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    br_pred_tensor = br_pred_tensor.to(device)
                    
                     #third region
                    third_pred_tensor = third_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    third_pred_tensor = third_pred_tensor.to(device)
                    
                    
                    tr_target_tensor = tr_target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    tr_target_tensor = tr_target_tensor.to(device)
                    
                    br_target_tensor = br_target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    br_target_tensor = br_target_tensor.to(device)
                    
                    third_target_tensor = third_target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    third_target_tensor = third_target_tensor.to(device)
                    
                    
                    #Atmospheric rms
                    ## TARGET REGION
                    rmse_tr = evaluation_rmse(
                                                tr_target_tensor, 
                                                tr_pred_tensor, 
                                                torch.tensor(tr_feature_hres_data.latitude.values),
                                                torch.tensor(tr_feature_hres_data.longitude.values)    
                                            ) 
                    if rmse_tr is not None and not np.isnan(rmse_tr.cpu()):
                        tr_atmospheric_rmses[atmos_var][c, r] += rmse_tr
                    
                 
                    
                    ## BASE REGION
                    
                    rmse_br =  evaluation_rmse(
                                                br_target_tensor, 
                                                br_pred_tensor, 
                                                torch.tensor(br_feature_hres_data.latitude.values),
                                                torch.tensor(br_feature_hres_data.longitude.values)    
                                            ) 
                    if rmse_br is not None and not np.isnan(rmse_br.cpu()):
                        br_atmospheric_rmses[atmos_var][c, r] +=rmse_br
                    
                   
                    
                    ## THIRD REGION
                    
                    rmse_third = evaluation_rmse(
                                                    third_target_tensor, 
                                                    third_pred_tensor, 
                                                    torch.tensor(third_feature_hres_data.latitude.values),
                                                    torch.tensor(third_feature_hres_data.longitude.values)    
                                                ) 
                    if not rmse_third:
                        logger.info(f"Na values detected")
                    if rmse_third is not None and not np.isnan(rmse_third.item()):
                    
                        third_atmospheric_rmses[atmos_var][c, r] += rmse_third
                    
                   
                 
        if not counter%10:
            logger.info(f"Iteration {counter} done")
    logger.info("Evaluation completed")
    tr_surface_rmses = {var:values/counter_tr for var, values in tr_surface_rmses.items()}
    tr_atmospheric_rmses = {var:values/counter_tr for var, values in tr_atmospheric_rmses.items()}
    br_surface_rmses = {var:values/counter_br for var, values in br_surface_rmses.items()}
    br_atmospheric_rmses = {var:values/counter_br for var, values in br_atmospheric_rmses.items()}
    third_surface_rmses = {var:values/counter_third for var, values in third_surface_rmses.items()}
    third_atmospheric_rmses = {var:values/counter_third for var, values in third_atmospheric_rmses.items()}
    
    return {
        'counter': counter,
        'target_region_surface_rmses': tr_surface_rmses,
        'target_region_atmospheric_rmses': tr_atmospheric_rmses,
        'base_region_surface_rmses': br_surface_rmses,
        'base_region_atmospheric_rmses': br_atmospheric_rmses,
        'third_region_surface_rmses': third_surface_rmses,
        'third_region_atmospheric_rmses': third_atmospheric_rmses,
    }

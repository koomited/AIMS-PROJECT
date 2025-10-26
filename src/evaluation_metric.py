import torch

# def evaluation_rmse_weights(latitudes: torch.Tensor, device: str = "cuda") -> torch.Tensor:
#     """
#     Compute latitude-based weights according to Rasp et al. (2020).
#     w(i) = cos(lat(i)) / mean(cos(lat))
#     """
#     latitudes = torch.deg2rad(latitudes.to(device))
#     weights = torch.cos(latitudes)
#     weights = weights / weights.mean()  # normalize to unit mean
#     return weights


def evaluation_rmse_weights(latitudes: torch.Tensor, device: str = "cuda") -> torch.Tensor: 
    """ Compute latitude-based weighting factors for RMSE evaluation. 
    The weights are computed based on the cosine of the latitude values,
    normalized such that their sum equals the number of latitude points. 
    Parameters: 
    - latitudes (torch.Tensor): Tensor containing latitude values in degrees. 
    - device (str): Device to perform computation (default is "cuda"). 
    Returns: 
    - torch.Tensor: Normalized latitude-based weights. """
    latitudes = torch.deg2rad(latitudes.to(device)) # Convert degrees to radians 
    weights = torch.cos(latitudes) 
    weights = weights * len(latitudes) / torch.sum(weights)
    return weights

# def evaluation_rmse(
#     actual: torch.Tensor, 
#     prediction: torch.Tensor, 
#     latitudes: torch.Tensor,
#     longitudes: torch.Tensor,
#     device: str = "cuda"
# ) -> torch.Tensor:
#     """
#     Latitude-weighted RMSE as in Rasp et al. (2020).
#     """
#     actual, prediction = actual.to(device), prediction.to(device)
#     latitudes_weights = evaluation_rmse_weights(latitudes, device)

#     H, W = actual.shape
#     squared_error = (actual - prediction) ** 2

#     # Expand weights to 2D grid
#     area_grid_weights = torch.outer(latitudes_weights, torch.ones(len(longitudes), device=device))

#     # Weighted mean squared error
#     weighted_mse = torch.sum(area_grid_weights * squared_error) / (H * W)

#     # RMSE
#     rmse = torch.sqrt(weighted_mse)
#     return rmse

def evaluation_rmse( actual: torch.Tensor, prediction: torch.Tensor, 
                    latitudes: torch.Tensor, 
                    longitudes: torch.Tensor, 
                    device: str = "cuda" ) -> torch.Tensor: 
    """ Compute the root mean square error (RMSE) with latitude-based weighting. 
    The RMSE is computed over a 2D spatial grid, where latitude-dependent weights are 
    applied to account for the varying grid cell areas. Parameters: - actual (torch.Tensor): 
    Ground truth values (H x W). - prediction (torch.Tensor): Predicted values (H x W). 
    - latitudes (torch.Tensor): Latitude values in degrees (H, ). 
    - longitudes (torch.Tensor): Longitude values (W, ). 
    - device (str): Device to perform computation (default is "cuda"). 
    Returns: - torch.Tensor: Weighted RMSE value. """ 
    actual, prediction = actual.to(device), prediction.to(device) 
    latitudes_weights = evaluation_rmse_weights(latitudes, device) 
    H, W = actual.shape 
    squared_error = (actual - prediction) ** 2 
    # Create a 2D grid of weights 
    area_grid_weights = torch.outer(latitudes_weights, torch.ones(len(longitudes), device=device)) 
    # Compute weighted RMSE 
    rmse = torch.mean(torch.sqrt(area_grid_weights * squared_error/(H*W))) 
    return rmse


def evaluation_acc(
    actual: torch.Tensor, 
    prediction: torch.Tensor, 
    climatology: torch.Tensor,
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Latitude-weighted Anomaly Correlation Coefficient (ACC) as in Rasp et al. (2020).
    
    Parameters:
    - actual: Ground truth tensor [H, W]
    - prediction: Model prediction tensor [H, W]
    - climatology: Climatology tensor [H, W] (aligned with actual/prediction)
    - latitudes: Latitude values [H]
    - longitudes: Longitude values [W]
    - device: "cuda" or "cpu"
    
    Returns:
    - acc: Scalar ACC value
    """
    actual = torch.as_tensor(actual, device=device)
    prediction = torch.as_tensor(prediction, device=device)
    climatology = torch.as_tensor(climatology, device=device)

    latitudes_weights = evaluation_rmse_weights(latitudes, device)

    H, W = actual.shape

    # Compute anomalies
    actual_anom = actual - climatology
    pred_anom   = prediction - climatology

    # Expand weights to 2D grid
    area_grid_weights = torch.outer(latitudes_weights, torch.ones(len(longitudes), device=device))

    # Weighted numerator and denominator
    numerator = torch.sum(area_grid_weights * pred_anom * actual_anom)
    denominator = torch.sqrt(
        torch.sum(area_grid_weights * pred_anom**2) *
        torch.sum(area_grid_weights * actual_anom**2)
    )

    acc = numerator / denominator
    return acc

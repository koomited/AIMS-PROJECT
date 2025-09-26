import torch

def evaluation_rmse_weights(latitudes: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Compute latitude-based weights according to Rasp et al. (2020).
    w(i) = cos(lat(i)) / mean(cos(lat))
    """
    latitudes = torch.deg2rad(latitudes.to(device))
    weights = torch.cos(latitudes)
    weights = weights / weights.mean()  # normalize to unit mean
    return weights

def evaluation_rmse(
    actual: torch.Tensor, 
    prediction: torch.Tensor, 
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Latitude-weighted RMSE as in Rasp et al. (2020).
    """
    actual, prediction = actual.to(device), prediction.to(device)
    latitudes_weights = evaluation_rmse_weights(latitudes, device)

    H, W = actual.shape
    squared_error = (actual - prediction) ** 2

    # Expand weights to 2D grid
    area_grid_weights = torch.outer(latitudes_weights, torch.ones(len(longitudes), device=device))

    # Weighted mean squared error
    weighted_mse = torch.sum(area_grid_weights * squared_error) / (H * W)

    # RMSE
    rmse = torch.sqrt(weighted_mse)
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
    actual, prediction, climatology = (
        actual.to(device), prediction.to(device), climatology.to(device)
    )
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

from sklearn.ensemble import RandomForestRegressor
from utils.config import cfg

def get_rf_model():
    """
    Get a Random Forest model with the specified parameters.
    :return: A Random Forest model.
    """
    return RandomForestRegressor(
        n_estimators=cfg.RF_PARAMS['n_estimators'],
        max_depth=cfg.RF_PARAMS['max_depth'],
        n_jobs=cfg.RF_PARAMS['n_jobs'],
        random_state=cfg.RANDOM_SEED
    )

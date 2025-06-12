from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np

def calculate_metrics(y_true, y_pred):
    """compute MSE、Pearson、Spearman"""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'Pearson': pearsonr(y_true, y_pred)[0],
        'Spearman': spearmanr(y_true, y_pred)[0]
    }


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def pearson_corr(y_true, y_pred):
    """
    返回 y_true 与 y_pred 之间的 Pearson 相关系数
    """
    return pearsonr(y_true, y_pred)[0]

def spearman_corr(y_true, y_pred):
    """
    返回 y_true 与 y_pred 之间的 Spearman 相关系数
    """
    return spearmanr(y_true, y_pred)[0]

def all_metrics(y_true, y_pred):
    """
    同时返回多项指标：MSE、MAE、R^2、Pearson、Spearman
    以一个字典形式输出，方便打印或记录。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred)
    }
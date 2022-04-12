"""
Metrics which are useful to analyse experiments (or for hyperoptimization). They have the same signature as
sklearn metrics to make them easily interchangeable. New metrics can be added for custom needs.
Each metric as one of the following signature (y_pred and y_score are the same, but sklearn functions interchange
them, for different use cases):
- (y_true: pd.Series, y_pred: pd.DataFrame) -> float
- (y_true: pd.Series, y_score: pd.DataFrame) -> float
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.custom_metrics_and_plots.utils import (
    get_f1_and_thresholds_from_precision_recall_curve,
)


def rmse(y_true: pd.Series, y_pred: pd.DataFrame) -> float:
    """Computes the RMSE"""

    return mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)


def max_f1_from_precision_recall_curve(
    y_true: pd.Series, y_pred: pd.DataFrame
) -> float:
    """
    Only applicable for binary-classification.
    Gets the max f1 score, among all the f1 scores possible.

    Parameters
    ----------
    y_true: ground truth
    y_pred: model's predictions, with one column per class, corresponding the probability of each class.
    """

    f1_scores_pdf = get_f1_and_thresholds_from_precision_recall_curve(y_true, y_pred)
    return f1_scores_pdf["f1"].max()


def multiclass_roc_auc_score(y_true: pd.Series, y_score: pd.DataFrame) -> float:
    """
    Only applicable for multi-classification.
    Computes the ROC AUC score.
    """

    return roc_auc_score(y_true=y_true, y_score=y_score, multi_class="ovo")

""" Utils for the custom metrics/plots """

import pandas as pd
from sklearn.metrics import precision_recall_curve


def get_f1_and_thresholds_from_precision_recall_curve(
    y_true: pd.Series, y_pred: pd.DataFrame
) -> pd.DataFrame:
    """
    Only applicable for binary-classification.
    Computes the f1 score at different thresholds.

    Parameters
    ----------
    y_true: ground truth
    y_pred: model's predictions, with one column per class, corresponding the probability of each class.

    Returns
    -------
    A pandas Dataframe with one row per threshold. The columns correspond to the threshold and the f1 score
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true=y_true, probas_pred=y_pred
    )
    precision, recall = (
        precision[:-1],
        recall[:-1],
    )  # last element of precision and recall does not correspond to any value of thresholds
    f1_scores = 2 * recall * precision / (recall + precision)
    return pd.DataFrame(
        {"f1": f1_scores, "thresholds": thresholds}
    ).dropna()  # dropna to remove values where precision = recall = 0

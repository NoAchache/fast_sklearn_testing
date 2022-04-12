"""
Plots which are useful to analyse experiments. They are built the same way as sklearn classes for plots to
make them easily interchangeable. New plots can be added for custom needs.
Each class implements one of the following method (depending on the inputs needed):
- from_dataset_and_model(X: pd.DataFrame, model: ModelType) -> None
- from_model(model: ModelType) -> None
- from_predictions(y_true: pd.Series, y_pred: pd.DataFrame) -> None
"""

# pylint: disable=missing-class-docstring # Many classes with only one method and hence no need for class docstring

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay

from src.custom_metrics_and_plots.utils import (
    get_f1_and_thresholds_from_precision_recall_curve,
)
from src.utils.types import ModelType


class ShapValuesDisplay:
    @staticmethod
    def from_dataset_and_model(X: pd.DataFrame, model: ModelType) -> None:
        """
        Only applicable for models supported by the shap library.
        Builds a plot showing the shap score corresponding to each feature in X (i.e. how important is each feature
        depending on the prediction).
        """

        try:
            shap_values = shap.TreeExplainer(model).shap_values(X)
            shap.summary_plot(shap_values, X, show=False)
        except Exception as e:  # pylint: disable=broad-except # we catch Exception (and not a specific exception) as
            # it is the exception used in the shap libray

            print(
                f"Cannot compute shap values: {e}"
            )  # Raised if model not supported by the library shap.


class FeaturesImportanceDisplay:
    @staticmethod
    def from_model(model: ModelType) -> None:
        """
        Only applicable for tree based models.
        Builds a plot showing the importance of each feature in the model. The importance a given feature is usually
        found by counting the number of branches split made on this feature. The model must have the attribute
         feature_importances_, which is usually the case for tree based models.
        """
        if not hasattr(model, "feature_importances_") or not hasattr(
            model, "feature_names"
        ):
            print(
                "Model does not either have the attribute feature_importances_ or"
                " features_names. Hence cannot compute feature importances plot."
            )
            return

        feature_importances = pd.Series(
            model.feature_importances_, index=model.feature_names
        ).sort_values()

        feature_importances.plot.barh()


class ConfusionMatrixFromBestF1Display:
    @staticmethod
    def from_predictions(y_true: pd.Series, y_pred: pd.DataFrame) -> None:
        """
        Only applicable for binary-classification.
        Plots a confusion matrix from thresholded predictions (i.e. containing only 1s and 0s). The threshold used
        is one which gives the best f1 score.

        Parameters
        ----------
        y_true: ground truth
        y_pred: model's predictions, with only one column corresponding the probability that the class is 1
        """

        f1_scores_pdf = get_f1_and_thresholds_from_precision_recall_curve(
            y_true, y_pred
        )
        best_threshold = f1_scores_pdf["thresholds"].iloc[f1_scores_pdf["f1"].argmax()]
        y_pred_thresholded = (y_pred > best_threshold).astype(int)
        return ConfusionMatrixDisplay.from_predictions(
            y_true=y_true, y_pred=y_pred_thresholded
        )


class ConfusionMatrixMulticlassDisplay:
    @staticmethod
    def from_predictions(y_true: pd.Series, y_pred: pd.DataFrame) -> None:
        """
        Only applicable for multi-classification.
        Plots a confusion matrix from reformated predictions, such that it has only one column which corresponds to the
        predicted class.

        Parameters
        ----------
        y_true: ground truth
        y_pred: model's predictions, with one column per class, corresponding the probability of each class.
        """

        return ConfusionMatrixDisplay.from_predictions(
            y_true=y_true, y_pred=y_pred.idxmax(axis=1)
        )


class PredictionsAgainstGroundTruth:
    @staticmethod
    def from_predictions(y_true: pd.Series, y_pred: pd.DataFrame) -> None:
        """
        Useful for regression.
        Plots the predictions against the ground truth. It provides a graphical representation of the r2 score:
        - if r2 is 1 (best model possible), then all points lie on the green line
        - if r2 is close to 0, then all points lie far away from the green line and/or have a similar x-value (which
        corresponds to y_pred).
        """

        plt.figure()
        ax = plt.axes()
        ax.scatter(x=y_pred, y=y_true, s=0.5)

        ax.plot(
            [0, y_true.max()], [0, y_true.max()], linewidth=0.5, c="g"
        )  # plots the line for which x=y.

        ax.set_xlabel("y_pred")
        ax.set_ylabel("y_true")
        ax.axis("tight")

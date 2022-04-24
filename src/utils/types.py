"""Defines types corresponding to variables used through the project"""
# pylint: disable=missing-function-docstring # Many classes with one function so the docstring is
# redundant between both

from typing import Protocol, Union

import numpy as np
import pandas as pd


class FitMixin(Protocol):
    """Implements the method fit of an sklearn model"""

    def fit(self, X: pd.DataFrame, y_true: pd.Series) -> None:
        ...


class FeatureImportanceMixin(Protocol):
    """Implements feature importances related attributes"""

    feature_importances_: list[float]
    feature_names: list[str]


class RegressorModel(FitMixin, Protocol):
    """Sklearn Model used for regression's tasks"""

    feature_importances_: list[float]
    feature_names: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class ClassifierModel(FitMixin, Protocol):
    """Sklearn Model used for binary and multi classification's tasks"""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...


class ClassifierModelWithFeatureImportance(
    ClassifierModel, FeatureImportanceMixin, Protocol
):
    """Sklearn Model used for binary and multi classification's tasks which is applicable for feature importances"""


class RegressorModelWithFeatureImportance(
    RegressorModel, FeatureImportanceMixin, Protocol
):
    """Sklearn Model used for regression's tasks which is applicable for feature importances"""


ModelType = Union[
    ClassifierModel,
    RegressorModel,
    ClassifierModelWithFeatureImportance,
    RegressorModelWithFeatureImportance,
]


class PlotDisplayerFromPredictions(Protocol):
    """Class to display plots based on predictions results"""

    @staticmethod
    def from_predictions(y_true: pd.Series, y_pred: pd.DataFrame) -> None:
        ...


class PlotDisplayerFromDatasetAndModel(Protocol):
    """Class to display plots based on the features and the model"""

    @staticmethod
    def from_dataset_and_model(X: pd.DataFrame, model: ModelType) -> None:
        ...


class PlotDisplayerFromModel(Protocol):
    """Class to display plots based on the model"""

    @staticmethod
    def from_model(model: ModelType) -> None:
        ...


PlotDisplayerType = Union[
    PlotDisplayerFromPredictions,
    PlotDisplayerFromDatasetAndModel,
    PlotDisplayerFromModel,
]

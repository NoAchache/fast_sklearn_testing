""" Util functions for the project"""

import datetime
import inspect
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from dill.source import getname

from src.utils.types import ModelType, PlotDisplayerType


def load_config(config_path: str) -> dict:
    """
    Loads the training config in a dict, from a yaml file.

    Parameters
    ----------
    config_path: path of the yaml file

    Returns
    -------
    Config dict to parametrize the training pipeline
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = yaml.safe_load(config_file)
    return config_dict


def compute_metric(metric: Callable, y_true: pd.Series, y_pred: pd.DataFrame) -> float:
    """
    Computes a metric from the predictions and corresponding ground truth. Since sklearn metrics can have two
    different signatures, the metric method is inspected to call it correctly.

    Parameters
    ----------
    metric: metric to compute
    y_true: ground truth
    y_pred: predictions given by the model

    Returns
    -------
    Metric result.
    """

    arguments_of_metric = inspect.signature(metric).parameters.keys()
    if "y_pred" in arguments_of_metric:
        return metric(y_true=y_true, y_pred=y_pred)
    if "y_score" in arguments_of_metric:
        return metric(y_true=y_true, y_score=y_pred)
    raise TypeError("Unknown signature of metric")


def get_callable_from_name(
    callable_name: str, callables_list: list[Callable]
) -> Callable:
    """
    Retrieve the callable of a list from its name.

    Parameters
    ----------
    callable_name: name of the callable
    callables_list: list of callables, that must include the callable corresponding to the callable_name specified

    Returns
    -------
    The callable retrieved

    Raises
    -------
    ValueError: raised if there is no callable corresponding to callable_name
    """

    callables_mapping = {getname(callable_): callable_ for callable_ in callables_list}
    callable_matched = callables_mapping.get(callable_name, None)
    if callable_matched:
        return callable_matched
    raise ValueError(
        f"The callable name {callable_name} does not match with any of the following"
        f" callables: {list(callables_mapping.keys())}"
    )


def start_experiment() -> Path:
    """
    Creates a directory for a new experiment, named from the current time and date.

    Returns
    -------
    The path of the experiments directory
    """
    now = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    experiment_dir = Path(f"experiments/{now}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def plot_and_save(
    plot_displayer: PlotDisplayerType,
    model: ModelType,
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.DataFrame,
    experiment_dir: Path,
) -> None:
    """
    Builds a plot and saves it.

    Parameters
    ----------
    plot_displayer: class containing method to build the plot which is either, from_predictions. Must implement one
    of the following methods (depending on the inputs needed): from_predictions, from_model or from_dataset_and_model.
    X: features
    model: Trained model
    y_true: ground truth of X
    y_pred: predictions obtained for X with the model
    experiment_dir: directory in which the plots are saved
    """

    def check_method_exists(
        plot_displayer: PlotDisplayerType, method_name: str
    ) -> bool:
        """Checks the input class has the method: method_name"""
        method = getattr(plot_displayer, method_name, None)
        return bool(method)

    if check_method_exists(plot_displayer, "from_predictions"):
        plot_displayer.from_predictions(y_true=y_true, y_pred=y_pred)

    elif check_method_exists(plot_displayer, "from_dataset_and_model"):
        plot_displayer.from_dataset_and_model(X=X, model=model)

    elif check_method_exists(plot_displayer, "from_model"):
        plot_displayer.from_model(model=model)

    else:
        raise NotImplementedError(
            "None of the known method is implement for the plot diplayer"
            f" {getname(plot_displayer)}"
        )
    plt.savefig(experiment_dir / getname(plot_displayer), bbox_inches="tight")
    plt.close()

""" Entry point of the training process. Contains the method to run the training pipeline"""

from typing import Callable, Optional

import pandas as pd

from src.training.prepare_dataset import split_dataset
from src.training.training_steps import TrainingSteps
from src.utils.constants import CONFIG_PATH
from src.utils.utils import load_config, start_experiment

# pylint: disable=too-many-locals  # This function requires many locals to run correctly
def run_training_pipeline(
    dataset_with_targets: pd.DataFrame,
    config_path: str = CONFIG_PATH,
    shuffle_before_splitting: bool = True,
    optimization_dict_function: Optional[Callable] = None,
) -> None:
    """
    Runs the training pipeline:
    - Loads the yaml config to parametrize the training
    - Loads the dataset
    - Fit the model
    - If applicable, hyperoptimize
    - If applicable, cross-validate and infer the test and validation sets. Log the resulting metrics and graphs.

    Parameters$
    ----------
    dataset_with_targets: The dataset with one column per feature (including the target) and one row per observation
    config_path: Path to the yaml with the training parameters
    shuffle_before_splitting: Whether to shuffle the dataset before splitting into train, validation and test sets
    optimization_dict_function: Function returning a dict of hyperparameters to optimize, with the associated
    optimization ranges.
    """

    config = load_config(config_path)
    training_steps = TrainingSteps(config)
    experiment_dir = start_experiment()

    (
        (X_train, y_true_train),
        (X_validation, y_true_validation),
        (X_test, y_true_test),
    ) = split_dataset(dataset_with_targets, config, shuffle_before_splitting)

    if optimization_dict_function:
        training_steps.hyperoptimization(
            X_train, y_true_train, optimization_dict_function
        )

    training_steps.fit(X_train, y_true_train)

    if config["cross_validation"]["run"]:
        y_pred_cross_val = training_steps.cross_validate(X_train, y_true_train)
        training_steps.compute_and_save_metrics_and_plots(
            X_train,
            y_true_train,
            y_pred_cross_val,
            experiment_dir,
            "cross_validation",
        )

    if not X_validation.empty:
        y_pred_validation = training_steps.get_predictions(X_validation)
        training_steps.compute_and_save_metrics_and_plots(
            X_validation,
            y_true_validation,
            y_pred_validation,
            experiment_dir,
            "validation",
        )

    if not X_test.empty:
        y_pred_test = training_steps.get_predictions(X_test)
        training_steps.compute_and_save_metrics_and_plots(
            X_test,
            y_true_test,
            y_pred_test,
            experiment_dir,
            "test",
        )

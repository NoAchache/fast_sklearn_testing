""" Functions to prepare and split the dataset into the required subsets."""

from typing import Literal, Tuple

import numpy as np
import pandas as pd


def get_subsets(
    dataset: pd.DataFrame,
    dataset_split_proportions: dict[Literal["train", "validation", "test"], int],
    shuffle: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the input dataset into a train, a validation and a test set.

    Parameters
    ----------
    dataset: input dataset, to be split.
    dataset_split_proportions: dict mapping each subset name to the proportion of the dataset it should take. The sum
    of all proportions must be equal to 1.
    shuffle: Whether to shuffle the dataset before splitting it

    Returns
    -------
    Train, validation and test datasets

    Raises
    -------
    ValueError: raised if the sum of the split proportions of all subsets is not equal to 1, as the input
    dataset cannot be correctly split.
    """

    if np.sum(list(dataset_split_proportions.values())) != 1:
        raise ValueError(
            "The sum of the split proportions of all subset must be equal to 1"
        )

    dataset = dataset.sample(frac=1) if shuffle else dataset

    def proportion_to_row_number(
        subset_name: Literal["train", "validation", "test"]
    ) -> int:
        return int(dataset_split_proportions[subset_name] * len(dataset))

    train_set_last_row = proportion_to_row_number("train")
    validation_set_last_row = train_set_last_row + proportion_to_row_number(
        "validation"
    )
    test_set_last_row = validation_set_last_row + proportion_to_row_number("test")

    train_set = dataset.iloc[:train_set_last_row]
    validation_set = dataset.iloc[train_set_last_row:validation_set_last_row]
    test_set = dataset.iloc[validation_set_last_row:test_set_last_row]

    return train_set, validation_set, test_set


def split_features_and_target(
    dataset: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits the input dataset into a features df and a target series"""
    y_true = dataset[target_column]
    X = dataset.loc[:, dataset.columns != target_column]
    return X, y_true


def split_dataset(
    dataset: pd.DataFrame,
    config: dict,
    shuffle: bool,
) -> list[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    """Splits the input dataset into train, validation and test"""
    train_set, validation_set, test_set = get_subsets(
        dataset,
        config["dataset_split_proportions"],
        shuffle=shuffle,
    )
    return [
        split_features_and_target(subset, config["target_column"])
        for subset in [train_set, validation_set, test_set]
    ]

"""Define a CLI to create training_config.yml file from user inputs."""

from typing import Any, Callable, Literal, Union

import numpy as np
import typer
import yaml
from dill.source import getname
from pydantic import BaseModel, root_validator, validator

from src.config.models_metrics_plots_per_task import MODELS_METRICS_PLOTS_PER_TASK
from src.utils.constants import CONFIG_YAML_PATH


class SimpleField(BaseModel):
    """
    Field with no specific feature

    Parameters
    ----------
    yaml_key: key to which the user_input input should be associated to in the output yaml (i.e. {yaml_key:user_input})
    user_input: input entered by the user
    """

    yaml_key: str
    user_input: str


class MultipleChoiceField(SimpleField):
    """
    Field the user choose from a list of choices

    Parameters
    ----------
    choices: List of the possible elements from which the user should choose
    """

    choices: list[str]

    @root_validator
    def check_choice_in_choices(cls, values: dict) -> dict:
        """Asserts the user_input is within the list of choices"""

        if values["user_input"] not in values["choices"]:
            raise ValueError(
                "incorrect input. Enter (without quotes) an input in:"
                f" {values['choices']}"
            )
        return values


class NumericField(SimpleField):
    """
    Field for a numerical value

    Parameters
    ----------
    condition: Condition the user_input must meet
    retry_on_error_message: Message to display if the user_input is incorrect.
    """

    condition: Callable
    retry_on_error_message: str

    @root_validator()
    def check_condition(cls, values: dict) -> dict:
        """Asserts the user_input meets the condition"""

        condition = values["condition"]
        if not condition(values["user_input"]):
            raise ValueError(f"incorrect input. {values['retry_on_error_message']}")
        return values

    @staticmethod
    def generic_check_and_convert_type(values: dict, numeric_type: type) -> dict:
        """Converts the user_input (originally a string) to the target numeric_type if possible"""

        try:
            values["user_input"] = numeric_type(values["user_input"])
        except:
            raise ValueError(
                f"incorrect input. Cannot convert to {getname(numeric_type)}"
            )
        return values


class IntegerField(NumericField):
    """
    Field for an integer value. When instantiated, the user_input is converted to an int.

    Parameters
    ----------
    user_input: input entered by the user. Overwrite the user_input of SimpleField to use the correct type
    """

    user_input: Union[int, str]

    @root_validator(pre=True)
    def check_and_convert_type(cls, values: dict) -> dict:
        """Check if possible and converts the user_input (originally a string) to an int if possible"""
        return cls.generic_check_and_convert_type(values, numeric_type=int)


class FloatField(NumericField):
    """
    Field for a floating value. When instantiated, the user_input is converted to a float.

    Parameters
    ----------
    user_input: input entered by the user. Overwrite the user_input of SimpleField to use the correct type
    """

    user_input: Union[float, str]

    @root_validator(pre=True)
    def check_and_convert_type(cls, values: dict) -> dict:
        """Check if possible and converts the user_input (originally a string) to a float if possible"""
        return cls.generic_check_and_convert_type(values, numeric_type=float)


class BooleanField(SimpleField):
    """Field for a boolean value."""

    @validator("user_input")
    def check_choice_in_choices(cls, user_input: str) -> bool:
        """Checks the user input in within (y/n) and return the corresponding boolean"""
        if user_input == "y":
            return True
        if user_input == "n":
            return False
        raise ValueError(f"incorrect input. Enter an input in: [y, n]")


class DatasetSplitProportions(BaseModel):
    """
    Field containing the split proportions for the dataset in train, validation and test.

    Parameters
    ----------

    yaml_key: key to which the user_input input should be associated to in the output yaml (i.e. {yaml_key:user_input})
    user_input: dict containing values entered by the user, and the corresponding yaml_key as keys.
    """

    yaml_key: str
    user_input: dict[Literal["train", "validation", "test"], float]

    @root_validator
    def check_total_split(cls, values: dict) -> dict:
        """Asserts the sum of all subsplit is 1 (i.e. 100% of the dataset)"""
        total_split = np.sum(
            [split_proportion for split_proportion in values["user_input"].values()]
        )
        if total_split != 1:
            raise ValueError(
                f"The total of all split proportions must be 1. Got {total_split}"
            )
        return values


def ask_question_until_validated_answer(
    field_checker: Union[FloatField, IntegerField, MultipleChoiceField, BooleanField],
    prompt_message: str,
    **kwargs,
) -> Union[FloatField, IntegerField, MultipleChoiceField, BooleanField]:
    """
    Asks a question to the user and validates its answer by putting it in a pydantic model. The question is re-asked
    until the model is validated.

    Parameters
    ----------
    field_checker: Pydantic model to use to validate the answer
    prompt_message: Question to ask
    kwargs: kwargs for the field_checker

    Returns
    -------
    field_checker instantiated with the user input
    """

    answer = typer.prompt(prompt_message)
    try:
        return field_checker(**kwargs, user_input=answer)
    except ValueError as e:
        print(e)
        return ask_question_until_validated_answer(
            field_checker=field_checker, prompt_message=prompt_message, **kwargs
        )


def multiple_choice_question(
    yaml_key: str, prompt_message: str, possible_choices: list[str]
) -> MultipleChoiceField:
    """
    Ask a multiple choice question

    Parameters
    ----------
    yaml_key: key to which the user_input input should be associated to in the output yaml (i.e. {yaml_key:user_input})
    prompt_message: Question to ask
    possible_choices: List of the possible elements from which the user should choose

    Returns
    -------
    MultipleChoiceField instantiated with the user input
    """
    return ask_question_until_validated_answer(
        field_checker=MultipleChoiceField,
        prompt_message=prompt_message,
        yaml_key=yaml_key,
        choices=possible_choices,
    )


def boolean_question(yaml_key: str, prompt_message: str) -> BooleanField:
    """
    Ask a boolean question

    Parameters
    ----------
    yaml_key: key to which the user_input input should be associated to in the output yaml (i.e. {yaml_key:user_input})
    prompt_message: Question to ask

    Returns
    -------
    BooleanField instantiated with the user input
    """
    return ask_question_until_validated_answer(
        field_checker=BooleanField,
        prompt_message=f"{prompt_message} [y/n]",
        yaml_key=yaml_key,
    )


def numerical_question(
    yaml_key: str,
    prompt_message: str,
    condition: callable,
    retry_on_error_message: str,
    field: Union[FloatField, IntegerField],
) -> Union[FloatField, IntegerField]:
    """
    Ask a question where the user input is a number.

    Parameters
    ----------
    yaml_key: key to which the user_input input should be associated to in the output yaml (i.e. {yaml_key:user_input})
    prompt_message: Question to ask
    condition: Condition the user_input must meet
    retry_on_error_message: Message to display if the user_input is incorrect.

    Returns
    -------
    FloatField or IntegerField instantiated with the user input
    """
    return ask_question_until_validated_answer(
        field_checker=field,
        prompt_message=prompt_message,
        yaml_key=yaml_key,
        condition=condition,
        retry_on_error_message=retry_on_error_message,
    )


def get_dataset_split_proportions() -> DatasetSplitProportions:
    """
    Get the split proportions for the dataset in train, validation and test.

    Returns
    -------
    DatasetSplitProportions instantiated with the user inputs
    """
    print(
        "Enter the train, validation and test split proportion. The sum of the 3 values"
        " must 1"
    )
    dataset_split_proportions = {}
    for split_type in ["train", "validation", "test"]:
        split_proportion = numerical_question(
            yaml_key=split_type,
            prompt_message=split_type,
            condition=lambda x: 0 <= x <= 1,
            retry_on_error_message="Enter a number between 0 and 1",
            field=FloatField,
        )
        dataset_split_proportions[
            split_proportion.yaml_key
        ] = split_proportion.user_input
    try:
        return DatasetSplitProportions(
            yaml_key="dataset_split_proportions",
            user_input=dataset_split_proportions,
        )
    except ValueError as e:
        print(e)
        return get_dataset_split_proportions()


def transform_field_list_to_dict(
    field_list: list[
        Union[
            DatasetSplitProportions,
            FloatField,
            IntegerField,
            MultipleChoiceField,
            BooleanField,
        ]
    ]
) -> dict[str, Any]:
    """
    Transforms the input list of field to a dict mapping the attribute yaml_key of each field to the user_input
    """

    return {field.yaml_key: field.user_input for field in field_list}


def get_main_config_dict() -> dict[str, Any]:
    """Get the main config dict from user inputs"""

    tasks = list(MODELS_METRICS_PLOTS_PER_TASK.keys())
    task = multiple_choice_question(
        yaml_key="task",
        prompt_message=f"Choose a task: {tasks}",
        possible_choices=tasks,
    )
    models_of_task = [
        getname(model)
        for model in MODELS_METRICS_PLOTS_PER_TASK[task.user_input]["models"]
    ]
    model = multiple_choice_question(
        yaml_key="model",
        prompt_message=f"Choose a {task.user_input} model: {models_of_task}",
        possible_choices=models_of_task,
    )
    target_column = SimpleField(
        yaml_key="target_column",
        user_input=typer.prompt(
            "Enter the name of the target (ground truth) column of your dataset"
        ),
    )
    dataset_split_proportions = get_dataset_split_proportions()
    return transform_field_list_to_dict(
        [task, model, target_column, dataset_split_proportions]
    )


def get_cross_val_dict() -> dict[str, Any]:
    """Get the config dict related to cross validation from user inputs"""

    do_cross_val = boolean_question(
        yaml_key="run",
        prompt_message=f"Run cross-validation on test set?",
    )
    if do_cross_val.user_input:
        cross_val_folds_number = numerical_question(
            yaml_key="folds_number",
            prompt_message=f"Number of folds in crossval (usually 4-10)",
            condition=lambda x: x > 0,
            retry_on_error_message="Enter a positive integer",
            field=IntegerField,
        )
        return transform_field_list_to_dict([do_cross_val, cross_val_folds_number])
    return transform_field_list_to_dict([do_cross_val])


def get_hyperopt_dict(
    task: Literal["regression", "binary-classification", "multi-classification"]
) -> dict[str, Any]:
    """Get the config dict related to hyperopt from user inputs"""

    do_hyperopt = boolean_question(
        yaml_key="",
        prompt_message=(
            f"Do you to hyperoptimize your model? If yes, you will also need to provide"
            f" a function returning an optimization dict (Optuna is used) to"
            f" run_training_pipeline (c.f. examples section)"
        ),
    )
    if do_hyperopt.user_input:
        metrics_of_task = [
            getname(metric) for metric in MODELS_METRICS_PLOTS_PER_TASK[task]["metrics"]
        ]
        hyperopt_metric = multiple_choice_question(
            yaml_key="metric",
            prompt_message=f"Choose an hyperopt metric for {task}: {metrics_of_task}",
            possible_choices=metrics_of_task,
        )
        hyperopt_evaluation_number = numerical_question(
            yaml_key="evaluation_number",
            prompt_message=f"Number of evaluations in the hyperopt",
            condition=lambda x: x > 0,
            retry_on_error_message="Enter a positive integer",
            field=IntegerField,
        )
        metric_decreases_when_optimized = boolean_question(
            yaml_key="metric_decreases_when_optimized",
            prompt_message=f"Should the metric decrease when optimized?",
        )
        return transform_field_list_to_dict(
            [
                hyperopt_metric,
                hyperopt_evaluation_number,
                metric_decreases_when_optimized,
            ]
        )
    return {}


def config_file_generator():
    """Get all different configs from user inputs and save them in a yaml file"""
    print(
        "Entering the yaml generator. You can still modify your answers later on in the"
        " output yaml file"
    )
    training_config = get_main_config_dict()
    training_config["cross_validation"] = get_cross_val_dict()
    training_config["hyperoptimization"] = get_hyperopt_dict(training_config["task"])

    with open(CONFIG_YAML_PATH, "w") as f:
        yaml.dump(training_config, f)

    print(f"Sucess! The yaml file was created in {CONFIG_YAML_PATH}")


if __name__ == "__main__":
    typer.run(config_file_generator)

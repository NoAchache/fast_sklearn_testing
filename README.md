# Fast Sklearn ML Testing
Quickly test the performance of different models on your structured/labelled data for classification or regression.
- Try different models
- Cross validate, hyperoptimize and get performance on validation and test sets
- Different metrics/plots (mix of sklearn and custom metrics/plots) specific to each task (binary-classification, multi-classification, regression) 
- Easily add new models, metrics or plots.

## Installation

## Python 3.9.11
The projects uses Python 3.9.11. You can use [pyenv](https://github.com/pyenv/pyenv) to manage your Python versions.

## Install the dependencies

Install [poetry](https://python-poetry.org/docs/#installation):
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python - --version 1.1.13
```

Install data-science packages (example commands for mac, similar for linux):
```bash
brew install lightgbm
brew install cmake libomp
```

Install the dependencies:
```bash
poetry install
```

**Mac M1 warning:** If you are using a Mac M1, you will currently not be able to download the shap library. Remove it from
the pyproject.yaml and remove the import from the project.


## Get Started

### Try an example
Examples are provided for binary-classification, multi-classification and regression in /examples.

### Parametrize your training
There are two ways to do it:
- With the CLI, running:
```bash
poetry run python generate_config_file.py
```

- Manually, by directly filling in the file /src/config/training_config.yml

### Run your pipeline
Create a script which loads your data in a pandas dataframe and pass it to `run_training_pipeline`
located in `src.training.training_pipeline.run_training_pipeline.py`. C.f. /examples.

## Add new plots / metrics

### Sklearn plots / metrics
Add them to `MODELS_METRICS_PLOTS_PER_TASK` located in `src.config.models_metrics_plots_per_task.py`

### Custom plots / metrics
- Create the plot / metric logic in `src.custom_metrics_and_plots` in either `custom_metrics.py` and 
`custom_plots.py`.
- Add them to `MODELS_METRICS_PLOTS_PER_TASK` located in `src.config.models_metrics_plots_per_task.py`

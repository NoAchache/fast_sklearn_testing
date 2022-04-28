from sklearn.datasets import load_diabetes

from examples.optimization_dict_hyperopt import optimization_dict_function_for_lgbm
from src.training.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    diabetes = load_diabetes(as_frame=True)
    dataset_with_target = diabetes.frame.assign(target=diabetes.target)

    run_training_pipeline(
        dataset_with_target,
        config_path="examples/regression/training_config.yml",
        optimization_dict_function=optimization_dict_function_for_lgbm,
    )

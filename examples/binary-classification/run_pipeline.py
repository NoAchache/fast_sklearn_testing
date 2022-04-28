from sklearn.datasets import load_breast_cancer

from examples.optimization_dict_hyperopt import optimization_dict_function_for_lgbm
from src.training.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    breast_cancer = load_breast_cancer(as_frame=True)
    dataset_with_target = breast_cancer.frame.assign(target=breast_cancer.target)

    run_training_pipeline(
        dataset_with_target,
        config_path="examples/regression/training_config.yml",
        optimization_dict_function=optimization_dict_function_for_lgbm,
    )

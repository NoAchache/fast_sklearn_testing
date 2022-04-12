from optuna import Trial


def optimization_dict_function_for_lgbm(trial: Trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "random_state": 1,
    }

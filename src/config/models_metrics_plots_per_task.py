""" Define for each task (binary-classification, multi-classification or regression) the possible models, the name
of the infer method, and the metrics and plots. Add new models, metrics, or plots from the sklearn api if needed.
You can define custom metrics and plots in src.custom_metrics_and_plots"""

from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.custom_metrics_and_plots.custom_metrics import (
    max_f1_from_precision_recall_curve,
    multiclass_roc_auc_score,
    rmse,
)
from src.custom_metrics_and_plots.custom_plots import (
    ConfusionMatrixFromBestF1Display,
    ConfusionMatrixMulticlassDisplay,
    FeaturesImportanceDisplay,
    PredictionsAgainstGroundTruth,
    ShapValuesDisplay,
)

MODELS_METRICS_PLOTS_PER_TASK = (
    {  # For more metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
        "regression": {
            "models": [
                LGBMRegressor,
                RandomForestRegressor,
                LinearRegression,
                LogisticRegression,
            ],
            "prediction_method": "predict",
            "metrics": [r2_score, mean_squared_error, mean_absolute_error, rmse],
            "plots": [
                FeaturesImportanceDisplay,
                ShapValuesDisplay,
                PredictionsAgainstGroundTruth,
            ],
        },
        "binary-classification": {
            "models": [LGBMClassifier, RandomForestClassifier, LogisticRegression],
            "prediction_method": "predict_proba",
            "metrics": [
                roc_auc_score,
                max_f1_from_precision_recall_curve,
            ],
            "plots": [
                RocCurveDisplay,
                PrecisionRecallDisplay,
                ConfusionMatrixFromBestF1Display,
                FeaturesImportanceDisplay,
                ShapValuesDisplay,
            ],
        },
        "multi-classification": {
            "models": [LGBMClassifier, RandomForestClassifier, LogisticRegression],
            "prediction_method": "predict_proba",
            "metrics": [multiclass_roc_auc_score],
            "plots": [
                ConfusionMatrixMulticlassDisplay,
                FeaturesImportanceDisplay,
                ShapValuesDisplay,
            ],
        },
    }
)

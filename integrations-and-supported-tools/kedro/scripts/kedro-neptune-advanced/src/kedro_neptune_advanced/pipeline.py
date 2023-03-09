"""
This is a boilerplate pipeline
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_models,
    get_predictions,
    split_data,
    train_mlp_model,
    train_rf_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["example_iris_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            node(
                func=train_rf_model,
                inputs=["X_train", "y_train", "parameters"],
                outputs="rf_model",
                name="train_rf_model",
            ),
            node(
                func=train_mlp_model,
                inputs=["X_train", "y_train", "parameters"],
                outputs="mlp_model",
                name="train_mlp_model",
            ),
            node(
                func=get_predictions,
                inputs=["rf_model", "mlp_model", "X_test"],
                outputs="predictions",
                name="get_predictions",
            ),
            node(
                func=evaluate_models,
                inputs=["predictions", "y_test", "neptune_run"],
                outputs=None,
                name="evaluate_models",
            ),
        ]
    )

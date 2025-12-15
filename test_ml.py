import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Importing functions
from ml.data import process_data
from ml.model import train_model, compute_model_metrics


def test_process_data_shapes():
    """
    Test that process_data returns the correct shapes and types
    """
    data = pd.DataFrame(
        {
            "age": [25, 40],
            "workclass": ["Private", "Self-emp"],
            "education": ["Bachelors", "HS-grad"],
            "salary": [">50K", "<=50K"],
        }
    )

    cat_features = ["workclass", "education"]

    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    # Check the shape and type
    assert X.shape[0] == 2
    assert len(y) == 2
    assert isinstance(X, (pd.DataFrame, np.ndarray))
    assert isinstance(y, (pd.Series, np.ndarray))


def test_train_model_returns_estimator():
    """
    Test that train_model returns a RandomForestClassifier
    """
    X = np.random.rand(10, 5)
    y = np.array([0, 1] * 5)

    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_metrics_output():
    """
    Test that compute_model_metrics returns floats
    """
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

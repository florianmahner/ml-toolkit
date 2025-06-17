import numpy as np

from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Array = np.ndarray


def cross_validated_logistic_regression(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    **kwargs,
):
    kwargs.setdefault("cv", 5)
    kwargs.setdefault("max_iter", 1000)
    kwargs.setdefault("multi_class", "multinomial")
    kwargs.setdefault("Cs", 10)
    pipeline = make_pipeline(StandardScaler(), LogisticRegressionCV(**kwargs))
    clf = pipeline.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return clf, score


def cross_validated_ridge_regression(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    **kwargs,
):
    kwargs.setdefault("cv", 5)
    kwargs.setdefault("alphas", np.logspace(-6, 6, 5))
    pipeline = make_pipeline(StandardScaler(), RidgeCV(**kwargs))
    clf = pipeline.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return clf, score

import collections

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer
import pathlib
import sys
import pickle

from sklearn.model_selection import ParameterGrid

from tqdm import tqdm

sys.path.append(str(pathlib.Path("../..").resolve()))


from source.data import (
    create_splits,
    explode_multiple_locc,
    get_label_to_index_mapping,
)
from source.files import get_book_text, get_embedding
from source.metrics import calculate_flat_binary_metrics
from source.predict import predict
from source.models.xgbhi import (
    create_xgbhi_tfidf_model,
    create_xgbhi_multi_model,
)
from source.data import explode_locc


class _XGBoostHierarchicalModel(BaseEstimator):
    def _parent_label(self, label: str) -> str:
        for l in reversed(explode_locc(label)):
            if l == label:
                continue

            if l in self.labels:
                return l
        return None

    def __init__(
        self,
        labels: list[str],
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_jobs: int = -1,
    ):
        self.labels = sorted(labels)
        self.mlb = MultiLabelBinarizer(classes=self.labels)
        self.mlb.fit([self.labels])

        self.roots = sorted([l for l in self.labels if self._parent_label(l) is None])
        self.hierarchy = collections.defaultdict(list)
        for l in self.labels:
            if (parent := self._parent_label(l)) is not None:
                self.hierarchy[parent].append(l)
        self.to_root = {v: k for k, vs in self.hierarchy.items() for v in vs}

        self.root_mlb = MultiLabelBinarizer(classes=self.roots)
        self.root_mlb.fit([self.roots])
        self.root_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
        )

        self.mlbs = {}
        self.models = {}
        for k, v in self.hierarchy.items():
            self.mlbs[k] = MultiLabelBinarizer(classes=sorted([k, *v]))
            self.mlbs[k].fit([sorted([k, *v])])
            self.models[k] = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=n_jobs,
            )

        self.subproba_thresholds = 0.5

    def transform(self, y: pd.Series) -> pd.Series:
        return self.mlb.transform(y)

    def fit(self, X: list, y: pd.Series):
        y_root = self.root_mlb.transform(
            [[self.to_root.get(l, l) for l in ls] for ls in y]
        )
        self.root_model.fit(X, y_root)

        for k, v in self.hierarchy.items():
            ls = set([k, *v])
            y_k = self.mlbs[k].fit_transform([set(i).intersection(ls) for i in y])
            self.models[k].fit(X, y_k)

    def predict(self, X: list) -> pd.Series:
        y_root = self.root_model.predict(X)
        y_root = self.root_mlb.inverse_transform(y_root)

        y_ks = []
        for k, v in self.hierarchy.items():
            y_k = self.models[k].predict_proba(X)
            y_k = np.array([[y >= self.subproba_thresholds for y in ys] for ys in y_k])
            y_k = self.mlbs[k].inverse_transform(y_k)

            y_ks.append(y_k)

        y_root = [
            list(set([*y_root[i], *[y_k[i] for y_k in y_ks]]))
            for i in range(len(y_root))
        ]
        return pd.Series(y_root)


class XGBoostHierarchicalModel:
    def __init__(
        self,
        labels: list[str],
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_jobs: int = -1,
    ):
        self.__model = _XGBoostHierarchicalModel(
            labels=labels,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
        )

    def fit(self, X: list, y: pd.Series):
        self.__model.fit(X, y)

    def predict(self, X: list) -> pd.Series:
        pred = self.__model.predict(X)
        return [
            list({i for y in ys for i in (y if isinstance(y, tuple) else [y])})
            for ys in pred
        ]

    def transform(self, y: pd.Series) -> pd.Series:
        return self.__model.transform(y)


def create_xgbhi_multi_model(
    labels: list[str],
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_jobs: int = -1,
) -> XGBoostHierarchicalModel:
    return XGBoostHierarchicalModel(
        labels=labels,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    splits = create_splits(verbose=False)
    X_train, X_test, y_train, y_test = splits

    labels, _, _ = get_label_to_index_mapping(splits)

    best_params = {
        "learning_rate": 0.2,
        "max_depth": 5,
        "n_estimators": 3000,
    }

    X_train_texts = np.array(
        [get_embedding(str(num)) for num in tqdm(X_train["Etext Number"])]
    )
    X_test_texts = np.array(
        [get_embedding(str(num)) for num in tqdm(X_test["Etext Number"])]
    )

    model_name = "multi"
    model = create_xgbhi_multi_model(
        labels,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
    )

    model.fit(X_train_texts, [explode_multiple_locc(locc) for locc in y_train])

    pred = model.predict(X_test_texts)

    test_bin = model.transform([explode_multiple_locc(locc) for locc in y_test])
    pred_bin = model.transform(pred)

    f1score, name, metrics_path = calculate_flat_binary_metrics(
        test_bin,
        pred_bin,
        labels,
        "xgbhi",
        model_name,
        hyperparams=best_params,
        save=True,
    )

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(name)

import collections

import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from source.data import explode_locc


class SentenceEmbeddingWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        step, overlap = 100, 10
        book_encodings = []
        for book in tqdm.tqdm(X):
            parts = []

            tokens = book.split()
            for part in range(0, len(tokens), step):
                if tokens[part : min(part + step + overlap, len(tokens))]:
                    parts.append(
                        " ".join(tokens[part : min(part + step + overlap, len(tokens))])
                    )

            book_encodings.append(np.array(self.model.encode(parts)).mean(axis=0))

        return np.array(book_encodings)


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

        self.subproba_thresholds = 0.35

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


def create_xgbhi_tfidf_model(
    labels: list[str],
    n_estimators: int = 100,
    max_depth: int = 6,
    n_jobs: int = -1,
) -> _XGBoostHierarchicalModel:
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            (
                "model",
                _XGBoostHierarchicalModel(
                    labels=labels,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


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

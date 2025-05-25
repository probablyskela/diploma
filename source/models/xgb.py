import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


def create_tfidf_xgboost_model() -> tuple[Pipeline, dict]:
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            (
                "classifier",
                OneVsRestClassifier(
                    xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )


class SentenceTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_length=256,
        overlap=10,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.tokenizer = self.model.tokenizer

        self._num_special_tokens = len(
            self.tokenizer.build_inputs_with_special_tokens([])
        )

        model_max_content_tokens = self.model.max_seq_length - self._num_special_tokens

        self.content_target_length = min(
            max_chunk_length - overlap, model_max_content_tokens
        )

        self.content_target_length = max(0, self.content_target_length)
        self.overlap = max(0, overlap)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        all_chunk_texts_global = []
        doc_chunk_indices = [0]

        for text_document in tqdm(X, desc="Chunking documents"):
            current_doc_chunk_texts = []
            encoded_input = self.tokenizer(
                text_document,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            input_ids = encoded_input["input_ids"]

            start_index = 0

            if self.content_target_length <= 0:
                step = 1
            else:
                step = max(1, self.content_target_length - self.overlap)

            while start_index < len(input_ids):
                end_index = min(
                    start_index + self.content_target_length, len(input_ids)
                )
                current_chunk_input_ids = input_ids[start_index:end_index]

                if not current_chunk_input_ids:
                    break

                chunk_text = self.tokenizer.decode(
                    current_chunk_input_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                if chunk_text.strip():
                    current_doc_chunk_texts.append(chunk_text)

                if end_index == len(input_ids):
                    break
                start_index += step

            all_chunk_texts_global.extend(current_doc_chunk_texts)
            doc_chunk_indices.append(len(all_chunk_texts_global))

        batch_embeddings = []
        if not all_chunk_texts_global:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            for _ in X:
                batch_embeddings.append(np.zeros(embedding_dim))
            return np.array(batch_embeddings)

        global_chunk_embeddings = self.model.encode(
            all_chunk_texts_global, show_progress_bar=True
        )

        embedding_dim = self.model.get_sentence_embedding_dimension()
        for i in range(len(doc_chunk_indices) - 1):
            start_slice_idx = doc_chunk_indices[i]
            end_slice_idx = doc_chunk_indices[i + 1]

            if start_slice_idx == end_slice_idx:
                batch_embeddings.append(np.zeros(embedding_dim))
            else:
                doc_embeddings_slice = global_chunk_embeddings[
                    start_slice_idx:end_slice_idx
                ]
                mean_embedding = np.mean(doc_embeddings_slice, axis=0)
                batch_embeddings.append(mean_embedding)

        return np.array(batch_embeddings)


def create_all_minilm_xgboost_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chunk_length: int = 256,
    overlap: int = 20,
) -> Pipeline:
    return Pipeline(
        [
            (
                "sentence_transformer",
                SentenceTransformerWrapper(
                    model_name=model_name,
                    max_chunk_length=max_chunk_length,
                    overlap=overlap,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        use_label_encoder=False,
                    )
                ),
            ),
        ]
    )


def create_paraphrase_multilingual_minilm_xgboost_model(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_chunk_length: int = 128,
    overlap: int = 20,
) -> Pipeline:
    return Pipeline(
        [
            (
                "sentence_transformer",
                SentenceTransformerWrapper(
                    model_name=model_name,
                    max_chunk_length=max_chunk_length,
                    overlap=overlap,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        use_label_encoder=False,
                    )
                ),
            ),
        ]
    )


class HierarchicalModelWrapper:
    def __init__(
        self,
        base_model_pipeline,
        labels: list[str],
        hierarchy: dict[str, list[str]],
    ):
        self.model = base_model_pipeline
        self.labels = labels
        self.lti = {label: i for i, label in enumerate(labels)}
        self.itl = {i: label for label, i in self.lti.items()}
        self.hierarchy = hierarchy

    def fit(self, X_text, y_multihot):
        self.model.fit(X_text, y_multihot)
        return self

    def predict_raw_binary_matrix(self, X_text):
        return self.model.predict(X_text)

    def predict_proba_raw(self, X_text):
        return self.model.predict_proba(X_text)

    def predict(self, X_text):
        raw_binary_predictions = self.predict_raw_binary_matrix(X_text)

        corrected_binary_predictions = raw_binary_predictions.copy()
        for i in range(corrected_binary_predictions.shape[0]):
            for parent_str, children_str_list in self.hierarchy.items():
                parent_idx = self.lti.get(parent_str)
                if parent_idx is None:
                    continue

                for child_str in children_str_list:
                    child_idx = self.lti.get(child_str)
                    if child_idx is None:
                        continue

                    if (
                        corrected_binary_predictions[i, child_idx] == 1
                        and corrected_binary_predictions[i, parent_idx] == 0
                    ):
                        corrected_binary_predictions[i, parent_idx] = 1

        output_labels_list = []
        for i in range(corrected_binary_predictions.shape[0]):
            sample_labels = [
                self.itl[j]
                for j, is_present in enumerate(corrected_binary_predictions[i])
                if is_present
            ]
            output_labels_list.append(sample_labels)

        return output_labels_list

    def get_params(self, deep=True):
        return {
            "base_model_pipeline": self.model,
            "all_labels": self.labels,
            "hierarchy_definitions": self.hierarchy,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def create_tfidf_xgboost_hierarchical_adjusted_model(
    labels: list[str],
    hierarchy: dict[str, list[str]],
) -> Pipeline:
    return HierarchicalModelWrapper(
        base_model_pipeline=create_tfidf_xgboost_model(),
        labels=labels,
        hierarchy=hierarchy,
    )


class HDSentenceTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.model.encode(X, batch_size=1)


def create_bge_m3_xgboost_model() -> Pipeline:
    return Pipeline(
        [
            ("sentence_transformer", HDSentenceTransformerWrapper()),
            (
                "clf",
                OneVsRestClassifier(
                    xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )

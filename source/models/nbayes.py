from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def create_vectorizer_naive_bayes_model() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("classifier", MultiOutputClassifier(MultinomialNB())),
        ]
    )


def create_tfidf_naive_bayes_model() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", MultiOutputClassifier(MultinomialNB())),
        ]
    )

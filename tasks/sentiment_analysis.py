from typing import List

import numpy as np
from sklearn.base import TransformerMixin, RegressorMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


def make_pipeline():
    """
    This is where we will build our pipeline.
    Pipelines are basically a chain of transformers followed by an estimator.
    The first transformer should be one (or more) methods of feature extraction.

    :return: a working pipeline.
    """
    from sklearn.linear_model import LinearRegression
    pipeline_steps = [
        ('sfe', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('svr', LinearRegression())
    ]

    return Pipeline(steps=pipeline_steps)


class StupidFeatureExtractor(TransformerMixin):
    words = set()
    """
    Just a stupid transformer that takes in some data and returns garbage.
    """

    def __init__(self):
        """
        This is where you would accept the hyper parameters
        """
        pass

    def fit(self, documents, y=None):
        for document in documents:
            for word in document:
                self.words.add(word)
        return self

    def transform(self, documents: List[str], y=None):
        """
        This does not actually transform the documents to anything.
        It just spits out a random matrix with the correct amount of rows and 5 cols.
        """

        retval = np.empty((len(documents), len(self.words)))  # type: np.ndarray
        for i, document in enumerate(documents):
            for j, word in enumerate(self.words):
                retval[i, j] = document.count(word)
        return retval
        # return np.random.rand(len(documents), 5)


class RandomRegressor(RegressorMixin):
    """
    Just a stupid predictor that takes in some data and returns garbage.
    """

    def __init__(self, minimum=1, maximum=11):
        """
        This is where you would accept the hyper parameters
        """
        self.minimum = minimum
        self.maximum = maximum

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randint(low=self.minimum, high=self.maximum, size=X.shape[0])

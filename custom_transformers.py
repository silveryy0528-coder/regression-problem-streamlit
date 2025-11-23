import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=0.01, other_label='Other'):
        self.min_freq = min_freq
        self.other_label = other_label

    def fit(self, X, y=None):
        Xs = self._to_series(X)
        counts = Xs.value_counts(normalize=True)
        self.keep_categories_ = counts[counts >= self.min_freq].index.tolist()
        return self

    def transform(self, X):
        Xs = self._to_series(X)
        def map_val(val):
            if pd.isna(val):
                return val
            return val if val in self.keep_categories_ else self.other_label

        Xt = Xs.map(map_val)
        return Xt.to_frame().values

    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            return X
        else:
            return pd.Series(np.asarray(X).ravel())
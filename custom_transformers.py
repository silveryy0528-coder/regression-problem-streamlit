#%%
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold


def _to_series(X):
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0]
    elif isinstance(X, pd.Series):
        return X
    else:
        return pd.Series(np.asarray(X).ravel())


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    '''
    Groups rare categories in a categorical feature into a single 'Other' category
    '''
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
        return Xt.to_frame()

    @staticmethod
    def _to_series(X):
        return _to_series(X)


class QuantileClipper(BaseEstimator, TransformerMixin):
    '''
    Clips numerical values to specified quantile bounds; default is 1st and 99th percentiles.
    '''
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bound_ = np.quantile(X, self.lower)
        self.upper_bound_ = np.quantile(X, self.upper)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bound_, self.upper_bound_)


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, delimiter=';'):
        self.delimiter = delimiter
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        Xs = self._to_list_column(X)
        self.mlb.fit(Xs)

        # store feature names for DataFrame output
        self.feature_names_out_ = self.mlb.classes_.tolist()
        return self

    def transform(self, X):
        Xs = self._to_list_column(X)
        arr = self.mlb.transform(Xs)

        # make DataFrame
        df_out = pd.DataFrame(arr, columns=self.feature_names_out_)

        # reset index to align with original X
        df_out.index = (
            X.index if isinstance(X, pd.DataFrame)
            else pd.RangeIndex(start=0, stop=len(df_out))
        )

        return df_out

    def _to_list_column(self, X):
        """Convert the column into list-of-strings per row."""
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0].astype(str)
        else:
            s = pd.Series(X).astype(str)

        return s.apply(
            lambda x: [item.strip() for item in x.split(self.delimiter) if item.strip()]
        )

    def get_feature_names_out(self, input_features=None):
        """Optional: allow ColumnTransformer to query feature names."""
        return self.feature_names_out_


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=5, shuffle=True, random_state=42, smoothing=1.0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.smoothing = smoothing

    @staticmethod
    def _to_series(X):
        return _to_series(X)

    def fit(self, X, y):
        Xs = self._to_series(X)
        y = pd.Series(y)

        # global mean (used for smoothing & unknown categories)
        self.global_mean_ = y.mean()

        # dictionary: category → mean target (full-data encoding)
        self.full_encoding_ = y.groupby(Xs).mean().to_dict()

        return self

    def transform(self, X):
        Xs = self._to_series(X)

        # map categories to learned means; use global mean for unseen categories
        return Xs.map(lambda v: self.full_encoding_.get(v, self.global_mean_)).to_frame().values

    def fit_transform(self, X, y):
        """KFold leakage-free encoding for training data."""

        Xs = self._to_series(X)
        y = pd.Series(y)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        oof_encoded = pd.Series(index=Xs.index, dtype=float)

        for train_idx, valid_idx in kf.split(Xs):
            X_train, X_valid = Xs.iloc[train_idx], Xs.iloc[valid_idx]
            y_train = y.iloc[train_idx]

            # category → mean target computed on the TRAIN part only
            enc = y_train.groupby(X_train).mean().to_dict()

            # transform validation part
            oof_encoded.iloc[valid_idx] = X_valid.map(
                lambda v: enc.get(v, y_train.mean())
            )

        # fit final encodings on full training (for inference)
        self.fit(Xs, y)

        return oof_encoded.to_frame()


#%%
if __name__ == "__main__":
    # simple test
    data = pd.DataFrame({
        'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'E', 'A', 'B', 'C', 'D'],
        'Value': [10, 20, 15, 30, 25, 10, 40, 50, 15, 20, 30, 40]
    })
    kte = KFoldTargetEncoder(n_splits=3, smoothing=1.0)
    encoded = kte.fit_transform(data['Category'], data['Value'])
    print("KFold Target Encoded values:\n", encoded)

# %%

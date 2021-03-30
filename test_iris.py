import numpy
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.interpolate import interp1d
import warnings
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from tslearn.utils import (to_time_series_dataset, check_equal_size, ts_size,
                           check_dims)
from tslearn.bases import TimeSeriesBaseEstimator

# from sklearn.datasets import load_iris
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# iris = load_iris()
# X = iris.data
# print(X)
# print(type(X))
# sil_score_max = -1  # this is the minimum possible score

# for n_clusters in range(2, 10):
#     model = KMeans(n_clusters=n_clusters, init='k-means++',
#                    max_iter=100, n_init=1)
#     labels = model.fit_predict(X)
#     sil_score = silhouette_score(X, labels)
#     print("The average silhouette score for %i clusters is %0.2f" %
#           (n_clusters, sil_score))
#     if sil_score > sil_score_max:
#         sil_score_max = sil_score
#         best_n_clusters = n_clusters


class TimeSeriesScalerMeanVariance(TransformerMixin, TimeSeriesBaseEstimator):
    def __init__(self, mu=0., std=1.):
        self.mu = mu
        self.std = std

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.
        Parameters
        ----------
        X
            Ignored
        Returns
        -------
        self
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)
        self._X_fit_dims = X.shape
        return self

    def transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.
        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled
        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)
        mean_t = numpy.nanmean(X_, axis=1, keepdims=True)
        std_t = numpy.nanstd(X_, axis=1, keepdims=True)
        std_t[std_t == 0.] = 1.

        X_ = (X_ - mean_t) * self.std / std_t + self.mu

        return X_

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.
        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled.
        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        return self.fit(X).transform(X)

    def _more_tags(self):
        return {'allow_nan': True}


# fit_transform([[0, 3, 6]])
TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform([[0, 3, 6]])
print(TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform([[0, 3, 6]]))

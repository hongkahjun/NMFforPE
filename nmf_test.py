# from nmf_numba import non_negative_factorization
from sklearn.decomposition.nmf import non_negative_factorization
from sklearn.decomposition.nmf import _multiplicative_update_w
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import time
from IPython import get_ipython

ipython = get_ipython()

t0 = time.time()
all_samples, all_targets = make_classification(n_samples=1000, n_features=13, n_informative=6,
                                               n_redundant=2, n_repeated=0, n_classes=2,
                                               n_clusters_per_class=1, random_state=0)
all_samples += 500
ipython.magic(
    "lprun -f _multiplicative_update_w non_negative_factorization(all_samples, n_components=32, solver='mu', beta_loss='itakura-saito', max_iter=100)")

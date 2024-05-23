from pathlib import Path
import numpy as np


data_dir = Path("data")
deriv_dir = data_dir / "derivatives"

pyspi_raw_dim = 257

# Excludes any zero-variance and not-finite terms
pyspi_exclude_indices = np.array(
    [2, 10, 18, 26, 62, 145, 151, 153, 157, 158, 159, 163, 164, 165, 202, 203, 204, 236]
)

pyspi_clean_indices = [
    _ for _ in range(pyspi_raw_dim) if _ not in pyspi_exclude_indices
]
pyspi_clean_dim = len(pyspi_clean_indices)


pyspi_clean_terms = np.loadtxt(
    deriv_dir / "pyspi_terms_clean.txt", dtype=str, comments="#"
)
pyspi_clean_terms_prefix = np.loadtxt(
    deriv_dir / "pyspi_terms_clean_prefix.txt", dtype=str, comments="#"
)

pyspi_clean_dim_iu = np.triu_indices(pyspi_clean_dim, 1)

term_prefix_set, term_prefix_full_name, term_prefix_category = zip(
    *[
        ("cov", "Covariance", "Basic statistics"),
        ("cov-sq", "Covariance", "Basic statistics"),
        ("prec", "Precision", "Basic statistics"),
        ("prec-sq", "Precision", "Basic statistics"),
        ("spearmanr", "Spearman's rank-correlation coefficient", "Basic statistics"),
        ("kendalltau", "Kendall's rank-correlation coefficient", "Basic statistics"),
        ("xcorr", "Cross correlation", "Basic statistics"),
        ("pdist", "Pairwise distance", "Distance similarity"),
        ("dcorr", "Distance correlation", "Distance similarity"),
        ("hsic", "Hilbert-Schmidt Independence Criterion", "Distance similarity"),
        ("dcorrx", "Cross distance correlation", "Distance similarity"),
        ("dtw", "Dynamic time warping", "Distance similarity"),
        ("softdtw", "Soft dynamic time warping", "Distance similarity"),
        ("lcss", "Longest common subsequence", "Distance similarity"),
        ("bary", "Barycenter", "Distance similarity"),
        ("anm", "Additive noise model", "Causal inference"),
        ("cds", "Conditional distribution similarity fit", "Causal inference"),
        ("reci", "Regression error-based causal inference", "Causal inference"),
        ("igci", "Information-geometric causal inference", "Causal inference"),
        ("je", "Joint entropy", "Information theory"),
        ("ce", "Conditional entropy", "Information theory"),
        ("cce", "Causally conditioned entropy", "Information theory"),
        ("xme", "Cross-map entropy", "Information theory"),
        ("di", "Directed information", "Information theory"),
        ("si", "Stochastic interaction", "Information theory"),
        ("mi", "Mutual information", "Information theory"),
        ("tlmi", "Time-lagged mutual information", "Information theory"),
        ("te", "Transfer entropy", "Information theory"),
        ("phi", "Integrated information", "Information theory"),
        ("phase", "Coherence phase", "Spectral"),
        ("cohmag", "Coherence magnitude", "Spectral"),
        ("icoh", "Imaginary coherence", "Spectral"),
        ("psi", "Phase slope index", "Spectral"),
        ("plv", "Phase locking value", "Spectral"),
        ("pli", "Phase lag index", "Spectral"),
        ("wpli", "Weighted phase lag index", "Spectral"),
        ("dspli", "Debiased squared phase lag index", "Spectral"),
        ("dswpli", "Debiased squared weighted phase lag index", "Spectral"),
        ("ppc", "Pairwise phase consistency", "Spectral"),
        ("dtf", "Directed transfer function", "Spectral"),
        ("dcoh", "Directed coherence", "Spectral"),
        ("pdcoh", "Partial directed coherence", "Spectral"),
        ("gpdcoh", "Generalized partial directed coherence", "Spectral"),
        ("ddtf", "Direct directed transfer function", "Spectral"),
        ("sgc", "Spectral Granger causality", "Spectral"),
        ("psi-wavelet", "Phase slope index (wavelet)", "Spectral"),
        ("lmfit", "Linear model fit", "Miscellaneous"),
        ("coint", "Cointegration", "Miscellaneous"),
        ("pec", "Power envelope correlation", "Miscellaneous"),
    ],
    strict=True,
)

pyspi_terms_prefix_first_idx = np.array(
    [np.where(pyspi_clean_terms_prefix == _)[0].min() for _ in term_prefix_set]
)

aparc_dim = 68
aparc_iu = np.triu_indices(aparc_dim, 1)
schaefer100x7_dim = 100
schaefer100x7_iu = np.triu_indices(schaefer100x7_dim, 1)
schaefer200x7_dim = 200
schaefer200x7_iu = np.triu_indices(schaefer200x7_dim, 1)

hcp_subj_dim = 326
hcp_run_dim = 4

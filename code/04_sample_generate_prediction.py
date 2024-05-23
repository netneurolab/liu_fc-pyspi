import h5py
from numba import njit
from tqdm import trange
from .utils import *

pyspi_hcp_schaefer100x7_resave_dir = None
hcp_subj_reidx = None
terms_ye_dim = 5
valid_ye_feats = None


@njit
def calc_qcod(q1, q3):
    upper = (q3 - q1) / 2
    lower = (q1 + q3) / 2
    return upper / lower


bbpred_exclude_qcod_idx = []
for term_i in range(pyspi_clean_dim):
    print(f"{term_i = } {pyspi_clean_terms[term_i] = }")
    f = h5py.File(pyspi_hcp_schaefer100x7_resave_dir / f"term_{term_i}_iu.h5", "r")
    curr_term_iu = f[f"term_{term_i}_iu"][:]
    curr_term_iu_q1, curr_term_iu_q3 = np.nanpercentile(curr_term_iu, [25, 75], axis=0)
    curr_term_iu_qcod = calc_qcod(curr_term_iu_q1, curr_term_iu_q3)
    if np.abs(curr_term_iu_qcod.max()) < 0.01:
        bbpred_exclude_qcod_idx.append(term_i)
    f.close()

from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from scipy.spatial.distance import correlation as distance_correlation
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

alpha_list = [0.1, 1, 10, 100]

bbpred_res = []
for term_i in range(pyspi_clean_dim):
    print(f"{term_i = } {pyspi_clean_terms[term_i] = }")
    if term_i in bbpred_exclude_qcod_idx:
        continue
    f = h5py.File(pyspi_hcp_schaefer100x7_resave_dir / f"term_{term_i}_iu.h5", "r")
    curr_term_iu = f[f"term_{term_i}_iu"][hcp_subj_reidx, :]

    curr_term_iu_q1, curr_term_iu_q3 = np.nanpercentile(curr_term_iu, [25, 75], axis=0)
    curr_term_iu_qcod = calc_qcod(curr_term_iu_q1, curr_term_iu_q3)
    curr_term_iu_qcod_low, curr_term_iu_qcod_high = np.nanpercentile(
        curr_term_iu_qcod, [10, 90]
    )
    curr_term_iu_qcod_valid = np.where(
        (curr_term_iu_qcod >= curr_term_iu_qcod_low)
        & (curr_term_iu_qcod <= curr_term_iu_qcod_high)
    )[0]
    input_X = curr_term_iu[:, curr_term_iu_qcod_valid]

    for feat_i in range(terms_ye_dim):
        print(f"{feat_i = }")
        input_Y = valid_ye_feats[:, feat_i]

        inner_cv = KFold(n_splits=10, shuffle=True)
        outer_cv = KFold(n_splits=10, shuffle=True)

        # kernel ridge w linear kernel
        pipeline_kernelridge_base = make_pipeline(
            StandardScaler(), KernelRidge(kernel="linear")
        )
        inner_kernelridge = GridSearchCV(
            estimator=pipeline_kernelridge_base,
            param_grid={"kernelridge__alpha": alpha_list},
            cv=inner_cv,
        )
        nestedcv_kernelridge = cross_validate(
            inner_kernelridge,
            input_X,
            input_Y,
            cv=outer_cv,
            scoring={
                "r2": "r2",
                "distance_correlation": make_scorer(distance_correlation),
            },
            return_estimator=False,
        )

        nestedcv_kernelridge_corr_mean = np.nanmean(
            1 - nestedcv_kernelridge["test_distance_correlation"]
        )
        bbpred_res.append(
            (
                term_i,
                feat_i,
                "kernelridgelinear",
                nestedcv_kernelridge["test_distance_correlation"],
            )
        )
        print(f"{nestedcv_kernelridge_corr_mean = }")
        if nestedcv_kernelridge_corr_mean < 0:
            print(nestedcv_kernelridge["test_distance_correlation"])

        # kernel ridge w cosine kernel
        pipeline_kernelridgecos_base = make_pipeline(
            StandardScaler(), KernelRidge(kernel="cosine")
        )
        inner_kernelridgecos = GridSearchCV(
            estimator=pipeline_kernelridgecos_base,
            param_grid={"kernelridge__alpha": alpha_list},
            cv=inner_cv,
        )
        nestedcv_kernelridgecos = cross_validate(
            inner_kernelridgecos,
            input_X,
            input_Y,
            cv=outer_cv,
            scoring={
                "r2": "r2",
                "distance_correlation": make_scorer(distance_correlation),
            },
            return_estimator=False,
        )
        nestedcv_kernelridgecos_corr_mean = np.nanmean(
            1 - nestedcv_kernelridgecos["test_distance_correlation"]
        )
        bbpred_res.append(
            (
                term_i,
                feat_i,
                "kernelridgecosine",
                nestedcv_kernelridgecos["test_distance_correlation"],
            )
        )
        print(f"{nestedcv_kernelridgecos_corr_mean = }")
        if nestedcv_kernelridgecos_corr_mean < 0:
            print(nestedcv_kernelridgecos["test_distance_correlation"])

        # ridge
        pipeline_ridge_base = make_pipeline(StandardScaler(), Ridge(solver="auto"))
        inner_ridge = GridSearchCV(
            estimator=pipeline_ridge_base,
            param_grid={"ridge__alpha": alpha_list},
            cv=inner_cv,
        )
        nestedcv_ridge = cross_validate(
            inner_ridge,
            input_X,
            input_Y,
            cv=outer_cv,
            scoring={
                "r2": "r2",
                "distance_correlation": make_scorer(distance_correlation),
            },
            return_estimator=False,
        )
        nestedcv_ridge_corr_mean = np.nanmean(
            1 - nestedcv_ridge["test_distance_correlation"]
        )
        bbpred_res.append(
            (term_i, feat_i, "ridge", nestedcv_ridge["test_distance_correlation"])
        )
        print(f"{nestedcv_ridge_corr_mean = }")
        if nestedcv_ridge_corr_mean < 0:
            print(nestedcv_ridge["test_distance_correlation"])

        # lasso
        pipeline_lasso_base = make_pipeline(StandardScaler(), Lasso())
        inner_lasso = GridSearchCV(
            estimator=pipeline_lasso_base,
            param_grid={"lasso__alpha": alpha_list},
            cv=inner_cv,
        )
        nestedcv_lasso = cross_validate(
            inner_lasso,
            input_X,
            input_Y,
            cv=outer_cv,
            scoring={
                "r2": "r2",
                "distance_correlation": make_scorer(distance_correlation),
            },
            return_estimator=False,
        )
        nestedcv_lasso_corr_mean = np.nanmean(
            1 - nestedcv_lasso["test_distance_correlation"]
        )
        bbpred_res.append(
            (term_i, feat_i, "lasso", nestedcv_lasso["test_distance_correlation"])
        )
        print(f"{nestedcv_lasso_corr_mean = }")
        if nestedcv_lasso_corr_mean < 0:
            print(nestedcv_lasso["test_distance_correlation"])
    f.close()

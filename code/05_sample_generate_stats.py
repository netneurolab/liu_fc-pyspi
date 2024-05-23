from pathlib import Path
from netneurotools.stats import get_dominance_stats
from scipy.io import loadmat
from .utils import *

data_dir = Path("data")
deriv_dir = data_dir / "derivatives"

pid_term_names = [
    "rtr",
    "rtx",
    "rty",
    "rts",
    "xtr",
    "xtx",
    "xty",
    "xts",
    "ytr",
    "ytx",
    "yty",
    "yts",
    "str",
    "stx",
    "sty",
    "sts",
]
pid_type_index = 0
full_pidres = loadmat(deriv_dir / "HCP_S1200_schaefer100x7_PhiIDFull_MMI.mat")[
    "full_res"
]

termwise_mean_fc_schaefer100x7 = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_mean.npy"
)

pidres_grouavg_dom_mat = np.zeros((pyspi_clean_dim, len(pid_term_names)))
# termwise_mean_fc
full_pidres_groupavg = np.mean(full_pidres, axis=(0, 1))

full_pidres_groupavg_iu = np.array(
    [
        full_pidres_groupavg[:, :, pid_idx][schaefer100x7_iu]
        for pid_idx in range(len(pid_term_names))
    ]
).T
for term_it in range(pyspi_clean_dim):
    curr_term_name = pyspi_clean_terms[term_it]
    curr_term_prefix = pyspi_clean_terms_prefix[term_it]
    curr_pyspi_term_iu = termwise_mean_fc_schaefer100x7[term_it, :, :][schaefer100x7_iu]
    model_metrics, _ = get_dominance_stats(
        full_pidres_groupavg_iu, curr_pyspi_term_iu, verbose=True
    )
    pidres_grouavg_dom_mat[term_it, :] = model_metrics["total_dominance"]

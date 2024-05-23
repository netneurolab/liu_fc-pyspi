from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import scipy.stats as sstats
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression

from utils import *

data_dir = Path("data")
deriv_dir = data_dir / "derivatives"

termwise_mean_fc_schaefer100x7 = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_mean.npy"
)
dist_mat_schaefer100x7 = np.load(deriv_dir / "dist_mat.npy")
sc_cons_wei_schaefer100x7 = np.load(deriv_dir / "sc_cons_wei.npy")
sc_avggm_neglog_schaefer100x7 = -1 * np.log(
    sc_cons_wei_schaefer100x7 / (np.max(sc_cons_wei_schaefer100x7) + 1)
)

# Panel a: termwise mean FC distribution

termwise_mean_fc_distrib_hist = []
bin_edges = np.arange(0, 1.001, 0.01)

for term_it in range(pyspi_clean_dim):
    curr_pyspi_term_iu = termwise_mean_fc_schaefer100x7[term_it, :, :][schaefer100x7_iu]
    curr_pyspi_term_iu_norm = (
        curr_pyspi_term_iu - np.min(curr_pyspi_term_iu)
    ) / np.ptp(curr_pyspi_term_iu)
    curr_hist, _ = np.histogram(curr_pyspi_term_iu_norm, bins=bin_edges)
    termwise_mean_fc_distrib_hist.append(curr_hist)

termwise_mean_fc_distrib_hist = np.array(termwise_mean_fc_distrib_hist)
termwise_mean_fc_distrib_hist_norm = termwise_mean_fc_distrib_hist / np.sum(
    termwise_mean_fc_distrib_hist, axis=0
)

fig, ax = plt.subplots(figsize=(7, 1), constrained_layout=True)
pcm = ax.pcolormesh(
    termwise_mean_fc_distrib_hist_norm.T, cmap="YlOrRd", rasterized=True
)
pcm.set_clim(0, np.percentile(termwise_mean_fc_distrib_hist_norm, 97.5))
ax.invert_yaxis()
xticks = [0] + pyspi_terms_prefix_first_idx.tolist() + [pyspi_clean_dim]
ax.set(yticks=[], yticklabels=[], xticks=xticks, xticklabels=[])
plt.show()

# Panel c: weight-distance correlation
# (Panel b is in the next snippet)

df_corr_with_dist = []
corr_with_dist_list = []
for term_it in trange(pyspi_clean_dim):
    curr_term_name = pyspi_clean_terms[term_it]
    curr_term_prefix = pyspi_clean_terms_prefix[term_it]
    curr_pyspi_term_iu = termwise_mean_fc_schaefer100x7[term_it, :, :][schaefer100x7_iu]
    sr = sstats.spearmanr(dist_mat_schaefer100x7[schaefer100x7_iu], curr_pyspi_term_iu)[
        0
    ]
    corr_with_dist_list.append(sr)
    df_corr_with_dist.append((curr_term_name, curr_term_prefix, sr))
df_corr_with_dist = pd.DataFrame(
    df_corr_with_dist, columns=["term_name", "term_prefix", "value"]
)

# corr_termwise_mean_fc_with_dist_bar

fig, ax = plt.subplots(figsize=(7, 1), layout="constrained")
cnorm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
colors = [cm.RdYlBu_r(cnorm(_)) for _ in corr_with_dist_list]
ax.axhline(y=0, color="k", linestyle="--", linewidth=0.3, zorder=0)
ax.bar(
    range(pyspi_clean_dim),
    height=corr_with_dist_list,
    width=0.6,
    color=colors,
    rasterized=True,
)
xticks = np.array(pyspi_terms_prefix_first_idx.tolist() + [pyspi_clean_dim])
sns.despine(top=True, right=True, left=False, bottom=True, ax=ax)
ax.set(
    xlabel="",
    ylabel="",
    xticks=xticks - 0.5,
    xticklabels=[],
    xlim=(-0.6, pyspi_clean_dim + 0.6),
    ylim=(-0.3, 0.3),
    yticks=[-0.3, 0, 0.3],
)
ax.xaxis.set_tick_params(width=0.5, length=2)
plt.show()

# Panel b: hubs

termwise_mean_fc_abs_flipped_hub_ranks = []

for i in trange(pyspi_clean_dim):
    curr_iu = termwise_mean_fc_schaefer100x7[i, :, :][schaefer100x7_iu]
    #
    do_flip = 1 if corr_with_dist_list[i] < 0 else -1
    curr_iu_abs = np.abs(curr_iu)
    curr_abs = np.zeros((schaefer100x7_dim, schaefer100x7_dim))
    curr_abs[schaefer100x7_iu] = curr_iu_abs
    curr_abs += curr_abs.T
    curr_abs_rank = sstats.rankdata(np.sum(curr_abs, axis=0) * do_flip)
    termwise_mean_fc_abs_flipped_hub_ranks.append(curr_abs_rank)

termwise_mean_fc_abs_flipped_hub_ranks = np.array(
    termwise_mean_fc_abs_flipped_hub_ranks
)

# termwise_mean_fc_abs_flipped_hub_ranks

fig, ax = plt.subplots(figsize=(7, 1.5), constrained_layout=True)
pcm = ax.pcolormesh(
    termwise_mean_fc_abs_flipped_hub_ranks.T, cmap="YlOrRd", rasterized=True
)
pcm.set_clim(1, schaefer100x7_dim)
ax.invert_yaxis()
xticks = [0] + pyspi_terms_prefix_first_idx.tolist() + [pyspi_clean_dim]
ax.set(yticks=[], yticklabels=[], xticks=xticks, xticklabels=[])
plt.show()

# Panel d: structure-function relationship

x_comm_mats = np.load(deriv_dir / "x_comm_mats.npy")
dist_mat_schaefer100x7, spl_mat, npe_mat, sri_mat, cmc_mat, dfe_mat = x_comm_mats

x_comm_names = ["dist", "spl", "npe", "sri", "cmc", "dfe"]


termwise_mean_fc_cplg_global = []
for term_it in range(pyspi_clean_dim):
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    curr_mat = termwise_mean_fc_schaefer100x7[term_it, :, :]
    X_zs = sstats.zscore(
        np.c_[
            dist_mat_schaefer100x7[schaefer100x7_iu],
            spl_mat[schaefer100x7_iu],
            npe_mat[schaefer100x7_iu],
            sri_mat[schaefer100x7_iu],
            cmc_mat[schaefer100x7_iu],
            dfe_mat[schaefer100x7_iu],
        ],
        ddof=1,
    )
    X = X_zs
    y = curr_mat[schaefer100x7_iu]
    reg_res = reg.fit(X, y)
    yhat = reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    termwise_mean_fc_cplg_global.append(adjusted_r_squared)

df_termwise_mean_fc_cplg_global = pd.DataFrame(
    {
        "cplg": termwise_mean_fc_cplg_global,
        "term_prefix": pyspi_clean_terms_prefix.tolist(),
        "term_name": pyspi_clean_terms.tolist(),
    }
)


fig, ax = plt.subplots(figsize=(7, 1), layout="constrained")
cnorm = mcolors.Normalize(vmin=0, vmax=0.25)
colors = [cm.YlOrRd(cnorm(_)) for _ in termwise_mean_fc_cplg_global]
ax.axhline(y=0, color="k", linestyle="--", linewidth=0.3, zorder=0)
ax.bar(
    range(pyspi_clean_dim),
    height=termwise_mean_fc_cplg_global,
    width=0.6,
    color=colors,
    rasterized=True,
)
xticks = np.array(pyspi_terms_prefix_first_idx.tolist() + [pyspi_clean_dim])
sns.despine(top=True, right=True, left=False, bottom=True, ax=ax)
ax.set(
    xlabel="",
    ylabel="",
    xticks=xticks - 0.5,
    xticklabels=[],
    xlim=(-0.6, pyspi_clean_dim + 0.6),
    ylim=(0, 0.25),
    yticks=[0, 0.1, 0.2],
)
ax.xaxis.set_tick_params(width=0.5, length=2)
plt.show()

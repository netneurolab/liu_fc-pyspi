from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import scipy.stats as sstats
import seaborn as sns
from sklearn.linear_model import LinearRegression

from utils import *

data_dir = Path("data")
deriv_dir = data_dir / "derivatives"


group_term_sim_profile_schaefer100x7 = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_subj_term_profile_updated.npy"
)
group_term_sim_profile_mean_schaefer100x7 = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_profile_mean_updated.npy"
)


# Panel a

# num_nulls = 1000
# num_subj = group_term_sim_profile_schaefer100x7.shape[0]
# disc_repl_corr_list = []
# for i in trange(num_nulls):
#     curr_reidx = np.random.permutation(num_subj)
#     curr_reidx_a = curr_reidx[:int(num_subj / 2)]
#     curr_reidx_b = curr_reidx[int(num_subj / 2):]
#     curr_profile_mean_a = np.nanmean(
#         group_term_sim_profile_schaefer100x7[curr_reidx_a, :, :, :],
#         axis=(0, 1)
#     )
#     curr_profile_mean_b = np.nanmean(
#         group_term_sim_profile_schaefer100x7[curr_reidx_b, :, :, :],
#         axis=(0, 1)
#     )
#     disc_repl_corr_list.append(
#         sstats.spearmanr(
#             curr_profile_mean_a[pyspi_clean_dim_iu],
#             curr_profile_mean_b[pyspi_clean_dim_iu]
#         )[0]
#     )

disc_repl_corr_list = np.load(deriv_dir / "disc_repl_corr_list.npy")

fig, ax = plt.subplots(figsize=(1.5, 1.5))
sns.kdeplot(data=disc_repl_corr_list, ax=ax, fill=True)
ax.set(yticks=[])
sns.despine(top=True, right=True, left=False, bottom=False, ax=ax)
plt.show()

# Panel b

group_term_sim_profile_mean_aparc = np.load(
    deriv_dir / "pyspi_hcp_aparc_term_profile_mean.npy"
)

fig, ax = plt.subplots(figsize=(1.5, 1.5))
ax.scatter(
    group_term_sim_profile_mean_schaefer100x7[pyspi_clean_dim_iu],
    group_term_sim_profile_mean_aparc[pyspi_clean_dim_iu],
    s=1,
    alpha=0.5,
    rasterized=True,
)
ax.set(xticks=[-1, 0, 1], yticks=[-1, 0, 1])
reg = LinearRegression(fit_intercept=True, n_jobs=-1)
reg.fit(
    group_term_sim_profile_mean_schaefer100x7[pyspi_clean_dim_iu].reshape(-1, 1),
    group_term_sim_profile_mean_aparc[pyspi_clean_dim_iu].reshape(-1, 1),
)
sr, _ = sstats.spearmanr(
    group_term_sim_profile_mean_schaefer100x7[pyspi_clean_dim_iu],
    group_term_sim_profile_mean_aparc[pyspi_clean_dim_iu],
)
plot_x = np.linspace(-1, 1)
plot_y = reg.predict(plot_x.reshape(-1, 1))
ax.plot(plot_x, plot_y, color="tab:red", lw=1)
ax.text(0.8, -0.9, f"{sr:.2f}")
sns.despine(top=True, right=True, left=False, bottom=False, ax=ax)
plt.show()


# Panel c

group_term_sim_profile_mean_schaefer100x7_intersect = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_intersect_term_profile_mean.npy"
)
group_term_sim_profile_mean_schaefer200x7_intersect = np.load(
    deriv_dir / "pyspi_hcp_schaefer200x7_intersect_term_profile_mean.npy"
)

schaefer200x7_intersect_dim = group_term_sim_profile_mean_schaefer100x7_intersect.shape[
    0
]
schaefer200x7_intersect_dim_iu = np.triu_indices(schaefer200x7_intersect_dim, 1)

fig, ax = plt.subplots(figsize=(1.5, 1.5))
ax.scatter(
    group_term_sim_profile_mean_schaefer100x7_intersect[schaefer200x7_intersect_dim_iu],
    group_term_sim_profile_mean_schaefer200x7_intersect[schaefer200x7_intersect_dim_iu],
    s=1,
    alpha=0.5,
    rasterized=True,
)
ax.set(xticks=[-1, 0, 1], yticks=[-1, 0, 1])
reg = LinearRegression(fit_intercept=True, n_jobs=-1)
reg.fit(
    group_term_sim_profile_mean_schaefer100x7_intersect[
        schaefer200x7_intersect_dim_iu
    ].reshape(-1, 1),
    group_term_sim_profile_mean_schaefer200x7_intersect[
        schaefer200x7_intersect_dim_iu
    ].reshape(-1, 1),
)
sr, _ = sstats.spearmanr(
    group_term_sim_profile_mean_schaefer100x7_intersect[schaefer200x7_intersect_dim_iu],
    group_term_sim_profile_mean_schaefer200x7_intersect[schaefer200x7_intersect_dim_iu],
)
plot_x = np.linspace(-1, 1)
plot_y = reg.predict(plot_x.reshape(-1, 1))
ax.plot(plot_x, plot_y, color="tab:red", lw=1)
ax.text(0.8, -0.9, f"{sr:.2f}")
sns.despine(top=True, right=True, left=False, bottom=False, ax=ax)
plt.show()


# Panel d

group_term_sim_profile_mean_schaefer100x7gsr = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7gsr_term_profile_mean.npy"
)

fig, ax = plt.subplots(figsize=(1.5, 1.5))
ax.scatter(
    group_term_sim_profile_mean_schaefer100x7[pyspi_clean_dim_iu],
    group_term_sim_profile_mean_schaefer100x7gsr[pyspi_clean_dim_iu],
    s=1,
    alpha=0.5,
    rasterized=True,
)
ax.set(xticks=[-1, 0, 1], yticks=[-1, 0, 1])
reg = LinearRegression(fit_intercept=True, n_jobs=-1)
reg.fit(
    group_term_sim_profile_mean_schaefer100x7[pyspi_clean_dim_iu].reshape(-1, 1),
    group_term_sim_profile_mean_schaefer100x7gsr[pyspi_clean_dim_iu].reshape(-1, 1),
)
sr, _ = sstats.spearmanr(
    group_term_sim_profile_mean_schaefer100x7[pyspi_clean_dim_iu],
    group_term_sim_profile_mean_schaefer100x7gsr[pyspi_clean_dim_iu],
)
plot_x = np.linspace(-1, 1)
plot_y = reg.predict(plot_x.reshape(-1, 1))
ax.plot(plot_x, plot_y, color="tab:red", lw=1)
ax.text(0.8, -0.9, f"{sr:.2f}")
sns.despine(top=True, right=True, left=False, bottom=False, ax=ax)
plt.show()

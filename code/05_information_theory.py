from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
from utils import *

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

# Panel b

full_pidres_groupavg = np.mean(full_pidres, axis=(0, 1))
for i, curr_name in enumerate(pid_term_names):
    curr_mat = full_pidres_groupavg[:, :, i]
    fig, ax = plt.subplots(figsize=(1, 1))
    if np.nanpercentile(curr_mat, 50) > 0:
        pcm = ax.pcolormesh(curr_mat, cmap="YlOrRd", rasterized=True)
        pcm.set_clim([0, np.nanpercentile(curr_mat, 97.5)])
    else:
        pcm = ax.pcolormesh(curr_mat, cmap="Blues_r", rasterized=True)
        pcm.set_clim([np.nanpercentile(curr_mat, 2.5), 0])
    ax.invert_yaxis()
    ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
    ax.set_aspect("equal")
    plt.show()

# Panel c

pidres_grouavg_dom_mat = np.load(
    deriv_dir / "pidres_HCP_S1200_schaefer100x7_PhiIDFull_MMI_grouavg_dom_mat.npy"
)

pidres_grouavg_dom_mat_perc = np.array(
    [
        pidres_grouavg_dom_mat[_, :] / np.sum(pidres_grouavg_dom_mat[_, :])
        for _ in range(pyspi_clean_dim)
    ]
)

fig, ax = plt.subplots(figsize=(7, 1))
pcm = ax.pcolormesh(pidres_grouavg_dom_mat_perc.T, cmap="YlOrBr", rasterized=True)
pcm.set_clim(0, np.nanpercentile(pidres_grouavg_dom_mat_perc, 95))
ax.invert_yaxis()
ticks = [0] + pyspi_terms_prefix_first_idx.tolist() + [pyspi_clean_dim]
ax.set(xticks=ticks, yticks=[], xticklabels=[], yticklabels=[])
sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
ax.set_aspect("equal")
plt.show()

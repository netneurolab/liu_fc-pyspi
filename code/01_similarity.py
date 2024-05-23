from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import *

data_dir = Path("data")
deriv_dir = data_dir / "derivatives"

group_term_sim_profile_mean_schaefer100x7 = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_profile_mean_updated.npy"
)
group_term_sim_profile_var_schaefer100x7 = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_profile_var_updated.npy"
)

# pyspi_term_profile_mean

fig, ax = plt.subplots(figsize=(4.5, 4.5))
pcm = ax.pcolormesh(
    group_term_sim_profile_mean_schaefer100x7, cmap="RdBu_r", rasterized=True
)
pcm.set_clim(-1, 1)
ax.invert_yaxis()
ticks = [0] + pyspi_terms_prefix_first_idx.tolist() + [pyspi_clean_dim]
ax.set(xticks=ticks, yticks=ticks, xticklabels=[], yticklabels=[])
ax.set_aspect("equal")
plt.show()

# pyspi_term_profile_var

fig, ax = plt.subplots(figsize=(4.5, 4.5))
pcm = ax.pcolormesh(
    group_term_sim_profile_var_schaefer100x7, cmap="Reds", rasterized=True
)
pcm.set_clim(0, 0.03)
ax.invert_yaxis()
ticks = [0] + pyspi_terms_prefix_first_idx.tolist() + [pyspi_clean_dim]
ax.set(xticks=ticks, yticks=ticks, xticklabels=[], yticklabels=[])
ax.set_aspect("equal")
plt.show()

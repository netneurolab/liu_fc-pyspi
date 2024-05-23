from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sstats
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

from utils import *

data_dir = Path("data")
deriv_dir = data_dir / "derivatives"

termwise_mean_fc_schaefer100x7 = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_mean.npy"
)

many_networks_path = deriv_dir / "many_networks"
many_networks_names = [
    "gene_coexpression",
    "receptor_similarity",
    "laminar_similarity",
    "metabolic_connectivity",
    "electrophysiological_connectivity",
]
many_networks_mats = np.array(
    [np.load(many_networks_path / f"{fname}.npy") for fname in many_networks_names]
)


termwise_mean_fc_corr_many_networks = []
for term_it in range(pyspi_clean_dim):
    curr_term_name = pyspi_clean_terms[term_it]
    curr_term_prefix = pyspi_clean_terms_prefix[term_it]
    curr_pyspi_term_iu = termwise_mean_fc_schaefer100x7[term_it, :, :][schaefer100x7_iu]
    for i, name in enumerate(many_networks_names):
        curr_corr = sstats.spearmanr(
            curr_pyspi_term_iu, many_networks_mats[i, :, :][schaefer100x7_iu]
        )[0]
        termwise_mean_fc_corr_many_networks.append(
            (curr_term_name, curr_term_prefix, name, curr_corr)
        )

df_termwise_mean_fc_corr_many_networks = pd.DataFrame(
    termwise_mean_fc_corr_many_networks,
    columns=["term_name", "term_prefix", "many_networks_name", "spearmanr"],
)

for i, name in enumerate(many_networks_names):
    curr_mat = many_networks_mats[i, :, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    cnorm = mcolors.TwoSlopeNorm(
        vmin=np.nanpercentile(curr_mat, 0),
        vcenter=0.0,
        vmax=np.nanpercentile(curr_mat, 97.5),
    )
    pcm = ax.pcolormesh(curr_mat, cmap="RdBu_r", norm=cnorm, rasterized=True)
    ax.invert_yaxis()
    ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
    ax.set_aspect("equal")
    plt.show()

for i, name in enumerate(many_networks_names):
    curr_df = df_termwise_mean_fc_corr_many_networks[
        df_termwise_mean_fc_corr_many_networks["many_networks_name"] == name
    ]
    fig, ax = plt.subplots(figsize=(7, 1), layout="constrained")
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.3, zorder=0)
    sns.boxplot(
        data=curr_df,
        x="term_prefix",
        y="spearmanr",
        linewidth=0.5,
        color="lightgray",
        width=0.6,
        fliersize=1,
        order=list(term_prefix_set),
        ax=ax,
    )
    sns.despine(top=True, right=True, left=False, bottom=True, ax=ax)
    ax.set(xlabel="", ylabel="", xticks=[], xticklabels=[], ylim=(-0.5, 0.5))
    plt.show()

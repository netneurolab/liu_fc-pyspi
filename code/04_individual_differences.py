from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import pickle

from utils import *

data_dir = Path("data")
deriv_dir = data_dir / "derivatives"

termwise_mean_fc_schaefer100x7 = np.load(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_mean.npy"
)

# Panel a: fingerprinting

with open(deriv_dir / "term_i_identifiablity_list.pkl", "rb") as f:
    term_i_identifiablity_list = pickle.load(f)


df_term_i_identifiablity = pd.DataFrame(
    term_i_identifiablity_list, columns=["term_i", "identifiablity"]
)
df_term_i_identifiablity["term_name"] = df_term_i_identifiablity.apply(
    lambda x: pyspi_clean_terms[int(x["term_i"])], axis=1
)
df_term_i_identifiablity["term_prefix"] = df_term_i_identifiablity.apply(
    lambda x: pyspi_clean_terms_prefix[int(x["term_i"])], axis=1
)

fig, ax = plt.subplots(figsize=(7, 1), layout="constrained")
ax.axhline(y=0, color="k", linestyle="--", linewidth=0.3, zorder=0)
sns.boxplot(
    data=df_term_i_identifiablity,
    x="term_prefix",
    y="identifiablity",
    linewidth=0.5,
    color="lightgray",
    width=0.6,
    fliersize=1,
    order=list(term_prefix_set),
    ax=ax,
)
sns.despine(top=True, right=True, left=False, bottom=True, ax=ax)
ax.set(
    xlabel="", ylabel="", xticks=[], xticklabels=[], ylim=(-0.1, 3), yticks=[0, 1, 2]
)
plt.show()

# Panel b: brain-behavior prediction

with open(deriv_dir / "bbpred_res_202401_2.pkl", "rb") as f:
    bbpred_res = pickle.load(f)

terms_ye_dim = 5

df_bbpred_res = pd.DataFrame(
    bbpred_res, columns=["term_i", "feat_i", "clf_name", "distance_correlation_list"]
)
df_bbpred_res["correlation_mean"] = df_bbpred_res.apply(
    lambda x: 1 - np.nanmean(x["distance_correlation_list"]), axis=1
)
df_bbpred_res["term_name"] = df_bbpred_res.apply(
    lambda x: pyspi_clean_terms[x["term_i"]], axis=1
)
df_bbpred_res["term_prefix"] = df_bbpred_res.apply(
    lambda x: pyspi_clean_terms_prefix[x["term_i"]], axis=1
)

for clf_name in ["kernelridgelinear", "kernelridgecosine", "ridge", "lasso"]:
    curr_df_clf = df_bbpred_res[df_bbpred_res["clf_name"] == clf_name]
    bbpred_plot_mat = np.zeros((terms_ye_dim, len(term_prefix_set)))
    for prefix_i, prefix_name in enumerate(list(term_prefix_set)):
        for feat_i in range(5):
            curr_df = curr_df_clf[
                (curr_df_clf["term_prefix"] == prefix_name)
                & (curr_df_clf["feat_i"] == feat_i)
            ]
            bbpred_plot_mat[feat_i, prefix_i] = np.nanmean(curr_df["correlation_mean"])

    fig, ax = plt.subplots(figsize=(7, 1), constrained_layout=True)
    cnorm = mcolors.TwoSlopeNorm(
        vmin=np.nanpercentile(bbpred_plot_mat, 0),
        vcenter=0.0,
        vmax=np.nanpercentile(bbpred_plot_mat, 97.5),
    )
    pcm = ax.pcolormesh(
        bbpred_plot_mat, cmap="RdYlBu_r", norm=cnorm, rasterized=True
    )  # RdYlBu_r
    # plt.colorbar(pcm, ax=ax)
    ax.invert_yaxis()
    # ticks = [0] + pyspi_terms_prefix_first_idx.tolist() + [pyspi_clean_dim]
    ax.set(xticks=[], xticklabels=[], yticks=[])
    ax.set_aspect("equal")
    plt.show()

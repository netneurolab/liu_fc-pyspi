"""
This is an example of deriving key variables from raw pyspi output.

"""
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
import scipy.stats as sstats
import h5py
from .utils import *

data_dir = Path("data")
deriv_dir = data_dir / "derivatives"
pyspi_res_dir = data_dir / "raw"
pyspi_hcp_schaefer100x7_dir = pyspi_res_dir / "pyspi_hcp_schaefer100x7"

hcp_subj_list = np.loadtxt(
    pyspi_res_dir / "subjects_reinder326.txt", dtype=str
).tolist()


# pyspi_hcp_schaefer100x7_subj_term_profile

pyspi_hcp_schaefer100x7_subj_term_profile = np.zeros(
    (hcp_subj_dim, hcp_run_dim, pyspi_clean_dim, pyspi_clean_dim)
)

for subj_it, subj_id in tqdm(
    enumerate(hcp_subj_list), desc="subj_it", total=hcp_subj_dim
):
    for run_it in trange(hcp_run_dim, desc="run_it", leave=False):
        f = h5py.File(
            pyspi_hcp_schaefer100x7_dir / f"subj-{subj_id}_run-{run_it+1}.h5", "r"
        )
        curr_pyspi = f[f"subj-{subj_id}_run-{run_it+1}"]
        #
        curr_pyspi_iu = np.array(
            [
                curr_pyspi[pyspi_clean_indices[term_it], :, :][schaefer100x7_iu]
                for term_it in range(pyspi_clean_dim)
            ]
        )
        #
        pyspi_hcp_schaefer100x7_subj_term_profile[
            subj_it, run_it, :, :
        ] = sstats.spearmanr(curr_pyspi_iu, axis=1).statistic
        f.close()

np.save(
    deriv_dir / "pyspi_hcp_schaefer100x7_subj_term_profile_updated.npy",
    pyspi_hcp_schaefer100x7_subj_term_profile,
)

# pyspi_hcp_schaefer100x7_term_profile_mean/var

pyspi_hcp_schaefer100x7_term_profile_mean = np.nanmean(
    pyspi_hcp_schaefer100x7_subj_term_profile.reshape(
        (-1, pyspi_clean_dim, pyspi_clean_dim)
    ),
    axis=0,
)
pyspi_hcp_schaefer100x7_term_profile_var = np.nanvar(
    pyspi_hcp_schaefer100x7_subj_term_profile.reshape(
        (-1, pyspi_clean_dim, pyspi_clean_dim)
    ),
    axis=0,
)
np.save(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_profile_mean_updated.npy",
    pyspi_hcp_schaefer100x7_term_profile_mean,
)
np.save(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_profile_var_updated.npy",
    pyspi_hcp_schaefer100x7_term_profile_var,
)


# resave for efficiency

pyspi_hcp_schaefer100x7_resave_dir = pyspi_res_dir / "schaefer100x7_resave_clean_terms"
pyspi_hcp_schaefer100x7_resave_dir.mkdir(exist_ok=True)

for term_i in trange(pyspi_clean_dim):
    term_i_mats = np.zeros(
        (hcp_subj_dim, hcp_run_dim, schaefer100x7_dim, schaefer100x7_dim)
    )
    for subj_it, subj_id in tqdm(
        enumerate(hcp_subj_list), leave=False, total=hcp_subj_dim
    ):
        for run_it in range(hcp_run_dim):
            with h5py.File(
                pyspi_hcp_schaefer100x7_dir / f"subj-{subj_id}_run-{run_it+1}.h5", "r"
            ) as f:
                term_i_mats[subj_it, run_it, :, :] = f[
                    f"subj-{subj_id}_run-{run_it+1}"
                ][pyspi_clean_indices[term_i], :, :]

    term_i_mats_iu = np.array(
        [
            np.nanmean(term_i_mats[_, :, :, :], axis=0)[schaefer100x7_iu]
            for _ in range(hcp_subj_dim)
        ]
    )
    term_i_mats_iu_ranked = np.apply_along_axis(sstats.rankdata, 1, term_i_mats_iu)

    with h5py.File(pyspi_hcp_schaefer100x7_resave_dir / f"term_{term_i}.h5", "w") as f:
        dset = f.create_dataset(f"term_{term_i}", data=term_i_mats)
    with h5py.File(
        pyspi_hcp_schaefer100x7_resave_dir / f"term_{term_i}_iu.h5", "w"
    ) as f:
        dset = f.create_dataset(f"term_{term_i}_iu", data=term_i_mats_iu)
    with h5py.File(
        pyspi_hcp_schaefer100x7_resave_dir / f"term_{term_i}_iu_ranked.h5", "w"
    ) as f:
        dset = f.create_dataset(f"term_{term_i}_iu_ranked", data=term_i_mats_iu_ranked)


# pyspi_hcp_schaefer100x7_term_mean

pyspi_hcp_schaefer100x7_term_mean = np.zeros(
    (pyspi_clean_dim, schaefer100x7_dim, schaefer100x7_dim)
)

for term_i in trange(pyspi_clean_dim):
    f = h5py.File(pyspi_hcp_schaefer100x7_resave_dir / f"term_{term_i}.h5", "r")
    term_i_mats = f[f"term_{term_i}"]
    pyspi_hcp_schaefer100x7_term_mean[term_i, :, :] = np.nanmean(
        term_i_mats, axis=(0, 1)
    )

np.save(
    deriv_dir / "pyspi_hcp_schaefer100x7_term_mean.npy",
    pyspi_hcp_schaefer100x7_term_mean,
)

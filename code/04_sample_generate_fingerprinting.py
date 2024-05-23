import h5py
from tqdm import trange
from .utils import *

pyspi_hcp_schaefer100x7_resave_dir = None

indices_flatten_list = []
for subj_i in range(hcp_subj_dim):
    for run_i in range(hcp_run_dim):
        indices_flatten_list.append((subj_i, run_i))

indices_flatten_intra = []
indices_flatten_inter = []
for it_1, idx_pair_1 in enumerate(indices_flatten_list):
    for it_2, idx_pair_2 in enumerate(indices_flatten_list):
        if it_1 == it_2:
            continue
        if idx_pair_1[0] == idx_pair_2[0]:
            indices_flatten_intra.append((it_1, it_2))
        else:
            indices_flatten_inter.append((it_1, it_2))

intra_len = len(indices_flatten_intra)
inter_len = len(indices_flatten_inter)

term_i_identifiablity_list = []
for term_i in trange(pyspi_clean_dim):
    f = h5py.File(pyspi_hcp_schaefer100x7_resave_dir / f"term_{term_i}.h5", "r")
    term_i_mat = f[f"term_{term_i}"][:]
    term_i_mat_iu = []
    for subj_i in range(hcp_subj_dim):
        for run_i in range(hcp_run_dim):
            term_i_mat_iu.append(term_i_mat[subj_i, run_i, :, :][schaefer100x7_iu])
    term_i_mat_iu = np.array(term_i_mat_iu)
    term_i_mat_iu_corr = np.corrcoef(term_i_mat_iu)
    curr_val_intra = [term_i_mat_iu_corr[_[0], _[1]] for _ in indices_flatten_intra]
    curr_val_inter = [term_i_mat_iu_corr[_[0], _[1]] for _ in indices_flatten_inter]

    curr_id_upper = np.abs(np.nanmean(curr_val_intra) - np.nanmean(curr_val_inter))
    curr_id_lower = np.sqrt(
        (
            (intra_len - 1) * np.nanstd(curr_val_intra) ** 2
            + (inter_len - 1) * np.nanstd(curr_val_inter) ** 2
        )
        / (intra_len + inter_len - 2)
    )
    curr_id_full = curr_id_upper / curr_id_lower
    print(term_i, curr_id_full)
    term_i_identifiablity_list.append((term_i, curr_id_full))
    f.close()

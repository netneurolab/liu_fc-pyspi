import sys
from pathlib import Path
import numpy as np
import h5py
import nibabel as nib
from oct2py import octave

subj_id = str(sys.argv[1])
parc_str = str(sys.argv[2])

work_dir = Path("/scratch/zqliu/temp_code/202308-pyspi-rerun")
src_data_dir = work_dir / f"data/HCP_S1200/{parc_str}"

output_dir = work_dir / f"outputs/HCP_S1200/{parc_str}"
output_dir.mkdir(exist_ok=True, parents=True)

run_dim = 4
if parc_str == "aparc":
    mat_dim = 68
elif parc_str == "schaefer100x7":
    mat_dim = 100
elif parc_str == "schaefer100x7-gsr":
    mat_dim = 100
elif parc_str == "schaefer200x7":
    mat_dim = 200
elif parc_str == "schaefer400x7":
    mat_dim = 400
else:
    mat_dim = None
# subj_dim, run_dim, mat_dim, time_dim = (326, 4, 100, 1200)

iu = np.triu_indices(mat_dim, 1)

from pyspi.calculator import Calculator
from pyspi.data import Data

for run_num in range(run_dim):
    print(f"{subj_id = } {run_num+1 = }")

    if (output_dir / f"subj-{subj_id}_run-{run_num+1}_spi.txt").exists():
        print(f"Found finished, skipping this run...")
        continue

    if parc_str == "schaefer100x7-gsr":
        curr_data = np.load(
            src_data_dir
            / f"subj-{subj_id}_run-{run_num+1}_atlas-{parc_str[:-4]}_gsr_zs.npy"
        )
    else:
        curr_data = (
            nib.load(
                src_data_dir
                / f"subj-{subj_id}_run-{run_num+1}_atlas-{parc_str}.ptseries.nii"
            )
            .get_fdata()
            .T
        )

    if parc_str == "aparc":
        curr_data = np.delete(curr_data, [3, 38], axis=0)

    curr_dataset = Data(data=curr_data, dim_order="ps", normalise=True)
    curr_calc = Calculator(
        dataset=curr_dataset, configfile=work_dir / "config-1-minimized.yaml"
    )
    curr_calc.compute()

    spi_list = list(curr_calc.spis.keys())
    curr_mat = np.zeros((len(spi_list), mat_dim, mat_dim))
    for spi_it, spi_name in enumerate(spi_list):
        curr_mat[spi_it, :, :] = curr_calc.table[spi_name].to_numpy()

    with h5py.File(output_dir / f"subj-{subj_id}_run-{run_num+1}.h5", "w") as f:
        dset = f.create_dataset(f"subj-{subj_id}_run-{run_num+1}", data=curr_mat)

    np.savetxt(
        str(output_dir / f"subj-{subj_id}_run-{run_num+1}_spi.txt"), spi_list, fmt="%s"
    )

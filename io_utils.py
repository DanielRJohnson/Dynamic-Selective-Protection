import os
import numpy as np
from scipy.io import loadmat, savemat


def extract_matrices_from_dir(matdir: str) -> dict[str, np.array]:
    matfiles = [matdir + "/" + f for f in os.listdir(matdir) if f.endswith(".mat")]
    matrices = {mf: loadmat(mf)["Problem"]["A"] for mf in matfiles}
    return matrices


def write_matrices(matrices: dict[str, np.array], varname: str) -> None:
    for fname, mat in matrices.items():
        mdict = {varname: mat}
        savemat(fname, mdict)

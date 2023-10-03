import os
import numpy as np
from scipy.io import loadmat, savemat


def load_matrices_from_dir(matdir: str, extract_func: callable, subset=None) -> dict[str, np.array]:
    subset = subset if subset is not None else [f for f in os.listdir(matdir) if f.endswith(".mat")]
    matfiles = [matdir + "/" + f for f in os.listdir(matdir) if f in subset]
    matrices = {mf: extract_func(loadmat(mf)) for mf in matfiles}
    return matrices


def write_matrices(matrices: dict[str, np.array], varname: str) -> None:
    for fname, mat in matrices.items():
        mdict = {varname: mat}
        savemat(fname, mdict)

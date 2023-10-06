import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix


def load_matrices_from_dir(matdir: str, subset=None) -> dict[str, csr_matrix]:
    extract_mode = matdir.split("/")[-1]  # name of folder
    extractors = {  # how to get the sparse matrix from the object m
        "raw": lambda m: m["Problem"]["A"][0][0],
        "preconditioners": lambda m: m["L"],
    }

    subset = [m + (".mat" if extract_mode == "raw" else "_precond.mat") for m in subset]
    matfiles = [matdir + "/" + f for f in os.listdir(matdir) if f in subset or subset is None]
    matrices = {mf: extractors[extract_mode](loadmat(mf)) for mf in matfiles}
    return matrices


def write_matrices(matrices: dict[str, np.array], varname: str) -> None:
    for fname, mat in matrices.items():
        mdict = {varname: mat}
        savemat(fname, mdict)

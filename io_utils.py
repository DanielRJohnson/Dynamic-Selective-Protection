import os
from collections import defaultdict, OrderedDict
from joblib import load

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

from pcg import PcgOptions, PcgResult


def load_matrices_from_dir(matdir: str, subset=None) -> dict[str, csr_matrix]:
    extract_mode = matdir.split("/")[-1]  # name of folder
    extractors = {  # how to get the sparse matrix from the object m
        "raw": lambda m: m["Problem"]["A"][0][0],
        "preconditioners": lambda m: m["L"],
    }

    subset = [m + (".mat" if extract_mode == "raw" else "_precond.mat")
              for m in subset] if subset is not None else None
    matfiles = [matdir + "/" +
                f for f in os.listdir(matdir) if subset is None or f in subset]

    matrices = OrderedDict()  # sort alphabetically to ensure consistency between calls
    for mf in sorted(matfiles, key=lambda m: m.split("/")[-1]):
        mat = loadmat(mf)
        matrices[mf.split("/")[-1]] = extractors[extract_mode](mat)

    return matrices


def write_matrices(matrices: dict[str, np.array], varname: str) -> None:
    for fname, mat in matrices.items():
        mdict = {varname: mat}
        savemat(fname, mdict)


def load_models(model_paths: list[str]):
    return [load(path) for path in model_paths]


def write_results_csv(mat_name: str, A: csr_matrix, errorfree_iterations: int,
                      opts_list: list[PcgOptions], results: list[PcgResult]) -> None:
    assert len(opts_list) == len(results)
    out = defaultdict(list)
    for opt, res in zip(opts_list, results):
        out["mat_name"].append(mat_name)
        out["errorfree_iterations"].append(errorfree_iterations)
        out["tol"].append(opt.tol)
        out["maxiter"].append(opt.maxiter)
        out["error_pos"].append(opt.error_pos)
        out["error_iter"].append(opt.error_iter)
        out["solve_iterations"].append(res.solve_iterations)
        out["final_relres"].append(res.final_relres)
        out["did_converge"].append(res.did_converge)
        out["realtime_s"].append(res.realtime_s)
        out["pos_2norm"].append(norm(A.getrow(opt.error_pos - 1)))
        out["n_rows"].append(A.shape[0])

    df = pd.DataFrame(out)
    df["slowdown"] = df["solve_iterations"] / df["errorfree_iterations"]

    fn = os.path.dirname(
        __file__) + f"/analyses/data/{mat_name[:-4]}_{len(results)}.csv"
    df.to_csv(fn, index=False)

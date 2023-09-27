import argparse
import numpy as np
from io_utils import extract_matrices_from_dir, write_matrices


def compute_row_2norms(mat: np.array) -> np.array:
    return np.linalg.norm(mat, axis=1, ord=None)  # 2norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Computes the 2-norms of all .mat files in a given directory,"
        " and outputs the results to a new directory on the same level")

    parser.add_argument("matrix_dir", type=str)
    args = parser.parse_args()
    matdir = args.matrix_dir

    matrices = extract_matrices_from_dir(matdir)
    norms = {(mf[:-4] + "_norms" + mf[-4:]).replace("raw", "2norms"):
                 compute_row_2norms(mat) for mf, mat in matrices.items()}
    write_matrices(norms, "A_row_2norms")

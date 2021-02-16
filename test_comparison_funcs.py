import comparison_funcs as cf
import numpy as np


def test_permute_matrix():
    D = np.array([[9, 7, 6, 4],
                  [8, 6, 2, 5],
                  [5, 1, 6, 7],
                  [3, 8, 6, 9]])
    E = np.array([[1, 6, 5, 7],
                  [6, 2, 8, 5],
                  [8, 6, 3, 9],
                  [7, 6, 9, 4]])
    F = cf.permute_matrix(D)
    check_list = [f == e for f, e in zip(F.ravel(), E.ravel())]
    assert all(check_list), "permute_matrix failed"

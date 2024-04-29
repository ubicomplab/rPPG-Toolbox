"""OMIT
Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces.
Álvarez Casado, C., & Bordallo López, M.
IEEE Journal of Biomedical and Health Informatics.
(2023).
"""

import numpy as np
from unsupervised_methods import utils


def OMIT(frames):
    precessed_data = utils.process_video(frames)
    precessed_data = precessed_data[0]
    Q, R = np.linalg.qr(precessed_data)
    S = Q[:, 0].reshape(1, -1)
    P = np.identity(3) - np.matmul(S.T, S)
    Y = np.dot(P, precessed_data)
    bvp = Y[1, :]
    bvp = bvp.reshape(-1)
    return bvp

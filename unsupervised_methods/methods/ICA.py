"""ICA
Non-contact, automated cardiac pulse measurements using video imaging and blind source separation.
Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).
Optics express, 18(10), 10762-10774. DOI: 10.1364/OE.18.010762
"""
import math

import numpy as np
from scipy import linalg
from scipy import signal
from unsupervised_methods import utils


def ICA(frames, FS):
    LPF = 0.7
    HPF = 2.5
    NyquistF = 1 / 2 * FS

    rgb = _process_video(frames)
    bgr_norm = np.zeros(rgb.shape)
    lambda_0 = 100
    for c in range(3):
        bgr_detrend = utils.detrend(rgb[:, c], lambda_0)
        bgr_norm[:, c] = (bgr_detrend - np.mean(bgr_detrend)) / np.std(bgr_detrend)
    _, s = ica(np.mat(bgr_norm).H, 3)

    # select BVP Source
    max_px = np.zeros((1, 3))
    for c in range(3):
        ff = np.fft.fft(s[c, :])
        ff = ff[:, 1:]
        ff = ff[0]
        n = ff.shape[0]
        px = np.abs(ff[:math.floor(n / 2)])
        px = np.multiply(px, px)
        px = px / np.sum(px, axis=0)
        max_px[0, c] = np.max(px)
    max_comp = np.argmax(max_px)
    bvp_i = s[max_comp, :]
    b, a = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
    bvp_f = signal.filtfilt(b, a, np.real(bvp_i).astype(np.double))

    bvp = bvp_f[0]
    return bvp


def _process_video(frames):
    "Calculates the average value of each frame."
    rgb = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        rgb.append(sum / (frame.shape[0] * frame.shape[1]))
    return np.asarray(rgb)


def ica(X, n_sources, w_prev=0):
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    if n_rows > n_cols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if n_sources > min(n_rows, n_cols):
        n_sources = min(n_rows, n_cols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', n_sources)

    w_inv, z_hat = jade(X, n_sources, w_prev)
    w = np.linalg.pinv(w_inv)
    return w, z_hat


def jade(x, m, w_prev):
    n = x.shape[0]
    t = x.shape[1]
    nem = m
    seuil = 1 / math.sqrt(t) / 100
    if m < n:
        d, u = np.linalg.eig(np.matmul(x, np.mat(x).H) / t)
        diag = d
        k = np.argsort(diag)
        pu = diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        w = np.matmul(np.diag(bl), np.transpose(u[0:n, k[n - m:n]]))
        iw = np.matmul(u[0:n, k[n - m:n]], np.diag(ibl))
    else:
        iw = linalg.sqrtm(np.matmul(x, x.H) / t)
        w = np.linalg.inv(iw)

    y = np.mat(np.matmul(w, x))
    r = np.matmul(y, y.H) / t
    c = np.matmul(y, y.T) / t
    q = np.zeros((m * m * m * m, 1))
    index = 0

    for lx in range(m):
        y1 = y[lx, :]
        for kx in range(m):
            yk1 = np.multiply(y1, np.conj(y[kx, :]))
            for jx in range(m):
                yjk1 = np.multiply(yk1, np.conj(y[jx, :]))
                for ix in range(m):
                    q[index] = np.matmul(yjk1 / math.sqrt(t), y[ix, :].T / math.sqrt(
                        t)) - r[ix, jx] * r[lx, kx] - r[ix, kx] * r[lx, jx] - c[ix, lx] * np.conj(c[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    d, u = np.linalg.eig(q.reshape(m * m, m * m))
    diag = abs(d)
    k = np.argsort(diag)
    la = diag[k]
    m = np.zeros((m, nem * m), dtype=complex)
    z = np.zeros(m)
    h = m * m - 1
    for u in range(0, nem * m, m):
        z = u[:, k[h]].reshape((m, m))
        m[:, u:u + m] = la[h] * z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    b = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    bt = np.mat(b).H

    encore = 1
    if w_prev == 0:
        v = np.eye(m).astype(complex)
    else:
        v = np.linalg.inv(w_prev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([m[p, Ip] - m[q, Iq], m[p, Iq], m[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(b, temp1)
                temp = np.matmul(temp2, bt)
                d, vcp = np.linalg.eig(np.real(temp))
                k = np.argsort(d)
                la = d[k]
                angles = vcp[:, k[2]]
                if angles[0, 0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    g = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    v[:, pair] = np.matmul(v[:, pair], g)
                    m[pair, :] = np.matmul(g.H, m[pair, :])
                    temp1 = c * m[:, Ip] + s * m[:, Iq]
                    temp2 = -np.conj(s) * m[:, Ip] + c * m[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    m[:, Ip] = temp1
                    m[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    a = np.matmul(iw, v)
    s = np.matmul(np.mat(v).H, y)
    return a, s

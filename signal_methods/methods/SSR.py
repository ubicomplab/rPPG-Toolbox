# import cupy
import math
import time
import numpy as np
import torch
import os
from sklearn.decomposition import PCA
from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float32, float64, matrix, multiply, ndarray, newaxis, \
    sign, sin, sqrt, zeros
import numpy as np

"""
This module contains a collection of known rPPG methods.

rPPG METHOD SIGNATURE
An rPPG method must accept theese parameters:
    > signal -> RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames], or a custom signal.
    > **kargs [OPTIONAL] -> usefull parameters passed to the filter method.
It must return a BVP signal as float32 ndarray with shape [num_estimators, num_frames].
"""


# ------------------------------------------------------------------------------------- #
#                                     rPPG METHODS                                      #
# ------------------------------------------------------------------------------------- #


def cpu_CHROM(signal):
    """
    CHROM method on CPU using Numpy.

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """
    X = signal
    Xcomp = 3 * X[:, 0] - 2 * X[:, 1]
    Ycomp = (1.5 * X[:, 0]) + X[:, 1] - (1.5 * X[:, 2])
    sX = np.std(Xcomp, axis=1)
    sY = np.std(Ycomp, axis=1)
    alpha = (sX / sY).reshape(-1, 1)
    alpha = np.repeat(alpha, Xcomp.shape[1], 1)
    bvp = Xcomp - np.multiply(alpha, Ycomp)
    return bvp


# def cupy_CHROM(signal):
#     """
#     CHROM method on GPU using Cupy.
#
#     De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
#     """
#     X = signal
#     Xcomp = 3 * X[:, 0] - 2 * X[:, 1]
#     Ycomp = (1.5 * X[:, 0]) + X[:, 1] - (1.5 * X[:, 2])
#     sX = cupy.std(Xcomp, axis=1)
#     sY = cupy.std(Ycomp, axis=1)
#     alpha = (sX / sY).reshape(-1, 1)
#     alpha = cupy.repeat(alpha, Xcomp.shape[1], 1)
#     bvp = Xcomp - cupy.multiply(alpha, Ycomp)
#     return bvp


def torch_CHROM(signal):
    """
    CHROM method on CPU using Torch.

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """
    X = signal
    Xcomp = 3 * X[:, 0] - 2 * X[:, 1]
    Ycomp = (1.5 * X[:, 0]) + X[:, 1] - (1.5 * X[:, 2])
    sX = torch.std(Xcomp, axis=1)
    sY = torch.std(Ycomp, axis=1)
    alpha = (sX / sY).reshape(-1, 1)
    alpha = torch.repeat_interleave(alpha, Xcomp.shape[1], 1)
    bvp = Xcomp - torch.mul(alpha, Ycomp)
    return bvp


def cpu_LGI(signal):
    """
    LGI method on CPU using Numpy.

    Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
    """
    X = signal
    U, _, _ = np.linalg.svd(X)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    sst = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - sst
    Y = np.matmul(P, X)
    bvp = Y[:, 1, :]
    return bvp


def cpu_POS(signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10 ** -9
    X = signal
    e, c, f = X.shape  # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])  # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)  # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H


# def cupy_POS(signal, **kargs):
#     """
#     POS method on GPU using Cupy.
#
#     The dictionary parameters are: {'fps':float}.
#
#     Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
#     """
#     # Run the pos algorithm on the RGB color signal c with sliding window length wlen
#     # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
#     eps = 10 ** -9
#     X = signal
#     fps = cupy.float32(kargs['fps'])
#     e, c, f = X.shape  # e = #estimators, c = 3 rgb ch., f = #frames
#     w = int(1.6 * fps)  # window length
#
#     # stack e times fixed mat P
#     P = cupy.array([[0, 1, -1], [-2, 1, 1]])
#     Q = cupy.stack([P for _ in range(e)], axis=0)
#
#     # Initialize (1)
#     H = cupy.zeros((e, f))
#     for n in cupy.arange(w, f):
#         # Start index of sliding window (4)
#         m = n - w + 1
#         # Temporal normalization (5)
#         Cn = X[:, :, m:(n + 1)]
#         M = 1.0 / (cupy.mean(Cn, axis=2) + eps)
#         M = cupy.expand_dims(M, axis=2)  # shape [e, c, w]
#         Cn = cupy.multiply(M, Cn)
#
#         # Projection (6)
#         S = cupy.dot(Q, Cn)
#         S = S[0, :, :, :]
#         S = cupy.swapaxes(S, 0, 1)  # remove 3-th dim
#
#         # Tuning (7)
#         S1 = S[:, 0, :]
#         S2 = S[:, 1, :]
#         alpha = cupy.std(S1, axis=1) / (eps + cupy.std(S2, axis=1))
#         alpha = cupy.expand_dims(alpha, axis=1)
#         Hn = cupy.add(S1, alpha * S2)
#         Hnm = Hn - cupy.expand_dims(cupy.mean(Hn, axis=1), axis=1)
#         # Overlap-adding (8)
#         H[:, m:(n + 1)] = cupy.add(H[:, m:(n + 1)], Hnm)
#
#     return H


def cpu_PBV(signal):
    """
    PBV method on CPU using Numpy.

    De Haan, G., & Van Leest, A. (2014). Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9), 1913.
    """
    sig_mean = np.mean(signal, axis=2)

    signal_norm_r = signal[:, 0, :] / np.expand_dims(sig_mean[:, 0], axis=1)
    signal_norm_g = signal[:, 1, :] / np.expand_dims(sig_mean[:, 1], axis=1)
    signal_norm_b = signal[:, 2, :] / np.expand_dims(sig_mean[:, 2], axis=1)

    pbv_n = np.array([np.std(signal_norm_r, axis=1), np.std(signal_norm_g, axis=1), np.std(signal_norm_b, axis=1)])
    pbv_d = np.sqrt(np.var(signal_norm_r, axis=1) + np.var(signal_norm_g, axis=1) + np.var(signal_norm_b, axis=1))
    pbv = pbv_n / pbv_d

    C = np.swapaxes(np.array([signal_norm_r, signal_norm_g, signal_norm_b]), 0, 1)
    Ct = np.swapaxes(np.swapaxes(np.transpose(C), 0, 2), 1, 2)
    Q = np.matmul(C, Ct)
    W = np.linalg.solve(Q, np.swapaxes(pbv, 0, 1))

    A = np.matmul(Ct, np.expand_dims(W, axis=2))
    B = np.matmul(np.swapaxes(np.expand_dims(pbv.T, axis=2), 1, 2), np.expand_dims(W, axis=2))
    bvp = A / B
    return bvp.squeeze(axis=2)


def cpu_PCA(signal, **kargs):
    """
    PCA method on CPU using Numpy.

    The dictionary parameters are {'component':str}. Where 'component' can be 'second_comp' or 'all_comp'.

    Lewandowska, M., Rumiński, J., Kocejko, T., & Nowak, J. (2011, September). Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity. In 2011 federated conference on computer science and information systems (FedCSIS) (pp. 405-410). IEEE.
    """
    bvp = []
    for i in range(signal.shape[0]):
        X = signal[i]
        pca = PCA(n_components=3)
        pca.fit(X)

        # selector
        if kargs['component'] == 'all_comp':
            bvp.append(pca.components_[0] * pca.explained_variance_[0])
            bvp.append(pca.components_[1] * pca.explained_variance_[1])
        elif kargs['component'] == 'second_comp':
            bvp.append(pca.components_[1] * pca.explained_variance_[1])
    bvp = np.array(bvp)
    return bvp


def cpu_GREEN(signal):
    """
    GREEN method on CPU using Numpy

    Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.
    """
    return signal[:, 1, :]


def cpu_OMIT(signal):
    """
    OMIT method on CPU using Numpy.

    Álvarez Casado, C., Bordallo López, M. (2022). Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. arXiv (eprint 2202.04101).
    """

    bvp = []
    for i in range(signal.shape[0]):
        X = signal[i]
        Q, R = np.linalg.qr(X)
        S = Q[:, 0].reshape(1, -1)
        P = np.identity(3) - np.matmul(S.T, S)
        Y = np.dot(P, X)
        bvp.append(Y[1, :])
    bvp = np.array(bvp)
    return bvp


def cpu_ICA(signal, **kargs):
    """
    ICA method on CPU using Numpy.

    The dictionary parameters are {'component':str}. Where 'component' can be 'second_comp' or 'all_comp'.

    Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.
    """
    bvp = []
    for X in signal:
        W = jadeR(X, verbose=False)
        bvp.append(np.dot(W, X))

    # selector
    bvp = np.array(bvp)
    l, c, f = bvp.shape  # l=#landmks c=#3chs, f=#frames
    if kargs['component'] == 'all_comp':
        bvp = np.reshape(bvp, (l * c, f))  # compact into 2D matrix
    elif kargs['component'] == 'second_comp':
        bvp = np.reshape(bvp[:, 1, :], (l, f))

    # collect
    return bvp


def cpu_SSR(raw_signal, **kargs):
    """
    SSR method on CPU using Numpy.

    'raw_signal' is a float32 ndarray with shape [num_frames, rows, columns, rgb_channels]; it can be obtained by
    using the :py:class:‵pyVHR.extraction.sig_processing.SignalProcessing‵ class ('extract_raw_holistic' method).

    The dictionary parameters are: {'fps':float}.

    Wang, W., Stuijk, S., & De Haan, G. (2015). A novel algorithm for remote photoplethysmography: Spatial subspace rotation. IEEE transactions on biomedical engineering, 63(9), 1974-1984.
    """

    # utils functions #
    def __build_p(τ, k, l, U, Λ):
        """
        builds P
        Parameters
        ----------
        k: int
            The frame index
        l: int
            The temporal stride to use
        U: numpy.ndarray
            The eigenvectors of the c matrix (for all frames up to counter).
        Λ: numpy.ndarray
            The eigenvalues of the c matrix (for all frames up to counter).
        Returns
        -------
        p: numpy.ndarray
            The p signal to add to the pulse.
        """
        # SR'
        SR = np.zeros((3, l), np.float32)  # dim: 3xl
        z = 0

        for t in range(τ, k, 1):  # 6, 7
            a = Λ[0, t]
            b = Λ[1, τ]
            c = Λ[2, τ]
            d = U[:, 0, t].T
            e = U[:, 1, τ]
            f = U[:, 2, τ]
            g = U[:, 1, τ].T
            h = U[:, 2, τ].T
            x1 = a / b
            x2 = a / c
            x3 = np.outer(e, g)
            x4 = np.dot(d, x3)
            x5 = np.outer(f, h)
            x6 = np.dot(d, x5)
            x7 = np.sqrt(x1)
            x8 = np.sqrt(x2)
            x9 = x7 * x4
            x10 = x8 * x6
            x11 = x9 + x10
            SR[:, z] = x11  # 8 | dim: 3
            z += 1

        # build p and add it to the final pulse signal
        s0 = SR[0, :]  # dim: l
        s1 = SR[1, :]  # dim: l
        p = s0 - ((np.std(s0) / np.std(s1)) * s1)  # 10 | dim: l
        p = p - np.mean(p)  # 11
        return p  # dim: l

    def __build_correlation_matrix(V):
        # V dim: (W×H)x3
        # V = np.unique(V, axis=0)
        V_T = V.T  # dim: 3x(W×H)
        N = V.shape[0]
        # build the correlation matrix
        C = np.dot(V_T, V)  # dim: 3x3
        C = C / N

        return C

    def __eigs(C):
        """
        get eigenvalues and eigenvectors, sort them.
        Parameters
        ----------
        C: numpy.ndarray
            The RGB values of skin-colored pixels.
        Returns
        -------
        Λ: numpy.ndarray
            The eigenvalues of the correlation matrix
        U: numpy.ndarray
            The (sorted) eigenvectors of the correlation matrix
        """
        # get eigenvectors and sort them according to eigenvalues (largest first)
        L, U = np.linalg.eig(C)  # dim Λ: 3 | dim U: 3x3
        idx = L.argsort()  # dim: 3x1
        idx = idx[::-1]  # dim: 1x3
        L_ = L[idx]  # dim: 3
        U_ = U[:, idx]  # dim: 3x3

        return L_, U_

    # ----------------------------------- #

    fps = int(kargs['fps'])

    raw_sig = raw_signal
    K = len(raw_sig)
    l = int(fps)

    P = np.zeros(K)  # 1 | dim: K
    # store the eigenvalues Λ and the eigenvectors U at each frame
    L = np.zeros((3, K), dtype=np.float32)  # dim: 3xK
    U = np.zeros((3, 3, K), dtype=np.float32)  # dim: 3x3xK

    for k in range(K):
        n_roi = len(raw_sig[k])
        VV = []
        V = raw_sig[k].astype(np.float32)
        idx = V != 0
        idx2 = np.logical_and(np.logical_and(idx[:, :, 0], idx[:, :, 1]), idx[:, :, 2])
        V_skin_only = V[idx2]
        VV.append(V_skin_only)

        VV = np.vstack(VV)

        C = __build_correlation_matrix(VV)  # dim: 3x3

        # get: eigenvalues Λ, eigenvectors U
        L[:, k], U[:, :, k] = __eigs(C)  # dim Λ: 3 | dim U: 3x3

        # build p and add it to the pulse signal P
        if k >= l:  # 5
            tau = k - l  # 5
            p = __build_p(tau, k, l, U, L)  # 6, 7, 8, 9, 10, 11 | dim: l
            P[tau:k] += p  # 11

        if np.isnan(np.sum(P)):
            print('NAN')
            print(raw_sig[k])

    bvp = P
    bvp = np.expand_dims(bvp, axis=0)
    return bvp


def jadeR(X, m=None, verbose=True):
    """
    Blind separation of real signals with JADE.
    jadeR implements JADE, an Independent Component Analysis (ICA) algorithm
    developed by Jean-Francois Cardoso. See http://www.tsi.enst.fr/~cardoso/guidesepsou.html , and papers cited
    at the end of the source file.

    Translated into NumPy from the original Matlab Version 1.8 (May 2005) by
    Gabriel Beckers, http://gbeckers.nl .

    Parameters:
        X -- an nxT data matrix (n sensors, T samples). May be a numpy array or
             matrix.
        m -- output matrix B has size mxn so that only m sources are
             extracted.  This is done by restricting the operation of jadeR
             to the m first principal components. Defaults to None, in which
             case m=n.
        verbose -- print info on progress. Default is True.

    Returns:
        An m*n matrix B (NumPy matrix type), such that Y=B*X are separated
        sources extracted from the n*T data matrix X. If m is omitted, B is a
        square n*n matrix (as many sources as sensors). The rows of B are
        ordered such that the columns of pinv(B) are in order of decreasing
        norm; this has the effect that the `most energetically significant`
        components appear first in the rows of Y=B*X.

    Quick notes (more at the end of this file):
    o This code is for REAL-valued signals.  A MATLAB implementation of JADE
    for both real and complex signals is also available from
    http://sig.enst.fr/~cardoso/stuff.html
    o This algorithm differs from the first released implementations of
    JADE in that it has been optimized to deal more efficiently
    1) with real signals (as opposed to complex)
    2) with the case when the ICA model does not necessarily hold.
    o There is a practical limit to the number of independent
    components that can be extracted with this implementation.  Note
    that the first step of JADE amounts to a PCA with dimensionality
    reduction from n to m (which defaults to n).  In practice m
    cannot be `very large` (more than 40, 50, 60... depending on
    available memory)
    o See more notes, references and revision history at the end of
    this file and more stuff on the WEB
    http://sig.enst.fr/~cardoso/stuff.html
    o For more info on NumPy translation, see the end of this file.
    o This code is supposed to do a good job!  Please report any
    problem relating to the NumPY code gabriel@gbeckers.nl

    Copyright original Matlab code : Jean-Francois Cardoso <cardoso@sig.enst.fr>
    Copyright Numpy translation : Gabriel Beckers <gabriel@gbeckers.nl>
    """

    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for X.

    assert isinstance(X, ndarray), \
        "X (input data matrix) is of the wrong type (%s)" % type(X)
    origtype = X.dtype  # remember to return matrix B of the same type
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert (verbose == True) or (verbose == False), \
        "verbose parameter should be either True or False"

    [n, T] = X.shape  # GB: n is number of input signals, T is number of samples

    if m == None:
        m = n  # Number of sources defaults to # of sensors
    assert m <= n, \
        "jade -> Do not ask more sources (%d) than sensors (%d )here!!!" % (m, n)

    if verbose:
        print("jade -> Looking for %d sources" % m)
        print("jade -> Removing the mean value")
    X -= X.mean(1)

    # whitening & projection onto signal subspace
    # ===========================================
    if verbose:
        print("jade -> Whitening the data")
    [D, U] = np.linalg.eig((X * X.T) / float(T))  # An eigen basis for the sample covariance matrix
    k = D.argsort()
    Ds = D[k]  # Sort by increasing variances
    PCs = arange(n - 1, n - m - 1, -1)  # The m most significant princip. comp. by decreasing variance

    # --- PCA  ----------------------------------------------------------
    B = U[:, k[PCs]].T  # % At this stage, B does the PCA on m components

    # --- Scaling  ------------------------------------------------------
    scales = sqrt(Ds[PCs])  # The scales of the principal components .
    B = diag(1. / scales) * B  # Now, B does PCA followed by a rescaling = sphering
    # B[-1,:] = -B[-1,:] # GB: to make it compatible with octave
    # --- Sphering ------------------------------------------------------
    X = B * X  # %% We have done the easy part: B is a whitening matrix and X is white.

    del U, D, Ds, k, PCs, scales

    # NOTE: At this stage, X is a PCA analysis in m components of the real data, except that
    # all its entries now have unit variance.  Any further rotation of X will preserve the
    # property that X is a vector of uncorrelated components.  It remains to find the
    # rotation matrix such that the entries of X are not only uncorrelated but also `as
    # independent as possible".  This independence is measured by correlations of order
    # higher than 2.  We have defined such a measure of independence which
    #   1) is a reasonable approximation of the mutual information
    #   2) can be optimized by a `fast algorithm"
    # This measure of independence also corresponds to the `diagonality" of a set of
    # cumulant matrices.  The code below finds the `missing rotation " as the matrix which
    # best diagonalizes a particular set of cumulant matrices.

    # Estimation of the cumulant matrices.
    # ====================================
    if verbose:
        print("jade -> Estimating cumulant matrices")

    # Reshaping of the data, hoping to speed up things a little bit...
    X = X.T
    dimsymm = int((m * (m + 1)) / 2)  # Dim. of the space of real symm matrices
    nbcm = dimsymm  # number of cumulant matrices
    CM = matrix(zeros([m, m * nbcm], dtype=float64))  # Storage for cumulant matrices
    R = matrix(eye(m, dtype=float64))
    Qij = matrix(zeros([m, m], dtype=float64))  # Temp for a cum. matrix
    Xim = zeros(m, dtype=float64)  # Temp
    Xijm = zeros(m, dtype=float64)  # Temp
    # Uns = numpy.ones([1,m], dtype=numpy.uint32)    # for convenience
    # GB: we don't translate that one because NumPy doesn't need Tony's rule

    # I am using a symmetry trick to save storage.  I should write a short note one of these
    # days explaining what is going on here.
    Range = arange(m)  # will index the columns of CM where to store the cum. mats.

    for im in range(m):
        Xim = X[:, im]
        Xijm = multiply(Xim, Xim)
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        Qij = multiply(Xijm, X).T * X / float(T) \
              - R - 2 * dot(R[:, im], R[:, im].T)
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = multiply(Xim, X[:, jm])
            Qij = sqrt(2) * multiply(Xijm, X).T * X / float(T) \
                  - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T
            CM[:, Range] = Qij
            Range = Range + m

    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big m x m*nbcm array.

    V = matrix(eye(m, dtype=float64))

    Diag = zeros(m, dtype=float64)
    On = 0.0
    Range = arange(m)
    for im in range(nbcm):
        Diag = diag(CM[:, Range])
        On = On + (Diag * Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM, CM).sum(axis=0)).sum(axis=0) - On

    seuil = 1.0e-6 / sqrt(T)  # % A statistically scaled threshold on `small" angles
    encore = True
    sweep = 0  # % sweep number
    updates = 0  # % Total number of rotations
    upds = 0  # % Number of rotations in a given seep
    g = zeros([2, nbcm], dtype=float64)
    gg = zeros([2, 2], dtype=float64)
    G = zeros([2, 2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    # Joint diagonalization proper

    if verbose:
        print("jade -> Contrast optimization by joint diagonalization")

    while encore:
        encore = False
        if verbose:
            print("jade -> Sweep #%3d" % sweep)
        sweep = sweep + 1
        upds = 0
        Vkeep = V

        for p in range(m - 1):
            for q in range(p + 1, m):

                Ip = arange(p, m * nbcm, m)
                Iq = arange(q, m * nbcm, m)

                # computation of Givens angle
                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0

                # Givens update
                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    G = matrix([[c, -s], [s, c]])
                    pair = array([p, q])
                    V[:, pair] = V[:, pair] * G
                    CM[pair, :] = G.T * CM[pair, :]
                    CM[:, concatenate([Ip, Iq])] = \
                        append(c * CM[:, Ip] + s * CM[:, Iq], -s * CM[:, Ip] + c * CM[:, Iq], \
                               axis=1)
                    On = On + Gain
                    Off = Off - Gain

        if verbose:
            print("completed in %d rotations" % upds)
        updates = updates + upds
    if verbose:
        print("jade -> Total of %d Givens rotations" % updates)

    # A separating matrix
    # ===================

    B = V.T * B

    # Permute the rows of the separating matrix B to get the most energetic components first.
    # Here the **signals** are normalized to unit variance.  Therefore, the sort is
    # according to the norm of the columns of A = pinv(B)

    if verbose:
        print("jade -> Sorting the components")

    A = np.linalg.pinv(B)
    keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]
    B = B[keys, :]
    B = B[::-1, :]  # % Is this smart ?

    if verbose:
        print("jade -> Fixing the signs")
    b = B[:, 0]
    signs = array(sign(sign(b) + 0.1).T)[0]  # just a trick to deal with sign=0
    B = diag(signs) * B

    return B.astype(origtype)
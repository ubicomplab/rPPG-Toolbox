# The Independent Component Analysis (ICA) Method from: Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). 
# Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774. 
# DOI: 10.1364/OE.18.010762
import numpy as np
import math
from scipy import signal
from scipy import linalg
from signal_methods import utils
from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float32,float64, matrix, multiply, ndarray, newaxis, \
    sign, sin, sqrt, zeros
import numpy as np

def ICA_POH(frames, FS):
    """
    ICA method on CPU using Numpy.
    The dictionary parameters are {'component':str}. Where 'component' can be 'second_comp' or 'all_comp'.
    Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.
    """
    bvp = []
    signal = utils.process_video(frames)
    print(signal.shape)
    for X in signal:
        W = jadeR(X, verbose=False)
        bvp.append(np.dot(W, X))

    # selector
    bvp = np.array(bvp).reshape(-1)
    l, c, f = bvp.shape  # l=#landmks c=#3chs, f=#frames
    # if kargs['component'] == 'all_comp':
    #     bvp = np.reshape(bvp, (l * c, f))  # compact into 2D matrix
    # elif kargs['component'] == 'second_comp':
    #     bvp = np.reshape(bvp[:, 1, :], (l, f))

    # collect
    return bvp








def ICA_POH2(frames, FS):
    # Cut off frequency.
    LPF = 0.7
    HPF = 2.5
    RGB = process_video(frames)

    NyquistF = 1/2*FS
    BGRNorm = np.zeros(RGB.shape)
    Lambda = 100
    for c in range(3):
        BGRDetrend = utils.detrend(RGB[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend-np.mean(BGRDetrend))/np.std(BGRDetrend)
    _, S = ica(np.mat(BGRNorm).H, 3)

    # select BVP Source
    MaxPx = np.zeros((1, 3))
    for c in range(3):
        FF = np.fft.fft(S[c, :])
        F = np.arange(0, FF.shape[1])/FF.shape[1]*FS*60
        FF = FF[:, 1:]
        FF = FF[0]
        N = FF.shape[0]
        Px = np.abs(FF[:math.floor(N/2)])
        Px = np.multiply(Px, Px)
        Fx = np.arange(0, N/2)/(N/2)*NyquistF
        Px = Px/np.sum(Px, axis=0)
        MaxPx[0, c] = np.max(Px)  
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')
    BVP_F = signal.filtfilt(B, A, BVP_I.astype(np.double))

    BVP = BVP_F[0]
    return BVP


def process_video(frames):
    "Calculates the average value of each frame."
    RGB = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(sum/(frame.shape[0]*frame.shape[1]))
    return np.asarray(RGB)


def ica(X, Nsources, Wprev=0):
    nRows = X.shape[0]
    nCols = X.shape[1]
    if nRows > nCols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', Nsources)

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat


def jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1/math.sqrt(T)/100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H)/T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n-m:n]-np.mean(pu[0:n-m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n-m:n]]))
        IW = np.matmul(U[0:n, k[n-m:n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H)/T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T)/T
    Q = np.zeros((m*m*m*m, 1))
    index = 0

    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1/math.sqrt(T), Y[ix, :].T/math.sqrt(
                        T))-R[ix, jx]*R[lx, kx]-R[ix, kx]*R[lx, jx]-C[ix, lx]*np.conj(C[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m*m, m*m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem*m), dtype=complex)
    Z = np.zeros(m)
    h = m*m-1
    for u in range(0, nem*m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u+m] = la[h]*Z
        h = h-1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0-1j, 0+1j]])
    Bt = np.mat(B).H

    encore = 1
    if(Wprev == 0):
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:  
        encore = 0
        for p in range(m-1):
            for q in range(p+1, m):
                Ip = np.arange(p, nem*m, m)
                Iq = np.arange(q, nem*m, m)
                g = np.mat([M[p, Ip]-M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]
                if(angles[0, 0] < 0):
                    angles = -angles
                c = np.sqrt(0.5+angles[0, 0]/2)
                s = 0.5*(angles[1, 0]-1j*angles[2, 0])/c

                if(abs(s) > seuil):
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)], [s, c]])  # Gavins旋转矩阵
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c*M[:, Ip]+s*M[:, Iq]
                    temp2 = -np.conj(s)*M[:, Ip]+c*M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)
    return A, S


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
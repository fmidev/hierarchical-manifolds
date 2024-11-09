"""
FILE:                   algorithms.py
COPYRIGHT:              (c) 2024 Finnish Meteorological Institute
                        P.O. BOX 503
                        FI-00101 Helsinki, Finland
                        https://www.fmi.fi/
LICENCE:                MIT
AUTHOR:                 Terhi Mäkinen (terhi.makinen@fmi.fi)
DESCRIPTION:

This module provides miscellaneous low-level algorithms used in Hierarchical
Data Analysis.

The LOBPCG implementation is based on the paper A.V. Knyazev, 2001. Toward the
Optimal Preconditioned Eigensolver: Locally Optimal Block Preconditioned
Conjugate Gradient Method. SIAM Journal on Scientific Computing 23 (2), 517–541,
doi:10.1137/S1064827500366124.
"""

import math
import numpy as np
from numpy import matmul as mm
from numpy.linalg import eig, eigh, pinv
from scipy.stats import norm
from scipy.special import digamma, gammaln
from scipy.optimize import brent, root_scalar
from sklearn.mixture import GaussianMixture as GMM

def bracket_minimum(f, x, fx=None, arg=None, nit=32, ftol=1e-4):
    """Find a triplet of points bracketing a minimum by golden section search.

    Parameters
    ----------
    f : callable object
        function to be minimized
    x : tuple(2) of float
        initial bounds
    fx : tuple(2) of float
        function evaluations at x
    nit : int
        maximal number of iterations
    arg : untyped
        additional arguments passed to f
    ftol : float
        floating point tolerance

    Returns
    -------
    tuple(3)
        tuple(3)
            bracketing triplet
        float
            minimal function value
        int
            number of iterations, or
    None
        if no minimum is found in the given range
    """
    assert callable(f)
    assert (isinstance(x, (tuple, list)) and
            (len(x) == 2) and
            isinstance(x[0], (int, float)) and
            isinstance(x[1], (int, float)) and
            (x[0] < x[1])
           )
    assert ((fx is None) or
            (isinstance(x, (tuple, list)) and
             (len(x) == 2) and
             isinstance(x[0], (int, float)) and
             isinstance(x[1], (int, float))
            )
           )
    assert isinstance(nit, int) and (nit > 0)

    if (x[1] - x[0] < ftol):
        return None
    if (arg is None):
        if (fx is None):
            fa = f(x[0])
            fb = f(x[1])
        else:
            fa, fb = fx
        if (fa < fb):
            xs = 0.618 * x[0] + 0.382 * x[1]
            fs = f(xs)
            if (fs < fa):
                return ((x[0], xs, x[1]), fs, nit)
        else:
            xs = 0.382 * x[0] + 0.618 * x[1]
            fs = f(xs)
            if (fs < fb):
                return ((x[0], xs, x[1]), fs, nit)
        if (nit < 2):
            return None
        r = bracket_minimum(f, (x[0],xs), (fa,fs), nit=nit-1, ftol=ftol)
        if (r is None):
            return bracket_minimum(f, (xs,x[1]), (fs,fb), nit=nit-1, ftol=ftol)
        else:
            return r
    else:
        if (fx is None):
            fa = f(x[0], arg)
            fb = f(x[1], arg)
        else:
            fa, fb = fx
        if (fa < fb):
            xs = 0.618 * x[0] + 0.382 * x[1]
            fs = f(xs, arg)
            if (fs < fa):
                return ((x[0], xs, x[1]), fs, nit)
        else:
            xs = 0.382 * x[0] + 0.618 * x[1]
            fs = f(xs, arg)
            if (fs < fb):
                return ((x[0], xs, x[1]), fs, nit)
        if (nit < 2):
            return None
        r = bracket_minimum(f, (x[0],xs), (fa,fs), arg=arg, nit=nit-1,
            ftol=ftol)
        if (r is None):
            return bracket_minimum(f, (xs,x[1]), (fs,fb), arg=arg, nit=nit-1,
                ftol=ftol)
        else:
            return r

def dirichlet_entropy(x):
    """Calculate entropy-based aleatoric and epistemic uncertainties.

    Parameters
    ----------
    x : ndarray(K) or ndarray(N,K)
        N instances of K-dimensional Dirichlet parameter vectors

    Returns
    -------
    ndarray(2) or ndarray(N,2)
        respective aleatoric and epistemic uncertainties
    """
    assert isinstance(x, np.ndarray) and (x.ndim < 3)

    if (x.ndim == 1):
        a = x.reshape((1,x.size))
    else:
        a = x
    N, K = a.shape
    a0 = np.sum(a, axis=1).flatten()
    u = np.zeros((N,2))
    for i in range(N):
        u[i,0] = -(np.sum(a[i,:] * (digamma(a[i,:] + 1) - digamma(a0[i] + 1))) /
            a0[i])
        u[i,1] = (np.sum(gammaln(a[i,:])) - gammaln(a0[i]) -
            np.sum((a[i,:] - 1) * (digamma(a[i,:]) - digamma(a0[i]))))
    if (x.ndim == 1):
        return u.flatten()
    return u

def dirichlet_uncertainty(x):
    """Calculate probability-based aleatoric and epistemic uncertainties.

    Parameters
    ----------
    x : ndarray(K) or ndarray(N,K)
        N instances of K-dimensional Dirichlet parameter vectors

    Returns
    -------
    ndarray(2) or ndarray(N,2)
        respective aleatoric and epistemic uncertainties
    """
    assert isinstance(x, np.ndarray) and (x.ndim < 3)

    if (x.ndim == 1):
        a = x.reshape((1,x.size))
    else:
        a = x
    N, K = a.shape
    a0 = np.sum(a, axis=1).reshape((N,1))
    p = a / a0
    q = 1 - p
    u = np.zeros((N,2))
    u[:,0] = 1 - np.sum(p**2, axis=1)
    u[:,1] = np.sum(p * q / (a0 + 1), axis=1)
    if (x.ndim == 1):
        return u.flatten()
    return u

def dirichlet_loss(x, prior=1.0):
    """Calculate the loss function for a sample from the Dirichlet distribution.

    Parameters
    ----------
    x : ndarray(K)
        sample measurement vector
    prior : float
        Bayesian prior

    Returns
    -------
    float
        loss
    """
    assert (isinstance(x, np.ndarray) and
            (x.ndim == 1) and
            (x.size > 1) and
            (np.amin(x) >= 0)
           )
    assert isinstance(prior, (int, float)) and (prior >= 0)

    a = x + prior
    a0 = np.sum(a)
    p = a / a0
    q = 1 - p
    return np.sum(p*q**2) + np.sum(p * q) / (a0 + 1)

def dirichlet_partition_loss(x, r, prior=1.0):
    """Calculate the partition gain function for a sample from the Dirichlet
    distribution and a partitioning.

    Parameters
    ----------
    x : ndarray(K)
        sample measurement vector
    r : ndarray(K)
        partitioning vector
    prior : float
        Bayesian prior

    Returns
    -------
    float
        differential gain between unpartitioned and partitioned data
    """
    assert (isinstance(x, np.ndarray) and
            (x.ndim == 1) and
            (x.size > 1) and
            (np.amin(x) >= 0)
           )
    assert (isinstance(r, np.ndarray) and
            (r.shape == x.shape) and
            (np.amin(r) >= 0) and
            (np.amax(r) <= 1)
           )
    assert isinstance(prior, (int, float)) and (prior >= 0)

    ax = r * x
    bx = (1 - r) * x
    c = np.sum(x)
    Pa = np.sum(ax) / c
    Pb = np.sum(bx) / c

    p = (x + prior) / (np.sum(x + prior))
    pa = (ax + prior) / (np.sum(ax + prior))
    pb = (bx + prior) / (np.sum(bx + prior))
    bias = (np.sum(ax * (p - pa)**2) + np.sum(bx * (p - pb)**2)) / c

    return (Pa * dirichlet_loss(ax, prior=prior) +
            Pb * dirichlet_loss(bx, prior=prior) -
            dirichlet_loss(x, prior=prior) - bias)

def split_loss_cb(x, C):
    """Callback function for minimizing the partitioning Dirichlet loss.

    Parameters
    ----------
    x : float
        partitioning pivot
    C : tuple(4)
        ndarray(K)
            class weights
        ndarray(K)
            1 / sigma
        ndarray(K)
            - mu / sigma
        float
            prior

    Returns
    -------
    float
        partitioning loss
    """
    return dirichlet_partition_loss(C[0], norm.cdf(C[1] * x + C[2]), prior=C[3])

def min_loss_offset(cn, cw, mu, s2, pr, force=False):
    """Determine the offset that minimizes Dirichlet partition loss for a set
    of classes with given normal distributions.

    Parameters
    ----------
    cn : ndarray(K)
        number of points per class
    cw : ndarray(K)
        class weights
    mu : ndarray(K)
        class means
    s2 : ndarray(K)
        class variances
    pr : float
        prior
    force : bool
        if True, always return a solution for multiclass data

    Returns
    -------
    tuple(3)
        float
            offset
        float
            loss
        ndarray(K)
            partition ratios
    """
    K = cn.size
    rx = np.argsort(mu)
    rn = cn[rx]
    rm = mu[rx]
    rs = s2[rx]
    rw = cw[rx]
    vx = (rs > 0)
    ru = np.zeros(K)
    ru[vx] = 1 / rs[vx]**0.5
    C = (rw, ru, -rm * ru, pr)
    ux = (rn > 0)
    Ku = np.count_nonzero(ux)
    if (Ku < 2):
        return (None, None, None)
    if (Ku == K):
        x = rm
    else:
        x = rm[ux]
    L = np.zeros(Ku)
    xmin = x[0]
    Lmin = L[0] = split_loss_cb(x[0], C)
    for k in range(1,Ku):
        L[k] = split_loss_cb(x[k], C)
        if (L[k] < Lmin):
            xmin = x[k]
            Lmin = L[k]
    inside_min = False
    for k in range(Ku-1):
        bk = bracket_minimum(split_loss_cb, (x[k], x[k+1]), arg=C, nit=3)
        if (bk is not None):
            xx, lx, nit, nfc = brent(split_loss_cb, args=(C,), brack=bk[0],
                full_output=True)
            if (lx < Lmin):
                inside_min = True
                xmin = xx
                Lmin = lx
    if (not inside_min):
        if force:
            xmin = np.sum(cw * mu) / np.sum(cw)
            Lmin = split_loss_cb(xmin, C)
        else:
            return (None, None, None)
    q = norm.cdf(C[1] * xmin + C[2])
    qx = np.zeros(K) + 0.5
    qx[rx] = q
    return (xmin, Lmin, qx)

def eig_matrix(S, num=0):
    """Solve the standard eigenvalue problem S x = lambda x.

    This version uses explicit covariance matrix and is thus not suitable for
    high-dimensional data.

    Parameters
    ----------
    S : ndarray(D,D)
        covariance matrix
    num  : int
        number of returned eigenvectors
        if num is zero return all eigenvectors in order of largest to smallest
        eigenvalue
        if num is larger than zero return num eigenvectors in order of largest
        to smallest eigenvalue
        if num is smaller than zero return num eigenvectors in order of smallest
        to largest eigenvalue

    Returns
    -------
    tuple(2)
        ndarray(num)
            eigenvalues in decreasing order
        ndarray(D, num)
            respective eigenvectors
    """
    assert (isinstance(S, np.ndarray) and
            (S.ndim == 2) and
            (S.shape[0] == S.shape[1])
           )
    assert isinstance(num, int) and (num >= -S.shape[0]) and (num <= S.shape[0])

    D = S.shape[0]
    if (num == 0):
        n = D
        c = -1.0
    elif (num > 0):
        n = num
        c = -1.0
    else:
        n = -num
        c = 1.0
    if (not S.any()):
        return (np.zeros(n), np.zeros((D,n)))
    w, P = eigh(S)
    wx = np.argsort(c * w.real)
    w = w[wx].real
    P = P[:,wx].reshape((D,D)).real
    if (n == D):
        return (w, P)
    return (w[0:n], P[:,0:n].reshape((D,n)))

def eig_lopcg1(xm, x0, reg=0, reverse=False, T=None, niter=100, ftol=1e-8):
    """Solve the standard eigenvalue problem S x = lambda x.

    Solve for the largest eigenvalue lambda by using the single component LOBPCG
    algorithm (Alg. 4.1) in Knyazev, 2001.

    Parameters
    ----------
    xm : ndarray(N,D)
        data
    x0 : ndarray(D,1)
        initial eigenvector
    reg : float
        optional regularization coefficient
    reverse : bool
        if set, return the smallest eigenvalue instead
    T : ndarray(D,1)
        preconditioner, or None
    niter : int
        maximal number of iterations, or if list(1) of int will contain the
        actual number of iterations on return
    ftol : float
        floating point tolerance

    Returns
    -------
    tuple
        float
            lambda
        ndarray(D,1)
            x
    """
    assert (isinstance(xm, np.ndarray) and
            (xm.ndim == 2)
           )
    assert (isinstance(x0, np.ndarray) and
            (x0.ndim == 2) and
            (x0.shape[0] == xm.shape[1]) and
            (x0.shape[1] == 1)
           )
    assert isinstance(reg, (int, float)) and (reg >= 0)
    assert isinstance(reverse, bool)
    assert ((T is None) or
            (isinstance(T, np.ndarray) and
             (T.ndim == 2) and
             (T.shape[0] == xm.shape[1]) and
             (T.shape[1] == 1)
            )
           )
    assert ((isinstance(niter, int) and
             (niter > 0)
            ) or
            (isinstance(niter, list) and
             isinstance(niter[0], int) and
             (niter[0] > 0)
            )
           )
    assert isinstance(ftol, float) and (ftol > 0)

    N, D = xm.shape
    if (not xm.any()):
        return (1.0, x0)
    W = N - 1
    if (W < 1):
        W = 1
    gam = reg / (W + reg)

    # V is the (D,3) Rayleigh-Ritz projection matrix
    # x, p and w are defined as views into V

    V = np.random.randn(D*3).reshape((D,3))
    V /= np.sum(V**2, axis=0).reshape((1,3))**0.5
    V[:,0] = x0[:,0]
    x = V[:,0].reshape((D,1))
    p = V[:,1].reshape((D,1))
    w = V[:,2].reshape((D,1))

    if isinstance(niter, list):
        nr = niter[0]
    else:
        nr = niter
    for i in range(nr):

        # find the Rayleigh quotient and calculate the residual; if it is
        # sufficiently small exit the loop

        Sx = mm(xm.T, mm(xm, x))
        if (reg == 0):
            lam = np.sum(x * Sx) / np.sum(x**2)
            r = Sx - lam * x
        else:
            q = (1 - gam) * np.sum(x * Sx) + gam * W
            lam = (np.sum(x**2) + q) / q
            r = (lam - 1) * ((1 - gam) * Sx + gam * W * x) - x
        dr = np.sum(r**2)**0.5
        if (dr < ftol):
            if isinstance(niter, list):
                niter[0] = i
            break

        # optionally apply conditioning to the residual

        if (T is None):
            w[:] = r
        else:
            w[:] = T * r

        # orthonormalize V and solve the Rayleigh-Ritz approximation

        for j in range(3):
            while True:
                for k in range(j):
                    V[:,j] -= np.sum(V[:,j] * V[:,k]) * V[:,k]
                dv = np.sum(V[:,j]**2)
                if (dv > ftol):
                    V[:,j] /= dv**0.5
                    break
                V[:,j] = np.random.randn(D)
        vm = mm(xm, V)
        if reverse:
            rrl, rrp = eig_matrix(mm(vm.T, vm), num=-1)
        else:
            rrl, rrp = eig_matrix(mm(vm.T, vm), num=1)

        #  update the projection matrix

        x[:] = mm(V, rrp)
        rrp[0,0] = 0
        p[:] = mm(V, rrp)

    return (lam, x / np.sum(x**2)**0.5)

def eig_lobpcg(xm, x0, reg=0, reverse=False, T=None, niter=100, ftol=1e-8):
    """Solve the standard eigenvalue problem S x = lambda x.

    Solve for the M largest eigenvalues lambda by using the LOBPCG algorithm
    (Alg. 5.1) in Knyazev, 2001.

    Parameters
    ----------
    xm : ndarray(N,D)
        data
    x0 : ndarray(D,M)
        initial eigenvectors
    reg : float
        optional regularization coefficient
    reverse : bool
        if set, return the smallest eigenvalues instead
    T : ndarray(D,1)
        preconditioner, or None
    niter : int
        maximal number of iterations, or if list(1) of int will contain the
        actual number of iterations on return
    ftol : float
        floating point tolerance

    Returns
    -------
    tuple
        ndarray(M)
            lambda
        ndarray(D,M)
            x
    """
    assert (isinstance(xm, np.ndarray) and
            (xm.ndim == 2)
           )
    assert (isinstance(x0, np.ndarray) and
            (x0.ndim == 2) and
            (x0.shape[0] == xm.shape[1]) and
            (x0.shape[1] >= 1)
           )
    assert isinstance(reg, (int, float)) and (reg >= 0)
    assert isinstance(reverse, bool)
    assert ((T is None) or
            (isinstance(T, np.ndarray) and
             (T.ndim == 2) and
             (T.shape[0] == xm.shape[1]) and
             (T.shape[1] == 1)
            )
           )
    assert ((isinstance(niter, int) and
             (niter > 0)
            ) or
            (isinstance(niter, list) and
             isinstance(niter[0], int) and
             (niter[0] > 0)
            )
           )
    assert isinstance(ftol, float) and (ftol > 0)

    N, D = xm.shape
    M = x0.shape[1]
    if (not xm.any()):
        return (1.0, x0)
    W = N - 1
    if (W < 1):
        W = 1
    gam = reg / (W + reg)

    # V is the (D,3*M) Rayleigh-Ritz projection matrix
    # x, p and w are defined as views into V

    M2 = 2 * M
    M3 = 3 * M
    V = np.random.randn(D*M3).reshape((D,M3))
    V /= np.sum(V**2, axis=0).reshape((1,M3))**0.5
    V[:,0:M] = x0[:,0:M]
    x = V[:,0:M].reshape((D,M))
    p = V[:,M:M2].reshape((D,M))
    w = V[:,M2:M3].reshape((D,M))

    if isinstance(niter, list):
        nr = niter[0]
    else:
        nr = niter
    for i in range(nr):

        # find the Rayleigh quotients and calculate the residuals; if they are
        # sufficiently small exit the loop

        Sx = mm(xm.T, mm(xm, x))
        if (reg == 0):
            lam = np.sum(x * Sx, axis=0) / np.sum(x**2, axis=0)
            r = Sx - lam.reshape((1,M)) * x 
        else:
            q = (1 - gam) * np.sum(x * Sx, axis=0) + gam * W
            lam = (np.sum(x**2, axis=0) + q) / q
            r = (lam.reshape((1,M)) - 1) * ((1 - gam) * Sx + gam * W * x) - x
        dr = np.sum(r**2, axis=0)**0.5
        if (np.amax(dr) < ftol):
            if isinstance(niter, list):
                niter[0] = i
            break

        # optionally apply conditioning to the residual

        if (T is None):
            w[:] = r
        else:
            w[:] = T * r

        # orthonormalize V and solve the Rayleigh-Ritz approximation

        for j in range(M3):
            while True:
                for k in range(j):
                    V[:,j] -= np.sum(V[:,j] * V[:,k]) * V[:,k]
                dv = np.sum(V[:,j]**2)
                if (dv > ftol):
                    V[:,j] /= dv**0.5
                    break
                V[:,j] = np.random.randn(D)
        vm = mm(xm, V)
        if reverse:
            rrl, rrp = eig_matrix(mm(vm.T, vm), num=-M)
        else:
            rrl, rrp = eig_matrix(mm(vm.T, vm), num=M)

        #  update the projection matrix

        x[:] = mm(V, rrp)
        rrp[0:M,:] = 0
        p[:] = mm(V, rrp)

    return (lam, x / np.sum(x**2, axis=0).reshape((1,M))**0.5)

def geig_matrix(Sb, Sw, r, num=0):
    """Solve the Linear Discriminant Analysis (LDA) specific regularized
    generalized eigenvalue problem (Sb + Rw) x = lambda Rw x with the applied
    Tikhonov regularization Rw = (1 - r) Sw + r 1.

    This version uses explicit covariance matrices and is thus not suitable
    for high-dimensional data.

    Parameters
    ----------
    Sb : ndarray(D,D)
        covariance matrix of class means
    Sw : ndarray(N,D)
        within-class covariance matrix
    r : float
        regularization coefficient; for a sample of N data points in K classes
        it should equal to w / (N - K + w) where w is the regularization factor
    num  : int
        number of returned eigenvectors
        if num is zero return all eigenvectors in order of largest to smallest
        eigenvalue
        if num is larger than zero return num eigenvectors in order of largest
        to smallest eigenvalue

    Returns
    -------
    tuple
        ndarray(num)
            eigenvalues in decreasing order
        ndarray(D, num)
            respective eigenvectors
    """
    assert (isinstance(Sb, np.ndarray) and
            (Sb.ndim == 2) and
            (Sb.shape[0] == Sb.shape[1])
           )
    assert (isinstance(Sw, np.ndarray) and
            (Sw.shape == Sb.shape)
           )
    assert isinstance(r, (int, float)) and (r >= 0) and (r <= 1)
    assert isinstance(num, int) and (num >= 0) and (num <= Sb.shape[0])

    D = Sb.shape[0]
    if (num == 0):
        n = D
    else:
        n = num
    if (not Sw.any()):
        return eig_matrix(Sb, num=n)
    Rw = (1 - r) * Sw + r * np.eye(D)
    if (not Sb.any()):
        return eig_matrix(Rw, num=-n)   # NB: return smallest of Rw
    w, P = eig(mm(pinv(Rw), Sb + Rw))   # NB: Sb + Rw is not a symmetric matrix
    wx = np.argsort(-w.real)
    w = w[wx].real
    P = P[:,wx].reshape((D,D)).real
    if (n == D):
        return (w, P)
    return (w[0:n], P[:,0:n].reshape((D,n)))

def geig_lopcg1(nb, xb, xw, x0, T=None, niter=100, ftol=1e-8, prior=1.0):
    """Solve the Linear Discriminant Analysis (LDA) specific regularized
    generalized eigenvalue problem (Sb + Rw) x = lambda Rw x for the largest
    eigenvalue lambda by using the single component LOBPCG algorithm (Alg. 4.1)
    in Knyazev, 2001, with the applied Tikhonov regularization
    Rw = (1 - gamma) Sw + gamma 1 and gamma = w / (N - K + w) for a sample of N
    data points in K classes and w is the regularization factor.

    Parameters
    ----------
    nb : ndarray(K)
        number of data points per class
    xb : ndarray(K,D)
        class means
    xw : ndarray(N,D)
        data with the respective class means removed
    x0 : ndarray(D,1)
        initial eigenvector
    T : ndarray(D,1)
        preconditioner, or None
    niter : int
        maximal number of iterations, or if list(1) of int will contain the
        actual number of iterations on return
    ftol : float
        floating point tolerance
    prior : float
        regularization factor

    Returns
    -------
    tuple(5)
        float
            (N - K) x^T Sb x
        float
            (N - K) x^T Sw x
        float
            (1 - gamma) b + gamma (N - K)
        ndarray(D,1)
            x
        ndarray(D,1)
            Sw x / ||Sw x||
    """
    assert (isinstance(nb, np.ndarray) and
            (nb.ndim == 1) and
            (nb.size >= 1)
           )
    assert (isinstance(xb, np.ndarray) and
            (xb.ndim == 2) and
            (xb.shape[0] == nb.size)
           )
    assert (isinstance(xw, np.ndarray) and
            (xw.ndim == 2) and
            (xw.shape[0] == np.sum(nb)) and
            (xw.shape[1] == xb.shape[1])
           )
    assert (isinstance(x0, np.ndarray) and
            (x0.ndim == 2) and
            (x0.shape[0] == xb.shape[1]) and
            (x0.shape[1] == 1)
           )
    assert ((T is None) or
            (isinstance(T, np.ndarray) and
             (T.ndim == 2) and
             (T.shape[0] == xw.shape[1]) and
             (T.shape[1] == 1)
            )
           )
    assert ((isinstance(niter, int) and
             (niter > 0)
            ) or
            (isinstance(niter, list) and
             isinstance(niter[0], int) and
             (niter[0] > 0)
            )
           )
    assert isinstance(ftol, float) and (ftol > 0)
    assert isinstance(prior, (int, float)) and (prior >= 0)

    K = nb.size
    N, D = xw.shape
    W = N - K
    if (W < 1):
        W = 1
    gam = prior / (W + prior)
    if (not xb.any()):
        b, x = eig_lopcg1(xw, x0, reg=prior, reverse=True, T=T, niter=niter,
            ftol=ftol)
        return (0, b - 1, (1 - gam) * (b - 1) + gam * W, x, np.zeros((D,1)))
    nxb = nb.reshape((K,1)) * xb
    if (not xw.any()):
        a, x = eig_lopcg1(nxb, x0, T=T, niter=niter, ftol=ftol)
        return (a, 0, 0, x, np.zeros((D,1)))

    # V is the (D,3) Rayleigh-Ritz projection matrix
    # x, p and w are defined as views into V

    V = np.random.randn(D*3).reshape((D,3))
    V /= np.sum(V**2, axis=0).reshape((1,3))**0.5
    V[:,0] = x0[:,0]
    x = V[:,0].reshape((D,1))
    p = V[:,1].reshape((D,1))
    w = V[:,2].reshape((D,1))

    if isinstance(niter, list):
        nr = niter[0]
    else:
        nr = niter
    for i in range(nr):

        # find the Rayleigh quotient and calculate the residual; if it is
        # sufficiently small exit the loop

        Ax = mm(xb.T, mm(nxb, x))
        a = np.sum(x * Ax)
        Bx = mm(xw.T, mm(xw, x))
        b = np.sum(x * Bx)
        q = (1 - gam) * b + gam * W
        lam = (a + q) / q
        r = (lam - 1) * ((1 - gam) * Bx + gam * W * x) - Ax
        dr = np.sum(r**2)**0.5
        if (dr < ftol):
            if isinstance(niter, list):
                niter[0] = i
            break

        # optionally apply conditioning to the residual

        if (T is None):
            w[:] = r
        else:
            w[:] = T * r

        # orthogonalize V and solve the Rayleigh-Ritz approximation

        for j in range(3):
            while True:
                for k in range(j):
                    V[:,j] -= np.sum(V[:,j] * V[:,k]) * V[:,k]
                dv = np.sum(V[:,j]**2)
                if (dv > ftol):
                    V[:,j] /= dv**0.5
                    break
                V[:,j] = np.random.randn(D)
        vb = mm(xb, V)
        nvb = mm(nxb, V)
        vw = mm(xw, V)
        rrl, rrp = geig_matrix(mm(vb.T, nvb) / W, mm(vw.T, vw) / W, gam, num=1)

        #  update the projection matrix

        x[:] = mm(V, rrp)
        rrp[0,0] = 0
        p[:] = mm(V, rrp)

    return (a, b, q, x / np.sum(x**2)**0.5, Bx / np.sum(Bx**2)**0.5)

def geig_lobpcg(nb, xb, xw, x0, T=None, niter=100, ftol=1e-8, prior=1.0):
    """Solve the Linear Discriminant Analysis (LDA) specific regularized
    generalized eigenvalue problem (Sb + Rw) x = lambda Rw x for the M largest
    eigenvalues lambda by using the LOBPCG algorithm (Alg. 5.1) in Knyazev, 2001
    with the applied Tikhonov regularization Rw = (1 - gamma) Sw + gamma 1 and
    gamma = w / (N - K + w) for a sample of N data points in K classes, and w is
    the regularization factor.

    Parameters
    ----------
    nb : ndarray(K)
        number of data points per class
    xb : ndarray(K,D)
        class means
    xw : ndarray(N,D)
        data with the respective class means removed
    x0 : ndarray(D,M)
        initial eigenvectors
    T : ndarray(D,1)
        preconditioner, or None
    niter : int
        maximal number of iterations, or if list(1) of int will contain the
        actual number of iterations on return
    ftol : float
        floating point tolerance
    prior : float
        regularization factor

    Returns
    -------
    tuple(5)
        float
            (N - K) x^T Sb x
        float
            (N - K) x^T Sw x
        float
            (1 - gamma) b + gamma (N - K)
        ndarray(D,M)
            x
        ndarray(D,M)
            Sw x / ||Sw x||
    """
    assert (isinstance(nb, np.ndarray) and
            (nb.ndim == 1) and
            (nb.size >= 1)
           )
    assert (isinstance(xb, np.ndarray) and
            (xb.ndim == 2) and
            (xb.shape[0] == nb.size)
           )
    assert (isinstance(xw, np.ndarray) and
            (xw.ndim == 2) and
            (xw.shape[0] == np.sum(nb)) and
            (xw.shape[1] == xb.shape[1])
           )
    assert (isinstance(x0, np.ndarray) and
            (x0.ndim == 2) and
            (x0.shape[0] == xb.shape[1]) and
            (x0.shape[1] >= 1)
           )
    assert ((T is None) or
            (isinstance(T, np.ndarray) and
             (T.ndim == 2) and
             (T.shape[0] == xw.shape[1]) and
             (T.shape[1] == 1)
            )
           )
    assert ((isinstance(niter, int) and
             (niter > 0)
            ) or
            (isinstance(niter, list) and
             isinstance(niter[0], int) and
             (niter[0] > 0)
            )
           )
    assert isinstance(ftol, float) and (ftol > 0)
    assert isinstance(prior, (int, float)) and (prior >= 0)

    K = nb.size
    N, D = xw.shape
    M = x0.shape[1]
    M2 = 2 * M
    M3 = 3 * M
    W = N - K
    if (W < 1):
        W = 1
    gam = prior / (W + prior)
    if (not xb.any()):
        b, x = eig_lobpcg(xw, x0, reg=prior, reverse=True, T=T, niter=niter,
            ftol=ftol)
        return (0, b - 1, (1 - gam) * (b - 1) + gam * W, x, np.zeros((D,1)))
    nxb = nb.reshape((K,1)) * xb
    if (not xw.any()):
        a, x = eig_lobpcg(nxb, x0, T=T, niter=niter, ftol=ftol)
        return (a, 0, 0, x, np.zeros((D,1)))

    # V is the (D,3*M) Rayleigh-Ritz projection matrix
    # x, p and w are defined as views into V

    V = np.random.randn(D*M3).reshape((D,M3))
    V /= np.sum(V**2, axis=0).reshape((1,M3))**0.5
    V[:,0:M] = x0[:,0:M]
    x = V[:,0:M].reshape((D,M))
    p = V[:,M:M2].reshape((D,M))
    w = V[:,M2:M3].reshape((D,M))

    if isinstance(niter, list):
        nr = niter[0]
    else:
        nr = niter
    for i in range(nr):

        # find the Rayleigh quotient and calculate the residual; if it is
        # sufficiently small exit the loop

        Ax = mm(xb.T, mm(nxb, x))
        a = np.sum(x * Ax, axis=0)
        Bx = mm(xw.T, mm(xw, x))
        b = np.sum(x * Bx, axis=0)
        q = (1 - gam) * b + gam * W
        lam = (a + q) / q
        r = (lam.reshape((1,M)) - 1) * ((1 - gam) * Bx + gam * W * x) - Ax
        dr = np.sum(r**2, axis=0)**0.5
        if (np.sum(dr) < M * ftol):
            if isinstance(niter, list):
                niter[0] = i
            break

        # optionally apply conditioning to the residual

        if (T is None):
            w[:] = r
        else:
            w[:] = T * r

        # solve the Rayleigh-Ritz approximation

        for j in range(M3):
            while True:
                for k in range(j):
                    V[:,j] -= np.sum(V[:,j] * V[:,k]) * V[:,k]
                dv = np.sum(V[:,j]**2)
                if (dv > ftol):
                    V[:,j] /= dv**0.5
                    break
                V[:,j] = np.random.randn(D)
        vb = mm(xb, V)
        nvb = mm(nxb, V)
        vw = mm(xw, V)
        rrl, rrp = geig_matrix(mm(vb.T, nvb) / W, mm(vw.T, vw) / W, gam, num=M)

        #  update the projection matrix

        x[:] = mm(V, rrp)
        rrp[0:M,:] = 0
        p[:] = mm(V, rrp)

    return (a, b, q, x / np.sum(x**2, axis=0).reshape((1,M))**0.5,
        Bx / np.sum(Bx**2, axis=0).reshape((1,M))**0.5)

class Cumulator(object):
    """Simple cumulative univariate moment estimator.

    Parameters
    ----------
    n  : int
        number of points
    m  : float
        mean
    m2 : float
        mean-removed square sum
    """
    def __init__(self, n=0, m=0.0, m2=0.0):
        assert isinstance(n, (int, float))
        assert isinstance(m, (int, float))
        assert isinstance(m2, (int, float))

        self._n = n
        self._m = m
        self._m2 = m2

    def add(self, x):
        """Add data to the cumulator.

        Parameters
        ----------
        x : float, ndarray or Cumulator object
            data
        """
        assert isinstance(x, (int, float, np.ndarray, Cumulator))

        if isinstance(x, Cumulator):
            if (x._n > 0):
                n = self._n + x._n
                d = x._m - self._m
                self._m = (self._n * self._m + x._n * x._m) / n
                self._m2 += x._m2 + self._n * x._n * d**2 / n
                self._n = n
        elif isinstance(x, np.ndarray):
            if (x.size == 0):
                pass
            elif (x.size == 1):
                self._n += 1
                d = x[0] - self._m
                self._m += d / self._n
                self._m2 += d * (x[0] - self._m)
            else:
                nx = x.size
                n = self._n + nx
                mx = np.mean(x)
                d = mx - self._m
                self._m = (self._n * self._m + nx * mx) / n
                self._m2 += np.sum((x - mx)**2) + self._n * nx * d**2 / n
                self._n = n
        else:
            self._n += 1
            d = x - self._m
            self._m += d / self._n
            self._m2 += d * (x - self._m)

    def copy(self):
        """Return a copy of the object.

        Returns
        -------
        Cumulator object
        """
        return Cumulator(self._n, self._m, self._m2)

    @property
    def m(self):
        """Return mean.

        Returns
        -------
        float
        """
        return self._m

    @property
    def m2(self):
        """Return mean-removed square sum.

        Returns
        -------
        float
        """
        return self._m2

    @property
    def n(self):
        """Return number of points.

        Returns
        -------
        int
        """
        return self._n

    @property
    def std(self):
        """Return sample standard deviation.

        Returns
        -------
        float
        """
        if (self._n < 2):
            return 0
        return (self._m2 / (self._n - 1))**0.5

    @property
    def var(self):
        """Return sample variance.

        Returns
        -------
        float
        """
        if (self._n < 2):
            return 0
        return self._m2 / (self._n - 1)

def nix_infer(C, p, k0=1.0, v0=1.0):
    """Estimate the sample mean and variance of a normal distribution given a
    Normal-Inverse-Chi-Squared (NIX) prior.

    Parameters
    ----------
    C : Cumulator object
        cumulated sample
    p : tuple(2) of float or ndarray(K)
        prior mean and variance
    k0 : float
        prior weight of mean
    v0 : float
        prior weight of variance

    Returns
    -------
    tuple(2)
        mean and variance
    """
    assert isinstance(C, Cumulator)
    assert isinstance(p, tuple) and (len(p) == 2)
    assert isinstance(k0, (int, float))
    assert isinstance(v0, (int, float))

    return ((k0 * p[0] + C.n * C.m) / (k0 + C.n), (v0 * p[1] + C.m2 +
        (k0 * C.n / (k0 + C.n)) * (p[0] - C.m)**2) / (v0 + C.n))

def sunion(A, B):
    """Sorted union of elements of A and B with source information.

    Parameters
    ----------
    A : ndarray
    B : ndarray

    Returns
    -------
    tuple(2)
        sorted union and bool ndarray indicating elements that belong to the
        original array A
    """
    if (A.size == 0):
        if (B.size == 0):
            return (np.array([]), np.array([], dtype=np.bool))
        else:
            U, z = np.unique(B, return_index=True)
    elif (B.size == 0):
        U, z = np.unique(A, return_index=True)
    else:
        U, z = np.unique(np.concatenate((A.flatten(), B.flatten())),
            return_index=True)
    return (U, z < A.size)

class AdditiveGMM(object):
    """Wrapper for the sklearn.mixture.GaussianMixture method enabling additive
    training. This implementation supports only univariate data.

    Parameters
    ----------
    nc : int
        maximal number of components
    """
    def __init__(self, nc):
        assert isinstance(nc, int) and (nc > 0)

        self._n = 0
        self._gmm = np.zeros((3, nc))
        self._opt = None

    def clear(self):
        """Reinitialize the model.
        """
        self._n = 0
        self._gmm.fill(0)
        self._opt = None

    def get_model(self):
        """Return the current optional model.

        Returns
        -------
        ndarray(3,K)
            parameters of the optimal model, 1 <= K <= nmax
        """
        if (self._opt is not None):
            return self._opt
        elif (self._n == 0):
            return None
        return self._get_opt(None)

    def insert(self, x, retain=False):
        """Add training data to the model and return the current optimal one.

        Parameters
        ----------
        x : ndarray(N)
            data
        retain : bool
            if True, retain the optimal model

        Returns
        -------
        ndarray(3,K)
            parameters of the optimal model, 1 <= K <= nmax
        """
        assert isinstance(x, np.ndarray) and (x.ndim == 1)

        if (x.size == 0):
            return None

        # restore previously inserted data, or if the number of data points is
        # greater than what can be stored, draw a proxy of previous insertions
        # from the distribution

        nmax = self._gmm.shape[1]
        was_trained = False
        if (self._n == 0):
            X = x
        elif (self._n <= 3 * nmax):
            X = np.concatenate((x, self._gmm.flat[0:self._n]))
        else:
            was_trained = True
            G = GMM(n_components=nmax, covariance_type='spherical')
            self._set(G)
            gx, y = G.sample(self._n)
            X = np.concatenate((x, gx.flatten()))

        # store exact data or train the maximal model
        # when training the model, initialize with the previous solution if that
        # was also a trained model and not just stored data

        N = X.size
        rc = 1e-6 + (0.9 * (X[-1] - X[0]) / N)**2
        if (N <= 3 * nmax):
            self._gmm.flat[0:N] = X
        else:
            G = GMM(n_components=nmax,
                covariance_type='spherical',
                reg_covar=rc,
                warm_start=was_trained)
            if was_trained:
                self._set(G)
            G.fit(X.reshape((-1,1)))
            self._gmm[0,:] = G.weights_
            self._gmm[1,:] = G.means_.flatten()
            self._gmm[2,:] = G.covariances_
        self._n = N

        # determine the optimal model

        M = self._get_opt(X)
        if retain:
            self._opt = M
        return M

    def predict(self, x):
        """Predict the labels of sample X.

        Parameters
        ----------
        x : ndarray(N)
            sample data

        Returns
        -------
        ndarray(N) of int
            component indices
        """
        M = self.get_model()
        if (M is None):
            return np.zeros(x.size, dtype=np.int)
        G = GMM(n_components=M.shape[1], covariance_type='spherical')
        self._set(G, model=M)
        return G.predict(x.reshape((-1,1)))

    def _get_opt(self, x):
        """Determine the current optimal model using the modified Akaike
        information criterion.

        Parameters
        ----------
        x : ndarray(N)
            data, or None

        Returns
        -------
        ndarray(3,K)
            parameters of the optimal model, 1 <= K <= nmax
        """
        nmax = self._gmm.shape[1]
        if (x is None):
            if (self._n == 0):
                return None
            elif (self._n <= 3 * nmax):
                X = self._gmm.flat[0:self._n]
            else:
                G = GMM(n_components=nmax, covariance_type='spherical')
                self._set(G)
                gx, y = G.sample(self._n)
                X = gx.flatten()
        else:
            X = x
        M = np.zeros((3,nmax))
        N = X.size
        if (N < 4):
            M[0,0] = 1
            M[1,0] = np.mean(X)
            if (N > 1):
                M[2,0] = np.std(X) + 1e-6
            else:
                M[2,0] = 1e-6
            Nc = 1
        else:
            rc = 1e-6 + (0.9 * (X[-1] - X[0]) / N)**2
            a = np.inf
            for i in range(nmax):
                Nc = i + 1
                K = 3 * Nc - 1
                if (K > N - 2):
                    Nc -= 1
                    break
                G = GMM(n_components=Nc, covariance_type='spherical',
                    reg_covar=rc)
                G.fit(X.reshape((-1,1)))
                anew = G.aic(X.reshape((-1,1))) + 2 * K * (K + 1) / (N - K - 1)
                if (a < anew):
                    Nc -= 1
                    break
                a = anew
                M[0,0:Nc] = G.weights_
                M[1,0:Nc] = G.means_.flatten()
                M[2,0:Nc] = G.covariances_**0.5
        return M[:,0:Nc].reshape((3,-1))

    def _set(self, G, model=None):
        """Set a GaussianMixture model to a previously fit and stored state.

        Parameters
        ----------
        G : GaussianMixture object
            object data
        model : ndarray(3,K) or None
            optional object data
        """
        if (model is None):
            G.weights_ = self._gmm[0,:].flatten()
            G.means_ = self._gmm[1,:].reshape((-1,1))
            G.covariances_ = self._gmm[2,:].flatten()
        else:
            G.weights_ = model[0,:].flatten()
            G.means_ = model[1,:].reshape((-1,1))
            G.covariances_ = model[2,:].flatten()
        G.precisions_cholesky_ = 1.0 / G.covariances_

def complement_split(x, c, G=None, x0=0.0, zc=5.0):
    """Determine the optimal partitioning complements.

    Parameters
    ----------
    x : ndarray(N)
        data
    c : ndarray(N)
        complement
    G : AdditiveGMM object
        prior state
    x0 : float
        partition point
    zc : float
        inclusion cutoff distance in standard deviations

    Returns
    -------
    tuple(2)
        tuple(2)
            ndarray(na) of int
                indices of elements of x in the complement of partition a
            ndarray(ma) of int
                indices of elements of c in the complement of partition a
        tuple(2)
            ndarray(nb) of int
                indices of elements of x in the complement of partition b
            ndarray(mb) of int
                indices of elements of c in the complement of partition b
    """
    assert isinstance(x, np.ndarray) and (x.ndim == 1)
    assert isinstance(c, np.ndarray) and (c.ndim == 1)
    assert isinstance(G, AdditiveGMM) or (G is None)
    assert isinstance(x0, (int, float))
    assert isinstance(zc, (int, float)) and (zc >= 0)

    # train the GMM and assign data points to GMM components

    s = np.concatenate((x, c))
    nx = x.size
    ns = s.size
    if (G is None):
        AG = AdditiveGMM(8)
    else:
        AG = G
    M = AG.insert(s, retain=True)
    y = AG.predict(s)

    # determine the inclusion coefficient Z for each component

    K = M.shape[1]
    mk = np.zeros(K+1)
    mk[0:K] = M[1,:]
    mk[K] = x0
    sk = np.zeros(K+1)
    sk[0:K] = M[2,:]
    sk[K] = 1e-6
    ix = np.argsort(mk)
    xix = np.where(ix == K)[0][0]
    ms = mk[ix]
    ss = sk[ix]
    z = norm.cdf(zc - (ms[1:K+1] - ms[0:K]) / (ss[1:K+1] + ss[0:K]))
    for i in range(xix+1,K):
        z[i] *= z[i-1]
    for i in range(xix-2,-1,-1):
        z[i] *= z[i+1]
    Z = np.zeros(K)
    Z[ix[ix != K]] = z

    # construct complements by random sampling of each component set

    ixa = np.zeros(0, dtype=np.int)
    ixb = np.zeros(0, dtype=np.int)
    ica = np.zeros(0, dtype=np.int)
    icb = np.zeros(0, dtype=np.int)
    for k in range(K):
        isk = np.arange(ns)[y == k]
        nk = isk.size
        if (nk > 0):
            ick = isk[(isk >= nx) | (s[isk] > x0)]
            nc = ick.size
            if (nc > 0):
                ncs = int(Z[k] * nc + 0.5)
                if (ncs > 0):
                    if (ncs == nc):
                        ics = ick
                    else:
                        ics = ick[np.random.choice(nc, size=ncs, replace=False)]
                    icx = (ics < nx)
                    ixa = np.concatenate((ixa, ics[icx]))
                    ica = np.concatenate((ica, ics[~icx] - nx))
            ick = isk[(isk >= nx) | (s[isk] < x0)]
            nc = ick.size
            if (nc > 0):
                ncs = int(Z[k] * nc + 0.5)
                if (ncs > 0):
                    if (ncs == nc):
                        ics = ick
                    else:
                        ics = ick[np.random.choice(nc, size=ncs, replace=False)]
                    icx = (ics < nx)
                    ixb = np.concatenate((ixb, ics[icx]))
                    icb = np.concatenate((icb, ics[~icx] - nx))
    return ((ixa, ica), (ixb, icb))

# EOF algorithms.py ____________________________________________________________

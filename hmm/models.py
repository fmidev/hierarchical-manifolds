"""
FILE:                   models.py
COPYRIGHT:              (c) 2024 Finnish Meteorological Institute
                        P.O. BOX 503
                        FI-00101 Helsinki, Finland
                        https://www.fmi.fi/
LICENCE:                MIT
AUTHOR:                 Terhi Mäkinen (terhi.makinen@fmi.fi)
DESCRIPTION:

This module provides objects for statistical description of data.
"""

from enum import Enum
import numpy as np
from scipy.special import gammaincc
from scipy.stats import norm, truncnorm
from . import algorithms as alg

# NB: In order to save memory, LDAData envelopes do not store data directly.
#     Instead, they store a reference to the original chunk of data, and a
#     vector of indices representing current selection, ordered by class if
#     assigned.

class LDAData(object):
    """LDA data encapsulation class.

    This is the interface definition.
    """
    def __init__(self): pass

    def add(self, B):
        """Add another LDAData selection to this one.

        Parameters
        ----------
        B : LDAData object
            selection
        """
        raise NotImplementedError()

    def copy(self):
        """Return a copy of the object.

        Returns
        -------
        LDAData object
        """
        raise NotImplementedError()

    @property
    def idx(self):
        """Return data selection indices.

        Returns
        -------
        ndarray of int
        """
        raise NotImplementedError()

    @property
    def is_empty(self):
        """Return True if object does not contain selected data.

        Returns
        -------
        bool
        """
        raise NotImplementedError()

    @property
    def n(self):
        """Return the number of selected data points.

        Returns
        -------
        int
        """
        raise NotImplementedError()

    def subset(self, s):
        """Return a subset of selected data.

        Parameters
        ----------
        s : ndarray(M) of bool or int
            selection

        Returns
        -------
        LDAData object
        """
        raise NotImplementedError()

    @property
    def x(self):
        """Return the selected data.

        Returns
        -------
        ndarray(N,D)
            data
        """
        raise NotImplementedError()

class XData(LDAData):
    """LDA uncategorized data encapsulation class.

    Parameters
    ----------
    x : ndarray(N,D)
        N points of D-dimensional data
    idx : ndarray(M) of int
        selection indices into x, or if boolean True selects all data points and
        False none
    ndim : int
        if nonzero check that data dimensionality agrees with the given argument
    finite : bool
        if True check that all elements of X are finite
    """
    def __init__(self, x, idx=True, ndim=0, finite=False):
        assert (isinstance(x, np.ndarray) and
                (x.ndim == 2)
               )
        assert (isinstance(idx, bool) or
                (isinstance(idx, np.ndarray) and
                 issubclass(idx.dtype.type, np.integer) and
                 (idx.ndim == 1) and
                 (idx.size <= x.shape[0]) and
                 (np.amin(idx) >= 0) and
                 (np.amax(idx) < x.shape[0])
                )
               )
        assert isinstance(ndim, int) and (ndim >= 0)
        assert isinstance(finite, bool)

        if (ndim > 0):
            assert (x.shape[1] == ndim)
        if finite:
            assert np.all(np.isfinite(x))
        self._x = x
        if isinstance(idx, np.ndarray):
            self._idx = idx.copy()
        elif idx:
            self._idx = np.arange(x.shape[0], dtype=np.int)
        else:
            self._idx = np.array([], dtype=np.int)

    def add(self, B):
        """Add another XData selection to this one.

        Parameters
        ----------
        B : XData object
            selection
        """
        assert isinstance(B, XData)
        assert (self._x is B._x)

        self._idx = np.union1d(self._idx, B._idx)

    def copy(self):
        """Return a copy of the object.

        Returns
        -------
        XData object
        """
        return XData(self._x, idx=self._idx)

    @property
    def idx(self):
        """Return data selection indices.

        Returns
        -------
        ndarray of int
        """
        return self._idx

    @property
    def is_empty(self):
        """Return True if object does not contain selected data.

        Returns
        -------
        bool
        """
        return (self._idx.size == 0)

    @property
    def n(self):
        """Return the number of selected data points.

        Returns
        -------
        int
        """
        return self._idx.size

    def subset(self, s):
        """Return a subset of selected data.

        Parameters
        ----------
        s : ndarray(M) of bool or int
            selection

        Returns
        -------
        XData object
        """
        assert (isinstance(s, np.ndarray) and
                ((s.dtype == np.bool) and
                 (s.size == self._idx.size)
                ) or
                ((s.dtype == np.int) and
                 (np.amin(s) >= 0) and
                 (np.amax(s) < self._idx.size)
                )
               )

        return XData(self._x, idx=self._idx[s])

    @property
    def x(self):
        """Return the selected data.

        Returns
        -------
        ndarray(N,D)
            data
        """
        if (self._idx.size == 0):
            return None        
        return self._x[self._idx,:].reshape((self._idx.size, -1))

# NB: The internal ordering of XYData indices loses class assignment information
#     of unselected data points. This is not an issue for the intended purpose
#     of the data envelope.

class XYData(LDAData):
    """LDA categorized data encapsulation class. Data indices are internally
    ordered by class for fast selection.

    Parameters
    ----------
    x : ndarray(N,D)
        N points of D-dimensional data
    y : ndarray(N) of int
        per-point data class, or
        ndarray(K) of int
        class starting indices.
        NB: if idx is a boolean y must be of the first format, otherwise second
    idx : ndarray(M) of int
        selection indices into x, or if boolean True selects all data points and
        False none
    w : ndarray(K) or None
        class weights, or if None use number of data points per class as weights
    ndim : int
        if nonzero check that data dimensionality agrees with the given argument
    ncls : int
        number of classes K; if zero determine K based on y, otherwise check
        that the number of classes agrees with the given argument
    finite : bool
        if True check that all elements of x are finite
    """
    def __init__(self, x, y,
        idx=True,
        w=None,
        ndim=0,
        ncls=0,
        finite=False):
        assert (isinstance(x, np.ndarray) and
                (x.ndim == 2)
               )
        assert (isinstance(y, np.ndarray) and
                (y.ndim == 1) and
                ((isinstance(idx, bool) and
                  (y.size == x.shape[0])
                 ) or
                 (isinstance(idx, np.ndarray) and
                  (np.all(y[:-1] <= y[1:])) and
                  (y[-1] <= idx.size)
                 )
                )
               )
        assert (isinstance(idx, bool) or
                (isinstance(idx, np.ndarray) and
                 issubclass(idx.dtype.type, np.integer) and
                 (idx.ndim == 1) and
                 (idx.size <= x.shape[0]) and
                 (np.amin(idx) >= 0) and
                 (np.amax(idx) < x.shape[0])
                )
               )
        assert ((w is None) or
                (isinstance(w, np.ndarray) and
                 (w.ndim == 1)
                )
               )
        assert isinstance(ndim, int) and (ndim >= 0)
        assert isinstance(ncls, int) and (ncls >= 0)
        assert isinstance(finite, bool)

        if (ndim > 0):
            assert (x.shape[1] == ndim)
        if finite:
            assert np.all(np.isfinite(x))
        self._x = x
        N = x.shape[0]
        if isinstance(idx, np.ndarray):
            K = y.size
            if (ncls > 0):
                assert (ncls == K)
            self._y = y.copy()
            self._idx = idx.copy()
        else:
            K = np.amax(y) + 1
            if (ncls > 0):
                assert (K <= ncls)
                if (K < ncls):
                    K = ncls
            if idx:
                self._y = np.zeros(K, dtype=np.int)
                self._idx = np.zeros(N, dtype=np.int)
                ndx = np.arange(N, dtype=np.int)
                n = 0
                for k in range(K):
                    self._y[k] = n
                    sx = (y == k)
                    nx = np.count_nonzero(sx)
                    if (nx > 0):
                        self._idx[n:n+nx] = ndx[sx]
                        n += nx
            else:
                self._y = np.zeros(K, dtype=np.int)
                self._idx = np.array([], dtype=np.int)
        if (w is None):
            self._w = np.zeros(K)
            self._w[0:K-1] = self._y[1:K] - self._y[0:K-1]
            self._w[K-1] = self._idx.size - self._y[K-1]
        else:
            assert (w.size == K)
            self._w = w.copy()

    def add(self, B):
        """Add another XYData selection to this one.

        Parameters
        ----------
        B : XYData object
            selection

        Returns
        -------
        ndarray(N) of bool
            True indicates elements that were in self before addition
        """
        assert isinstance(B, XYData)
        assert (self._x is B._x)

        K = self._y.size
        idx = []
        src = []
        y = np.zeros(K, dtype=np.int)
        n = 0
        for k in range(1,K):
            u, s = alg.sunion(self._idx[self._y[k-1]:self._y[k]],
                B._idx[B._y[k-1]:B._y[k]])
            for i in range(u.size):
                idx.append(u[i])
                src.append(s[i])
            n += u.size
            y[k] = n
        u, s = alg.sunion(self._idx[self._y[K-1]:], B._idx[B._y[K-1]:])
        for i in range(u.size):
            idx.append(u[i])
            src.append(s[i])
        self._idx = np.array(idx, dtype=np.int)
        self._y = y
        self._w += B._w
        return np.array(src, dtype=np.bool)

    def copy(self):
        """Return a copy of the object.

        Returns
        -------
        XYData object
        """
        if (self._idx.size == 0):
            return XYData(self._x, np.zeros(self._x.shape[0], dtype=np.int),
                idx=False, w=np.zeros(self._w.size), ncls=self.nclass)
        return XYData(self._x, self._y, idx=self._idx, w=self._w)

    @property
    def idx(self):
        """Return data selection indices.

        Returns
        -------
        ndarray of int
        """
        return self._idx

    @property
    def is_empty(self):
        """Return True if object does not contain selected data.

        Returns
        -------
        bool
        """
        return (self._idx.size == 0)

    @property
    def n(self):
        """Return the number of selected data points.

        Returns
        -------
        int
        """
        return self._idx.size

    @property
    def nclass(self):
        """Return the number of classes.

        Returns
        -------
        int
        """
        return self._y.size

    def subset(self, s, q=None, w=None):
        """Return a subset of selected data.

        Parameters
        ----------
        s : ndarray(M) of bool or int
            selection
        q : ndarray(K) or None
            weight ratios
        w : ndarray(K) or None
            weights; if neither q nor w are provided use the ratio of selected
            to total number of points

        Returns
        -------
        XYData object
        """
        assert (isinstance(s, np.ndarray) and
                ((s.dtype == np.bool) and
                 (s.size == self._idx.size)
                ) or
                ((s.dtype == np.int) and
                 ((s.size == 0) or
                  ((np.amin(s) >= 0) and
                   (np.amax(s) < self._idx.size)
                  )
                 )
                )
               )
        assert ((q is None) or
                (isinstance(q, np.ndarray) and
                 (q.ndim == 1) and
                 (q.size == self._y.size)
                )
               )
        assert ((w is None) or
                (isinstance(w, np.ndarray) and
                 (w.ndim == 1) and
                 (w.size == self._y.size)
                )
               )

        if (s.size == 0):
            return XYData(self._x, np.zeros(self._x.shape[0], dtype=np.int),
                idx=False, w=np.zeros(self._w.size), ncls=self.nclass)
        K = self._y.size
        Y = np.zeros(K, dtype=np.int)
        if (s.dtype == np.bool):
            for k in range(1,K):
                if (self._y[k-1] < self._y[k]):
                    Y[k] = Y[k-1] + np.count_nonzero(s[self._y[k-1]:self._y[k]])
                else:
                    Y[k] = Y[k-1]
        else:
            for k in range(1,K):
                Nk = 0
                if (self._y[k-1] < self._y[k]):
                    slo = np.argwhere(s >= self._y[k-1])
                    if (slo.size > 0):
                        slox = slo.min()
                        shi = np.argwhere(s < self._y[k])
                        if (shi.size > 0):
                            shix = shi.max()
                            if (slox <= shix):
                                Nk = shix - slox + 1
                Y[k] = Y[k-1] + Nk
        idx = self._idx[s]
        if (w is None):
            if (q is None):
                Nt = np.zeros(K)
                Nt[0:K-1] = self._y[1:K] - self._y[0:K-1]
                Nt[K-1] = self._idx.size - self._y[K-1]
                Ns = np.zeros(K)
                Ns[0:K-1] = Y[1:K] - Y[0:K-1]
                Ns[K-1] = idx.size - Y[K-1]
                W = np.zeros(K)
                sx = Nt > 0
                if (np.count_nonzero(sx) > 0):
                    W[sx] = Ns[sx] / Nt[sx]
                W *= self._w
            else:
                W = self._w * q
        else:
            W = w
        return XYData(self._x, Y, idx=idx, w=W)

    @property
    def w(self):
        """Return class weights.

        Returns
        -------
        ndarray
        """
        return self._w

    @property
    def x(self):
        """Return the selected data.

        Returns
        -------
        ndarray(N,D)
        """
        if (self._idx.size == 0):
            return None        
        return self._x[self._idx,:].reshape((self._idx.size, -1))

    @property
    def y(self):
        """Return class starting indices in selected data.

        Returns
        -------
        ndarray of int
        """
        return self._y

    @property
    def yxt(self):
        """Return extended class starting indices in selected data.

        Returns
        -------
        ndarray of int
        """
        K = self._y.size
        Y = np.ndarray(K+1, dtype=np.int)
        Y[0:K] = self._y
        Y[K] = self._idx.size
        return Y

class CMxType(Enum):
    """Enumeration of possible covariance matrix models for MultiStat

    NONE        do not calculate the covariance matrix
    AVERAGE     calculate a single covariance matrix for all data
    CLASS       calculate a separate covariance matrix for each class
    """
    NONE = 1
    AVERAGE = 2
    CLASS = 3

#   NB: MultiStat stores both number of points per class (cn) and class weights
#       (cw) and these cannot be combined to save memory because they are needed
#       for different purposes and may differ in value. Number of points is used
#       for calculating means and variances, and for applying inference; thus,
#       they represent the relative certainty of the parametrization. Class
#       weights are used for estimating the local point density and are not
#       required to be integer; specifically, the partitioning process yields
#       fractional weights both for total (volume integrated) data probability
#       and complement density.

class MultiStat(object):
    """Multiclass multivariate mean and covariance.

    Parameters
    ----------
    ncls : int
        number of classes
    ndim : int
        number of data dimensions
    ctype : CMxType
        covariance matrix model type
    """
    def __init__(self, ncls, ndim, ctype):
        assert isinstance(ncls, int) and (ncls > 0)
        assert isinstance(ndim, int) and (ndim > 0)
        assert isinstance(ctype, CMxType)

        self._cw = np.zeros(ncls)
        self._cn = np.zeros(ncls)
        self._cm = np.zeros((ncls,ndim))
        if (ctype == CMxType.CLASS):
            self._m = None
            self._cc = np.zeros((ncls,ndim,ndim))

            # variables for calculating pdf, updated on demand

            self._iv = None
            self._pp = None
        else:
            self._m = np.zeros((1,ndim))
            if (ctype == CMxType.AVERAGE):
                self._cc = np.zeros((ndim,ndim))
            else:
                self._cc = None

    def clear(self):
        """Reinitialize the model state.
        """
        self._cw.fill(0)
        self._cn.fill(0)
        self._cm.fill(0)
        if (self._m is not None):
            self._m.fill(0)
        if (self._cc is not None):
            self._cc.fill(0)
        self._iv = None
        self._pp = None

    @property
    def cmean(self):
        """Return class means.

        Returns
        -------
        ndarray
        """
        return self._cm

    @property
    def cn(self):
        """Return number of points per class.

        Returns
        -------
        ndarray of int
        """
        return self._cn

    @property
    def covar(self):
        """Return covariance matrix or matrices, or None for CMxType.NONE.

        Returns
        -------
        ndarray(D,D) or ndarray(K,D,D) or None
            covariance matrix
        """
        return self._cc

    @property
    def ctype(self):
        """Return covariance matrix model type.

        Returns
        -------
        CMxType
        """
        if (self._cc is None):
            return CMxType.NONE
        elif (self._cc.ndim == 2):
            return CMxType.AVERAGE
        else:
            return CMxType.CLASS

    @property
    def cw(self):
        """Return class weights.

        Returns
        -------
        ndarray
        """
        return self._cw

    def debias(self, A, R=None, full=False):
        """Remove mixture mean and optionally class means from data.

        Parameters
        ----------
        A : LDAData object
            data
        R : ndarray(D,M) or None
            if D, the dimensionality of A differs from M, the dimensionality of
            the model, then R must be a projection matrix from R^D to R^M
        full : bool
            if set, also remove class means

        Returns
        -------
        ndarray(N,M)
            data with means removed
        """
        assert (isinstance(A, LDAData) and
                (A.x.shape[1] == self.ndim) or
                (isinstance(R, np.ndarray) and
                 (R.ndim == 2) and
                 (A.x.shape[1] == R.shape[0]) and
                 (R.shape[1] == self.ndim)
                )
               )
        assert isinstance(full, bool)

        K, D = self.shape
        if (A.x.shape[1] == D):
            if (self._m is None):
                x = A.x.copy()
            else:
                x = A.x - self._m
        else:
            if (self._m is None):
                x = np.matmul(A.x, R)
            else:
                x = np.matmul(A.x, R) - self._m
        if full:
            y = A.yxt
            for k in range(K):
                nk = y[k+1] - y[k]
                if (nk > 0):
                    x[y[k]:y[k+1],:] -= self._cm[k,:].reshape((1,D))
        return x

    def insert(self, A, R=None):
        """Insert data.

        Parameters
        ----------
        A : XYData object
            data
        R : ndarray(D,M) or None
            if D, the dimensionality of A differs from M, the dimensionality of
            the model, then R must be a projection matrix from R^D to R^M

        Returns
        -------
        tuple(3)
            ndarray(K)
                class mean delta coefficients
            ndarray(K,M)
                class mean delta
            ndarray(N,M)
                data with class means removed
        """
        assert (isinstance(A, XYData) and
                (A.x.shape[1] == self.ndim) or
                (isinstance(R, np.ndarray) and
                 (R.ndim == 2) and
                 (A.x.shape[1] == R.shape[0]) and
                 (R.shape[1] == self.ndim)
                )
               )

        if ((self._cc is not None) and (self._cc.ndim == 3)):
            self._iv = None             # invalidate pdf if stored
            self._pp = None
        K, D = self._cm.shape
        nn = np.zeros(K)
        dd = np.zeros((K,D))
        N, Da = A.x.shape
        if (R is None):
            x = A.x.copy()
        else:
            x = np.matmul(A.x, R)
        y = A.yxt
        self._cw += A.w
        if (self._m is None):           # CMxType.CLASS
            for k in range(K):
                nb = y[k+1] - y[k]
                if (nb > 0):
                    nab = self._cn[k] + nb
                    nn[k] = self._cn[k] * nb / nab
                    mb = np.mean(x[y[k]:y[k+1],:].reshape((nb,D)),
                        axis=0).reshape((1,D))
                    xb = x[y[k]:y[k+1],:].reshape((nb,D)) - mb
                    cb = np.matmul(xb.T, xb)
                    dd[k,:] = mb - self._cm[k,:]
                    self._cm[k,:] = ((self._cn[k] * self._cm[k,:] + nb * mb) /
                        nab)
                    x[y[k]:y[k+1],:] -= self._cm[k,:]
                    self._cc[k,:,:] += cb + nn[k] * np.outer(dd[k,:], dd[k,:])
                    self._cn[k] = nab
        else:
            Na = np.sum(self._cn)
            Nab = Na + N
            m = np.mean(x, axis=0).reshape((1,D))
            xb = x - m
            cb = np.matmul(xb.T, xb)
            d = m - self._m
            self._cm += self._m
            self._m = (Na * self._m + N * m) / Nab
            self._cm -= self._m
            x -= self._m
            if (self._cc is not None):  # CMxType.AVERAGE
                self._cc += cb + (Na * N / Nab) * np.matmul(d.T, d)
            for k in range(K):          # CMxType.NONE and CMxType.AVERAGE
                nb = y[k+1] - y[k]
                if (nb > 0):
                    nab = self._cn[k] + nb
                    nn[k] = self._cn[k] * nb / nab
                    mb = np.mean(x[y[k]:y[k+1],:].reshape((nb,D)),
                        axis=0).reshape((1,D))
                    dd[k,:] = mb - self._cm[k,:]
                    self._cm[k,:] = ((self._cn[k] * self._cm[k,:] +
                        nb * mb) / nab)
                    x[y[k]:y[k+1],:] -= self._cm[k,:]
                    self._cn[k] = nab
        return (nn, dd, x)

    @property
    def is_empty(self):
        """Return True if the model is empty.

        Returns
        -------
        bool
        """
        return (np.sum(self._cw) == 0)

    @property
    def is_multi(self):
        """Return True if the model contains points from more than one class.

        Returns
        -------
        bool
        """
        return (np.count_nonzero(self._cn) > 1)

    @property
    def mean(self):
        """Return mixture mean.

        Returns
        -------
        ndarray
        """
        return self._m

    @property
    def n(self):
        """Return total number of points.

        Returns
        -------
        int
        """
        return np.sum(self._cn)

    @property
    def ndim(self):
        """Return the dimensionality of the model.

        Returns
        -------
        int
        """
        return self._cm.shape[1]

    @property
    def nonzero(self):
        """Return nonzero class number of points and means.

        Returns
        -------
        tuple(5)
            int
                total number of points
            int
                number of nonzero classes
            ndarray(nx) of int
                nonzero class indices
            ndarray(nx,1)
                number of points per class
            ndarray(nx,D)
                class means
        """
        sx = (self._cn > 0)
        nx = np.count_nonzero(sx)
        if (nx < 1):
            return (0, None, None, None)
        K, D = self._cm.shape
        return (np.sum(self._cn),
                nx,
                np.arange(K)[sx],
                self.cn[sx].reshape((nx,1)),
                self.cmean[sx,:].reshape((nx,D))
               )

    def predict_sf(self, x):
        """Determine the per-class survival function at the given points.

        Parameters
        ----------
        x : ndarray(N,D)
            data
        R : ndarray(D,M)
            projection matrix from R^D to R^M

        Returns
        -------
        ndarray(N,K)
            per-class survival function
        """
        assert (self._cc is not None) and (self._cc.ndim == 3)
        assert (isinstance(x, np.ndarray) and
                (x.ndim == 2) and
                (x.shape[1] == self._cm.shape[1])
               )

        K, D = self._cm.shape
        N = x.shape[0]
        if (self._pp is None):
            v, self._pp = np.linalg.eigh(
                (self._cc + np.eye(D).reshape((1,D,D))) /
                (self._cn.reshape((K,1,1)) + 1.0))
            self._iv = 1.0 / v.reshape((K,D,1))
        return gammaincc(0.5 * D, 0.5 * np.matmul(np.matmul(x.reshape((1,N,D)) -
            self._cm.reshape((K,1,D)), self._pp)**2, self._iv).reshape((K,N)).T)

    @property
    def shape(self):
        """Return a tuple of number of classes and data dimensionality.

        Returns
        -------
        tuple(2)
            int
                number of classes
            int
                data dimensionality
        """
        return self._cm.shape

class EigType(Enum):
    """Enumeration of available eigensystem algorithms
    NONE        do not solve the eigensystem
    AUTO        choose the algorithm based on data properties
    MATRIX      explicit covariance matrix algorithm
    LOBPCG      LOBPCG algorithm for high-dimensional data
    """
    NONE = 1
    AUTO = 2
    MATRIX = 3
    LOBPCG = 4

class IEigSolver(object):
    """Eigenvalue problem solver.

    This is the interface definition.
    """
    def __init__(self): pass

    def clear(self):
        """Reset the model.
        """
        raise NotImplementedError()

    @property
    def etype(self):
        """Return the algorithm type.

        Returns
        -------
        EigType
        """
        raise NotImplementedError()

    def insert(self, x, w, niter, ftol):
        """Insert data to model.

        Parameters
        ----------
        x : ndarray(N,D)
            debiased data
        w : float
            weight of the current data set
        niter : int
            maximal number of iterations
        ftol : float
            floating point tolerance
        """
        raise NotImplementedError()

    def pc(self, num=0):
        """Return principal components.

        Parameters
        ----------
        num : int
            number of returned components; if num is zero return all available
            components

        Returns
        -------
        ndarray(D,num)
        """
        raise NotImplementedError()

class EigMatrix(IEigSolver):
    """Eigenvalue problem solver, covariance matrix algorithm.

    Parameters
    ----------
    ndim : int
        number of dimensions
    neig : int
        number of eigenvectors
    """
    def __init__(self, ndim, neig):
        assert isinstance(ndim, int) and (ndim > 0)
        assert isinstance(neig, int) and (neig > 0) and (neig <= ndim)

        self._cc = np.zeros((ndim,ndim))
        self._pc = np.zeros((ndim,neig))

    def clear(self):
        """Reset the model.
        """
        self._cc.fill(0)
        self._pc.fill(0)

    @property
    def etype(self):
        """Return the algorithm type.

        Returns
        -------
        EigType
        """
        return EigType.MATRIX

    def insert(self, x, w, niter, ftol):
        """Insert data to model.

        Parameters
        ----------
        x : ndarray(N,D)
            debiased data
        w : N/A
            retained for compatibility
        niter : N/A
            retained for compatibility
        ftol : N/A
            retained for compatibility
        """
        assert isinstance(x, np.ndarray)

        self._cc += np.matmul(x.T, x)
        v, self._pc = alg.eig_matrix(self._cc, num=self._pc.shape[1])

    def pc(self, num=0):
        """Return principal components.

        Parameters
        ----------
        num : int
            number of returned components; if num is zero return all available
            components

        Returns
        -------
        ndarray(D,num)
        """
        assert isinstance(num, int) and (num <= self._pc.shape[1])

        if ((num <= 0) or (num == self._pc.shape[1])):
            return self._pc
        return self._pc[:,0:num].reshape((-1,num))

class EigLOBPCG(IEigSolver):
    """Eigenvalue problem solver, LOBPCG algorithm.

    Parameters
    ----------
    ndim : int
        number of dimensions
    neig : int
        number of eigenvectors
    """
    def __init__(self, ndim, neig):
        assert isinstance(ndim, int) and (ndim > 0)
        assert isinstance(neig, int) and (neig > 0) and (3 * neig <= ndim)

        self._pc = np.random.randn(ndim*neig).reshape((ndim,neig))
        self._pc /= np.sum(self._pc**2, axis=0).reshape((1,neig))**0.5

    def clear(self):
        """Reset the model.
        """
        ndim, neig = self._pc.shape
        self._pc = np.random.randn(ndim*neig).reshape((ndim,neig))
        self._pc /= np.sum(self._pc**2, axis=0).reshape((1,neig))**0.5

    @property
    def etype(self):
        """Return the algorithm type.

        Returns
        -------
        EigType
        """
        return EigType.LOBPCG

    def insert(self, x, w, niter, ftol):
        """Insert data to model.

        Parameters
        ----------
        x : ndarray(N,D)
            debiased data
        w : float
            weight of the current data set
        niter : int
            maximal number of iterations
        ftol : float
            floating point tolerance
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(w, float) and (w > 0)
        assert isinstance(niter, int) and (niter > 0)
        assert isinstance(ftol, float) and (ftol > 0)

        H = self._pc.shape[1]
        if (H == 1):
            lam, pc = alg.eig_lopcg1(x, self._pc, niter=niter, ftol=ftol)
        else:
            lam, pc = alg.eig_lobpcg(x, self._pc, niter=niter, ftol=ftol)
        for i in range(H):
            if (np.sum(pc[:,i] * self._pc[:,i]) < 0):
                self._pc[:,i] = (1 - w) * self._pc[:,i] - w * pc[:,i]
            else:
                self._pc[:,i] = (1 - w) * self._pc[:,i] + w * pc[:,i]
            for j in range(i):
                self._pc[:,i] -= (np.sum(self._pc[:,i] * self._pc[:,j]) *
                    self._pc[:,j])
            self._pc[:,i] /= np.sum(self._pc[:,i]**2)**0.5

    def pc(self, num=0):
        """Return principal components.

        Parameters
        ----------
        num : int
            number of returned components; if num is zero return all available
            components

        Returns
        -------
        ndarray(D,1)
        """
        assert isinstance(num, int) and (num <= self._pc.shape[1])

        if ((num == 0) or (num == self._pc.shape[1])):
            return self._pc
        return self._pc[:,0:num].reshape((-1,num))

class IGEigSolver(object):
    """Generalized eigenvalue problem solver.

    This is the interface definition.
    """
    def __init__(self): pass

    def clear(self):
        """Reset the model.
        """
        raise NotImplementedError()

    @property
    def data(self):
        """Return a reference to the data model.

        Returns
        -------
        MultiStat object
        """
        raise NotImplementedError()

    @property
    def etype(self):
        """Return the algorithm type.

        Returns
        -------
        EigType
        """
        raise NotImplementedError()

    def insert(self, A, niter, ftol, prior):
        """Insert data to model.

        Parameters
        ----------
        A : XYData object
            data
        niter : int
            maximal number of iterations
        ftol : float
            floating point tolerance
        prior : float
            regularization factor

        Returns
        -------
        tuple(4)
            ndarray(K)
                class mean delta coefficients
            ndarray(K,D)
                class mean delta
            ndarray(N,D)
                data with class means removed
            ndarray(D,1)
                first principal component
        """
        raise NotImplementedError()

    @property
    def is_valid(self):
        """Return True if splitter is valid.

        Returns
        -------
        bool
        """
        raise NotImplementedError()

    def pc(self, num=0):
        """Return principal components.

        Parameters
        ----------
        num : int
            number of returned components; if num is zero return all available
            components

        Returns
        -------
        ndarray(D,num)
        """
        raise NotImplementedError()

class GEigNone(IGEigSolver):
    """Generalized eigenvalue problem solver without the solver.

    This dummy solver provides an IGEigSolver interface to the underlying
    MultiStat object in cases where the solver itself is not required.

    Parameters
    ----------
    ncls : int
        number of classes
    ndim : int
        number of dimensions
    neig : int
        number of eigenvectors
    """
    def __init__(self, ncls, ndim, neig):
        assert isinstance(ncls, int) and (ncls > 0)
        assert isinstance(ndim, int) and (ndim > 0)

        self._dat = MultiStat(ncls, ndim, CMxType.NONE)
        self._pc = None

    def clear(self):
        """Reset the model.
        """
        self._dat.clear()

    @property
    def data(self):
        """Return a reference to the data model.

        Returns
        -------
        MultiStat object
        """
        return self._dat

    @property
    def etype(self):
        """Return the algorithm type.

        Returns
        -------
        EigType
        """
        return EigType.NONE

    def insert(self, A, niter=None, ftol=None, prior=None):
        """Insert data to model.

        Parameters
        ----------
        A : XYData object
            data
        ftol : N/A
            retained for compatibility
        niter : N/A
            retained for compatibility
        prior : N/A
            retained for compatibility

        Returns
        -------
        tuple(4)
            ndarray(K)
                class mean delta coefficients
            ndarray(K,D)
                class mean delta
            ndarray(N,D)
                data with class means removed
            None
                retained for compatibility
        """
        assert isinstance(A, XYData)

        nn, dd, xw = self._dat.insert(A)
        return (nn, dd, xw, None)

    @property
    def is_valid(self):
        """Return True if splitter is valid.

        Returns
        -------
        bool
        """
        return False

    def pc(self, num=0):
        """Return principal components.

        Parameters
        ----------
        num : int
            number of returned components; if num is zero return all available
            components

        Returns
        -------
        ndarray(D,num)
        """
        return False

class GEigMatrix(IGEigSolver):
    """Generalized eigenvalue problem solver, covariance matrix algorithm.

    Parameters
    ----------
    ncls : int
        number of classes
    ndim : int
        number of dimensions
    neig : int
        number of eigenvectors
    """
    def __init__(self, ncls, ndim, neig):
        assert isinstance(ncls, int) and (ncls > 0)
        assert isinstance(ndim, int) and (ndim > 0)
        assert isinstance(neig, int) and (neig > 0) and (neig <= ndim)

        self._dat = MultiStat(ncls, ndim, CMxType.AVERAGE)
        self._pc = np.zeros((ndim,neig))

    def clear(self):
        """Reset the model.
        """
        self._dat.clear()
        self._pc.fill(0)

    @property
    def data(self):
        """Return a reference to the data model.

        Returns
        -------
        MultiStat object
        """
        return self._dat

    @property
    def etype(self):
        """Return the algorithm type.

        Returns
        -------
        EigType
        """
        return EigType.MATRIX

    def insert(self, A, niter=None, ftol=None, prior=1.0):
        """Insert data to model.

        Parameters
        ----------
        A : XYData object
            data
        ftol : N/A
            retained for compatibility
        niter : N/A
            retained for compatibility
        prior : float
            regularization factor

        Returns
        -------
        tuple(4)
            ndarray(K)
                class mean delta coefficients
            ndarray(K,D)
                class mean delta
            ndarray(N,D)
                data with class means removed
            ndarray(D,1)
                first principal component
        """
        assert isinstance(A, XYData)
        assert isinstance(prior, (int, float)) and (prior >= 0)

        D, H = self._pc.shape
        nn, dd, xw = self._dat.insert(A)
        N, K, ix, nb, xb = self._dat.nonzero
        if (N > K):
            W = N - K
        else:
            W = 1
        Sw = self._dat.covar / W
        Sb = np.matmul(xb.T, nb * xb) / W
        lam, self._pc = alg.geig_matrix(Sb, Sw, prior / (W + prior), num=H)
        if (H > 1):
            for i in range(H):
                while True:
                    for j in range(i):
                        self._pc[:,i] -= np.sum(self._pc[:,i] *
                             self._pc[:,j]) * self._pc[:,j]
                    dp = np.sum(self._pc[:,i]**2)
                    if (dp > 0):
                        break
                    self._pc[:,i] = np.random.randn(D)
                self._pc[:,i] /= dp**0.5
        if (np.count_nonzero(nb > 0) < 2):
            return (nn, dd, xw, None)
        return (nn, dd, xw, self._pc[:,0].reshape((-1,1)))

    @property
    def is_valid(self):
        """Return True if splitter is valid.

        Returns
        -------
        bool
        """
        return (np.count_nonzero(self._pc) > 0)

    def pc(self, num=0):
        """Return principal components.

        Parameters
        ----------
        num : int
            number of returned components; if num is zero return all available
            components

        Returns
        -------
        ndarray(D,num)
        """
        assert isinstance(num, int) and (num <= self._pc.shape[1])

        if ((num <= 0) or (num == self._pc.shape[1])):
            return self._pc
        return self._pc[:,0:num].reshape((-1,num))

class GEigLOBPCG(IGEigSolver):
    """Generalized eigenvalue problem solver, LOBPCG algorithm.

    Parameters
    ----------
    ncls : int
        number of classes
    ndim : int
        number of dimensions
    neig : int
        number of eigenvectors
    """
    def __init__(self, ncls, ndim, neig):
        assert isinstance(ncls, int) and (ncls > 0)
        assert isinstance(ndim, int) and (ndim > 0)
        assert isinstance(neig, int) and (neig > 0) and (3 * neig <= ndim)

        self._dat = MultiStat(ncls, ndim, CMxType.NONE)
        self._pc = np.random.randn(ndim*neig).reshape((ndim,neig))
        self._pc /= np.sum(self._pc**2, axis=0).reshape((1,neig))**0.5
        if (neig > 1):
            self._po = np.zeros((ndim,neig))
        else:
            self._po = None

    def clear(self):
        """Reset the model.
        """
        self._dat.clear()
        ndim, neig = self._pc.shape
        self._pc = np.random.randn(ndim*neig).reshape((ndim,neig))
        self._pc /= np.sum(self._pc**2, axis=0).reshape((1,neig))**0.5
        if (self._po is not None):
            self._po.fill(0)

    @property
    def data(self):
        """Return a reference to the data model.

        Returns
        -------
        MultiStat object
        """
        return self._dat

    @property
    def etype(self):
        """Return the algorithm type.

        Returns
        -------
        EigType
        """
        return EigType.LOBPCG

    def insert(self, A, niter=100, ftol=1e-8, prior=1.0):
        """Insert data to model.

        Parameters
        ----------
        A : XYData object
            data
        niter : int
            maximal number of iterations
        ftol : float
            floating point tolerance
        prior : float
            regularization factor

        Returns
        -------
        tuple(4)
            ndarray(K)
                class mean delta coefficients
            ndarray(K,D)
                class mean delta
            ndarray(N,D)
                data with class means removed
            ndarray(D,1)
                first principal component
        """
        assert isinstance(A, XYData)
        assert isinstance(niter, int) and (niter > 0)
        assert isinstance(ftol, float) and (ftol > 0)
        assert isinstance(prior, (int, float)) and (prior >= 0)

        D, H = self._pc.shape
        nn, dd, xw = self._dat.insert(A)
        N, K, ix, nb, xb = self._dat.nonzero
        if (K == 0):
            return (nn, dd, xw, self._pc[:,0].reshape((-1,1)))
        g = xw.shape[0] / N
        if (H == 1):
            R = alg.geig_lopcg1(nb.flatten(), xb, xw, self._pc, niter=niter,
                ftol=ftol, prior=prior)
        else:
            R = alg.geig_lobpcg(nb.flatten(), xb, xw, self._pc, niter=niter,
                ftol=ftol, prior=prior)
        pc = R[3]
        for i in range(H):
            if (np.sum(pc[:,i] * self._pc[:,i]) < 0):
                self._pc[:,i] = (1 - g) * self._pc[:,i] - g * pc[:,i]
            else:
                self._pc[:,i] = (1 - g) * self._pc[:,i] + g * pc[:,i]
        if (self._po is not None):
            for i in range(H):
                self._po[:,i] = self._pc[:,i]
                while True:
                    for j in range(i):
                        self._po[:,i] -= (np.sum(self._po[:,i] * self._po[:,j])
                            * self._po[:,j])
                    dp = np.sum(self._po[:,i]**2)
                    if (dp > 0):
                        break
                    self._po[:,i] = np.random.randn(D)
                self._po[:,i] /= dp**0.5
        if (np.count_nonzero(nb > 0) < 2):
            return (nn, dd, xw, None)
        return (nn, dd, xw, self._pc[:,0].reshape((-1,1)))

    @property
    def is_valid(self):
        """Return True if splitter is valid.

        Returns
        -------
        bool
        """
        N, K, ix, nb, xb = self._dat.nonzero
        return (np.count_nonzero(nb > 0) >= 2)

    def pc(self, num=0):
        """Return principal components.

        Parameters
        ----------
        num : int
            number of returned components; if num is zero return all available
            components

        Returns
        -------
        ndarray(D,num)
        """
        assert isinstance(num, int) and (num <= self._pc.shape[1])

        if (self._po is None):
            if ((num <= 0) or (num == self._pc.shape[1])):
                return self._pc
            return self._pc[:,0:num].reshape((-1,num))
        if ((num <= 0) or (num == self._po.shape[1])):
            return self._po
        return self._po[:,0:num].reshape((-1,num))

class ProjData(object):
    """Statistics of the univariate projection of data on the normal.

    Parameters
    ----------
    ncls : int
        number of classes
    ngmm : int
        maximal number of Gaussian mixture model components
    """
    def __init__(self, ncls, ngmm):
        assert isinstance(ncls, int) and (ncls > 0)
        assert isinstance(ngmm, int) and (ngmm >= 0)

        self.offset = None
        self.cc = np.zeros(ncls)
        if (ngmm > 0):
            self.gmm = [alg.AdditiveGMM(ngmm) for k in range(ncls)]
        else:
            self.gmm = None

    def clear(self):
        """Reset the model.
        """
        self.offset = None
        self.cc.fill(0)
        if (self.gmm is not None):
            for gmm in self.gmm:
                gmm.clear()

class LDA(object):
    """Linear Discriminant Analysis with optimal pivot and optional density
    estimators.

    Parameters
    ----------
    ncls : int
        number of classes
    ndim : int
        number of dimensions
    solver : EigType
        eigenvalue algorithm type
    ngmm : int
        maximal number of Gaussian mixture model components
    sub : tuple(2)
        subspace dimension and projection matrix
    """
    def __init__(self, ncls, ndim, solver=EigType.AUTO, ngmm=0, sub=(0, None)):
        assert isinstance(ncls, int) and (ncls > 0)
        assert isinstance(ndim, int) and (ndim > 0)
        assert isinstance(solver, EigType)
        assert isinstance(ngmm, int) and (ngmm >= 0)
        assert (isinstance(sub, tuple) and
                (len(sub) == 2) and
                isinstance(sub[0], int) and
                (sub[0] >= 0) and
                ((sub[1] is None) or
                 (isinstance(sub[1], np.ndarray) and
                  (sub[1].ndim == 2) and
                  (sub[1].shape[0] == ndim)
                 )
                )
               )

        nsub = sub[0]
        if (sub[1] is None):
            leaf_solver = nsub
        else:
            leaf_solver = -1
        if (solver == EigType.AUTO):
            if ((3 * nsub > ndim) or (ndim < 6)):
                eigalg = EigType.MATRIX
            else:
                eigalg = EigType.LOBPCG
        else:
            eigalg = solver
        if (eigalg == EigType.MATRIX):
            self._split_solver = GEigMatrix(ncls, ndim, 1)
        else:
            self._split_solver = GEigLOBPCG(ncls, ndim, 1)
        self._prj = ProjData(ncls, ngmm)
        self._loss = 0.0
        if leaf_solver > 0:
            if (eigalg == EigType.MATRIX):
                self._leaf_solver = GEigMatrix(ncls, ndim, nsub)
            else:
                self._leaf_solver = GEigLOBPCG(ncls, ndim, nsub)
        elif leaf_solver < 0:
            self._leaf_solver = GEigNone(ncls, ndim, nsub)
        else:
            self._leaf_solver = None
        if (nsub > 0):
            self._sub = MultiStat(ncls, nsub, CMxType.CLASS)
        else:
            self._sub = None
        self._P = None

    def clear(self):
        """Reset the model.
        """
        self._split_solver.clear()
        self._prj.clear()
        self._loss = 0.0
        if (self._leaf_solver is not None):
            self._leaf_solver.clear()
        if (self._sub is not None):
            self._sub.clear()
        self._P = None

    @property
    def etype(self):
        """Return the eigenvalue algorithm type.

        Returns
        -------
        EigType
        """
        return self._split_solver.etype

    def insert(self, A, C=None, R=None, subtype='none',
        prior=(1.0, (1.0, 1.0), 1.0), niter=100, ftol=1e-8):
        """Insert data to model and determine the offset that minimizes
        partitioning loss.

        Parameters
        ----------
        A : XYData object
            data
        C : XYData object or None
            complement
        R : ndarray(D,M) or None
            projection matrix
        subtype : str
            density subspace type
        prior : tuple(3)
            Beta and NIX priors, and regularization factor
        niter : int
            maximal number of iterations
        ftol : float
            floating point tolerance

        Returns
        -------
        ndarray(K)
            partitioning vector, or None if data cannot be partitioned
        """
        assert isinstance(A, XYData)
        assert isinstance(C, XYData) or (C is None)
        assert isinstance(R, np.ndarray) or (R is None)
        assert (subtype in ('none', 'fixed', 'leaf', 'splitter'))
        assert isinstance(prior, tuple) and (len(prior) == 3)
        assert isinstance(prior[1], tuple) and (len(prior[1]) == 2)
        assert isinstance(niter, int) and (niter > 0)
        assert isinstance(ftol, float) and (ftol > 0)

        # train the split solver

        K, D = self._split_solver.data.shape
        nn, dd, xw, pc = self._split_solver.insert(A, prior=prior[2],
            niter=niter, ftol=ftol)

        # if leaf solver is present, train it with data and complement

        if (C is not None):
            U = A.copy()
            U.add(C)
        else:
            U = A
        if (self._leaf_solver is not None):

            ret = self._leaf_solver.insert(U, prior=prior[2],
                niter=niter, ftol=ftol)

        # if density data model is present, train it with data and complement

        if ((self._sub is not None) and (subtype != 'none')):
            if (subtype == 'fixed'):
                P = R
            elif ((subtype == 'leaf') or (not self._split_solver.is_valid)):
                self._P = P = self._leaf_solver.pc().copy()
            else:
                P = self._leaf_solver.pc().copy()
                N = P.shape[1]
                P[:,0] = self._split_solver.pc()[:,0]
                for i in range(N):
                    for j in range(i):
                        P[:,i] -= np.sum(P[:,i] * P[:,j]) * P[:,j]
                    P[:,i] /= np.sum(P[:,i]**2)**0.5
                self._P = P
            self._sub.insert(U, R=P)

        if (pc is None):
            return None

        # determine the optimal pivot

        xp = np.matmul(xw, pc).flatten()
        dp = np.matmul(dd, pc).flatten()
        y = A.yxt
        for k in range(K):
            self._prj.cc[k] += np.sum(xp[y[k]:y[k+1]]**2) + nn[k] * dp[k]**2
        cn = self._split_solver.data.cn
        cm = np.matmul(self._split_solver.data.cmean, pc).flatten()
        cw = self._split_solver.data.cw
        C0 = alg.Cumulator(np.sum(cn), 0, np.sum(self._prj.cc))
        p0 = alg.nix_infer(C0, (0, 1), k0=prior[1][0], v0=prior[1][1])
        mu = np.ndarray(K)
        s2 = np.ndarray(K)
        for k in range(K):
            Ck = alg.Cumulator(cn[k], cm[k], self._prj.cc[k])
            mu[k], s2[k] = alg.nix_infer(Ck, p0, k0=prior[1][0], v0=prior[1][1])
        p, L, q = alg.min_loss_offset(cn, cw, mu, s2, prior[0])

        if (q is not None):
            self._prj.offset = p
            self._loss = L
        return q

    @property
    def leaf_solver(self):
        """Return a reference to the leaf solver model.

        Returns
        -------
        IGEigSolver object
        """
        return self._leaf_solver

    @property
    def loss(self):
        """Return partitioning loss.

        Returns
        -------
        float
        """
        return self._loss

    @property
    def normal(self):
        """Return the partitioning hyperplane normal.

        Returns
        -------
        ndarray(D,1)
        """
        return self._split_solver.pc(num=1)

    @property
    def offset(self):
        """Return partitioning offset.

        Returns
        -------
        float
        """
        return self._prj.offset

    @property
    def pivot(self):
        """Return a pivot point on the partitioning hyperplane.

        Returns
        -------
        ndarray(1,D) or None if partitioning is not allowed
        """
        if (self._prj.offset is None):
            return None
        return (self._split_solver.data.mean + self._prj.offset *
            self._split_solver.pc(num=1).T)

    def predict_weight(self, x, R=None, density=True):
        """Determine per-class weights at the given points.

        Parameters
        ----------
        x : ndarray(N,D)
            evaluation points
        R : ndarray(D,M) or None
            projection matrix
        density : bool
            if set and density model is present, use it

        Returns
        -------
        ndarray(N,K)
            per-class weights
        """
        assert isinstance(x, np.ndarray) and (x.ndim == 2)
        assert isinstance(R, np.ndarray) or (R is None)
        assert isinstance(density, bool)

        if ((not density) or (self._sub is None)):
            return np.matmul(np.ones((x.shape[0],1)),
                self._split_solver.data.cw.reshape((1,-1)))
        if ((self._leaf_solver is None) or
            (self._leaf_solver.etype == EigType.NONE)
           ):
            assert (R is not None)
            P = R
        else:
            P = self._leaf_solver.pc()
        return (self._leaf_solver.data.cw.reshape((1,-1)) *
            self._sub.predict_sf(np.matmul(x, P)))

    @property
    def proj(self):
        """Return a reference to the projection model.

        Returns
        -------
        MultiStat object
        """
        return self._prj

    @property
    def split_solver(self):
        """Return a reference to the split solver model.

        Returns
        -------
        IGEigSolver object
        """
        return self._split_solver

    @property
    def sub(self):
        """Return the density subspace projection matrix.

        Returns
        -------
        ndarray(D,M)
        """
        return self._P

# EOF models.py ________________________________________________________________

# test the models.py module
#
# Usage: python -m unittest tests.test_models

import unittest
import numpy as np
from hmm import models
from .testutils import orthox, projx

def assert_models_equal(A, B):
    if (A._m is None):
        assert (B._m is None)
    else:
        np.testing.assert_almost_equal(A._m, B._m)
    np.testing.assert_almost_equal(A._cw, B._cw)
    np.testing.assert_almost_equal(A._cn, B._cn)
    np.testing.assert_almost_equal(A._cm, B._cm)
    if (A._cc is None):
        assert (B._cc is None)
    else:
        np.testing.assert_almost_equal(A._cc, B._cc)

class TestAlgorithms(unittest.TestCase):
    def test_xyadd(self):
        """ Test XYData addition.
        """
        K = 3
        N = 32
        X = np.arange(N, dtype=np.int).reshape((N,1))
        Y = (K * np.random.rand(N)).astype(int)
        D = models.XYData(X, Y)
        D1 = D.subset(np.random.rand(N) < 0.35)
        y1 = D1.yxt
        D2 = D.subset(np.random.rand(N) < 0.35)
        y2 = D2.yxt
        Du = models.XYData(X, Y, idx=False)
        Du.add(D1)
        Du.add(D2)
        yu = Du.yxt
        for k in range(K):
            ix1 = D1.idx[y1[k]:y1[k+1]]
            ix2 = D2.idx[y2[k]:y2[k+1]]
            ixu = Du.idx[yu[k]:yu[k+1]]
            for i in ix1:
                self.assertTrue(i in ixu)                                      #
            for i in ix2:
                self.assertTrue(i in ixu)                                      #
            for i in ixu:
                self.assertTrue((i in ix1) or (i in ix2))                      #

    def test_mstat(self):
        """ Test MultiStat cumulative insertion for different model
            configurations.
        """
        K = 3
        D = 3
        N = 32
        X = np.random.randn(N*D).reshape((N,D))
        Y = (K * np.random.rand(N)).astype(int)
        data = models.XYData(X, Y)
        Q = (models.CMxType.NONE,
             models.CMxType.AVERAGE,
             models.CMxType.CLASS)
        for ctype in Q:
            A = models.MultiStat(K, D, ctype)
            A.insert(data)
            B = models.MultiStat(K, D, ctype)
            sx = (np.random.rand(N) < 0.5)
            B.insert(data.subset(sx))
            B.insert(data.subset(~sx))
            assert_models_equal(A, B)                                          #

    def test_geigsolver(self):
        D = 4
        K = 3
        N = 32
        Y = (K * np.random.rand(N)).astype(int)
        cm = np.random.randn(K*D).reshape((K,D))
        W = orthox(D)
        X = np.zeros((N,D))
        for i in range(N):
            X[i,:] = (np.sum(W * np.random.randn(D).reshape((D,1)), axis=0) +
                      cm[Y[i],:]
                     )
        data = models.XYData(X, Y)
        M = models.MultiStat(K, D, ctype=models.CMxType.AVERAGE)
        M.insert(data)
        Sx = models.GEigMatrix(D)
        px = Sx.insert(M)
        Sl = models.GEigLOBPCG(D)
        pl = Sl.insert(M, M.debias(data))
        pp = np.abs(np.sum(pl * px))
        np.testing.assert_almost_equal(pp, 1, decimal=5)                       #

if __name__ == '__main__':
    unittest.main()

# test the algorithms.py module
#
# Usage: python -m unittest tests.test_algorithms

import unittest
import numpy as np
from scipy.stats import truncnorm
from hmm import algorithms
from .testutils import orthox

class TestAlgorithms(unittest.TestCase):
    def test_eig(self):
        D = 6
        X = orthox(D)
        d = np.sum(X**2, axis=1).reshape((D,1))**0.5
        U = X / d
        C = np.matmul(X.T, X)

        # eig_matrix

        v, P = algorithms.eig_matrix(C)
        np.testing.assert_almost_equal(v, d.flatten()**2)                      #
        pu = np.abs(np.sum(P.T * U, axis=1)).flatten()
        np.testing.assert_almost_equal(pu, np.ones(D))                         #

        # eig_matrix reverse

        vi, Pi = algorithms.eig_matrix(C, num=-D)
        for i in range(D):
            np.testing.assert_almost_equal(v[i], vi[D-i-1])                    #
            pu = np.abs(np.sum(P[:,i] * Pi[:,D-i-1]))
            np.testing.assert_almost_equal(pu, 1)                              #

        # eig_lopcg1

        p0 = np.random.randn(D).reshape((D,1))
        p0 /= np.sum(p0**2)**0.5
        v, p = algorithms.eig_lopcg1(X, p0)
        np.testing.assert_almost_equal(v, d[0,0]**2)                           #
        pu = np.abs(np.sum(p.flatten() * U[0,:]))
        np.testing.assert_almost_equal(pu, 1)                                  #

        # eig_lopcg1 reverse

        p0 = np.random.randn(D).reshape((D,1))
        p0 /= np.sum(p0**2)**0.5
        v, p = algorithms.eig_lopcg1(X, p0, reverse=True)
        np.testing.assert_almost_equal(v, d[D-1,0]**2)                         #
        pu = np.abs(np.sum(p.flatten() * U[D-1,:]))
        np.testing.assert_almost_equal(pu, 1)                                  #

        # eig_lobpcg 1

        p0 = np.random.randn(D).reshape((D,1))
        p0 /= np.sum(p0**2)**0.5
        v, p = algorithms.eig_lobpcg(X, p0)
        np.testing.assert_almost_equal(v, d[0,0]**2)                           #
        pu = np.abs(np.sum(p.flatten() * U[0,:]))
        np.testing.assert_almost_equal(pu, 1)                                  #

        # eig_lobpcg 1 reverse

        p0 = np.random.randn(D).reshape((D,1))
        p0 /= np.sum(p0**2)**0.5
        v, p = algorithms.eig_lobpcg(X, p0, reverse=True)
        np.testing.assert_almost_equal(v, d[D-1,0]**2)                         #
        pu = np.abs(np.sum(p.flatten() * U[D-1,:]))
        np.testing.assert_almost_equal(pu, 1)                                  #

        # eig_lobpcg 2

        p0 = np.random.randn(2*D).reshape((D,2))
        p0 /= np.sum(p0**2, axis=0).reshape((1,2))**0.5
        v, p = algorithms.eig_lobpcg(X, p0)
        for i in range(2):
            np.testing.assert_almost_equal(v[i], d[i,0]**2)                    #
            pu = np.abs(np.sum(p[:,i].flatten() * U[i,:]))
            np.testing.assert_almost_equal(pu, 1)                              #

        # eig_lobpcg 1 reverse

        p0 = np.random.randn(2*D).reshape((D,2))
        p0 /= np.sum(p0**2, axis=0).reshape((1,2))**0.5
        v, p = algorithms.eig_lobpcg(X, p0, reverse=True)
        for i in range(2):
            j = D - i - 1
            np.testing.assert_almost_equal(v[i], d[j,0]**2)                    #
            pu = np.abs(np.sum(p[:,i].flatten() * U[j,:]))
            np.testing.assert_almost_equal(pu, 1)                              #

    def test_geig(self):
        D = 6
        K = 2
        nb = np.ones(K, dtype=np.int)
        N = 5
        for n in range(N-K):
            nb[int(K * np.random.rand())] += 1
        xb = np.random.randn(K*D).reshape((K,D))
        nxb = nb.reshape((K,1)) * xb
        Sb = np.matmul(xb.T, nxb) / (N - K)
        W = orthox(D)
        xw = np.zeros((N,D))
        for i in range(N):
            xw[i,:] = np.sum(W * np.random.randn(D).reshape((D,1)), axis=0)
        Sw = np.matmul(xw.T, xw) / (N - K)
        r = 1 / (N - K + 1)
        Rw = (1 - r) * Sw + r * np.eye(D)

        # geig_matrix

        vm, Pm = algorithms.geig_matrix(Sb, Sw, r, num=D)
        for i in range(D):
            if (i > 0):
                assert (vm[i] <= vm[i-1])                                      #
            pm = Pm[:,i].reshape((D,1))
            np.testing.assert_almost_equal(np.matmul(Sb + Rw, pm),             #
                                           vm[i] * np.matmul(Rw, pm)
                                          )

        # geig_lopcg1

        p0 = np.random.randn(D).reshape((D,1))
        p0 /= np.sum(p0**2)**0.5
        a, b, q, pl, ql = algorithms.geig_lopcg1(nb, xb, xw, p0)
        np.testing.assert_almost_equal((a + q) / q, vm[0])                     #
        pp = np.abs(np.sum(Pm[:,0] * pl[:,0]))
        np.testing.assert_almost_equal(pp, 1)                                  #

        # geig_lobpcg 1

        p0 = np.random.randn(D).reshape((D,1))
        p0 /= np.sum(p0**2)**0.5
        a, b, q, pl, ql = algorithms.geig_lobpcg(nb, xb, xw, p0)
        np.testing.assert_almost_equal((a + q) / q, vm[0])                     #
        pp = np.abs(np.sum(Pm[:,0] * pl[:,0]))
        np.testing.assert_almost_equal(pp, 1)                                  #

        # geig_lobpcg 2

        p0 = np.random.randn(2*D).reshape((D,2))
        p0 /= np.sum(p0**2, axis=0).reshape((1,2))**0.5
        a, b, q, pl, ql = algorithms.geig_lobpcg(nb, xb, xw, p0)
        for i in range(2):
            np.testing.assert_almost_equal((a[i] + q[i]) / q[i], vm[i])        #
            pp = np.abs(np.sum(Pm[:,i] * pl[:,i]))
            np.testing.assert_almost_equal(pp, 1)                              #

if __name__ == '__main__':
    unittest.main()

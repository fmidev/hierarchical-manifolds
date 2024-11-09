# test the lda.py module
#
# Usage: python -m unittest tests.test_lda

import unittest
import numpy as np
import matplotlib.pyplot as plt
from hmm import hlda

class TestAlgorithms(unittest.TestCase):
    def test_train(self):
        """ Test basic training and evaluation.
        """
        X = np.array([[-1, -1],
                      [-1.5, 0],
                      [-1, 1],
                      [1, -1],
                      [1.5, 0],
                      [1, 1]
                     ]
                    )
        Y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int)

        T = hlda(2, 2)
        trs = T.fit(X, Y)

        # check that the total number of partitions is one

        self.assertEqual(T.root.nsplits, 1)                                    #

        # check that all partitions contain only single class data points

        for tnode, tix in trs:
            y = [Y[i] for i in tix]
            self.assertTrue(y.count(y[0]) == len(y))                           #

        # check that data points evaluate into the same partition as in training

        evs = T.eval(X)
        for enode, eix in evs:
            for i in eix:
                for tnode, tix in trs:
                    if (i in tix):
                        self.assertIs(enode, tnode)                            #

    def test_train_rho(self):
        """ Test basic training and evaluation with density.
        """
        X = np.array([[-0.9, -1],
                      [-1.8,  0],
                      [-0.9,  1],
                      [ 0.9, -1],
                      [ 1.8,  0],
                      [ 0.9,  1]
                     ]
                    )
        Y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int)
        sub = np.array([[1.0],[0.0]])

        T = hlda(2, 2, sub=sub)
        trs = T.fit(X, Y)

        # check that the total number of partitions is one

        self.assertEqual(T.root.nsplits, 1)                                    #

        # check that all partitions contain only single class data points

        for tnode, tix in trs:
            y = [Y[i] for i in tix]
            self.assertTrue(y.count(y[0]) == len(y))                           #

        # check that data points evaluate into the same partition as in training

        evs = T.eval(X)
        for enode, eix in evs:
            for i in eix:
                for tnode, tix in trs:
                    if (i in tix):
                        self.assertIs(enode, tnode)                            #

        # check pdf evaluation

        if False:
            z = np.linspace(-3.5, 3.5, num=100)
            Z = np.zeros((100,2))
            Z[:,0] = z
            p = np.exp(T.score(Z))
            fig = plt.figure()
            plt.title('H-LDA pdf test 1')
            plt.plot(X[0:3,0], X[0:3,1], 'o', c='blue')
            plt.plot(X[3:6,0], X[3:6,1], 'o', c='green')
            plt.plot(z, p[:,0], '-', c='skyblue')
            plt.plot(z, p[:,1], '-', c='limegreen')
            plt.axvline(x=0, color='gray', linestyle=':')
            plt.savefig('pdftest_1.pdf', format='pdf')
            plt.close(fig)

        p = np.exp(T.score(X))
        self.assertEqual(p.shape, (6, 2))                                      #

if __name__ == '__main__':
    unittest.main()

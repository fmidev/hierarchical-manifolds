# visualization of the partition loss function
# Usage: python -m extras.loss
#
# T.Makinen terhi.makinen@fmi.fi

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import gammainc, gammaincc, gammaln
import hmm.algorithms as alg

def _gammapmax(x, a, k):
    """ Probability density P(x_K = x, x_i < x for i < K) for a random vector
        x = (x_1,..,x_K) ~ Dir(a) where a_i are sorted in ascending order
        Input:
            x : float, argument in the range [0,inf]
            a : ndarray(K), ascending Dirichlet parameter vector
            k : int, number of least significant components discarded
    """
    K = a.size
    p = math.exp((a[K-1] - 1) * math.log(x) - x - gammaln(a[K-1]))
    for i in range(k,K-1):
        p *= gammainc(a[i], x)
    return p

def dir_pmax(a, z=5.0):
    """ Return the probability P(argmax_i x_i = argmax_j a_j) for a random
        vector x = (x_1,..,x_K) ~ Dir(a).
        Input:
            a : ndarray(K), Dirichlet parameter vector
            z : float, per-component upper integration limit in standard
                deviations
        Output:
            float, probability of consistency
    """
    assert isinstance(a, np.ndarray) and (a.ndim == 1) and (a.size > 1)

    K = a.size
    sa = np.sort(a)
    x = np.zeros(K)
    x[1:K] = sa[0:K-1] + z * np.exp(1 / sa[0:K-1]) * sa[0:K-1]**0.5
    p = gammaincc(sa[K-1], x[K-1])
    for k in range(K-1):
        p += quad(_gammapmax, x[k], x[k+1], args=(sa,k))[0]
    return p

def d0(a, b):
    if (a == b):
        return 0
    return 1

def pmax_loss(x, q, prior=1.0):
    Pa = np.sum(x * q) / np.sum(x)
    aa = x * q + prior
    aa0 = np.sum(aa)
    pa = aa / aa0
    Pb = np.sum(x * (1 - q)) / np.sum(x)
    ab = x * (1 - q) + prior
    ab0 = np.sum(ab)
    pb = ab / ab0
    a = x + prior
    a0 = np.sum(a)
    p = a / a0
    kp = np.argmax(p)
    bias = Pa * d0(kp, np.argmax(pa)) + Pb * d0(kp, np.argmax(pb))
    return ((2 - dir_pmax(aa) - np.amax(pa)) * Pa +
            (2 - dir_pmax(ab) - np.amax(pb)) * Pb -
            (2 - dir_pmax(a) - np.amax(p) + bias))

def gini_loss(x, q):
    xa = x * q
    na = np.sum(xa)
    pa = xa / na
    xb = x * (1 - q)
    nb = np.sum(xb)
    pb = xb / nb
    n = na + nb
    p = x / n
    return ((1 - np.sum(pa**2)) * na +
            (1 - np.sum(pb**2)) * nb -
            (1 - np.sum(p**2)) * n) / n

def entropy_loss(x, q):
    xa = x * q
    na = np.sum(xa)
    pa = xa / na
    xb = x * (1 - q)
    nb = np.sum(xb)
    pb = xb / nb
    n = na + nb
    p = x / n
    return ((-np.sum(pa * np.log2(pa + 1e-8))) * na +
            (-np.sum(pb * np.log2(pb + 1e-8))) * nb -
            (-np.sum(p * np.log2(p + 1e-8))) * n) / n

K = 3
N = [ 10, 30, 100, 1000 ]
Nn = len(N)
Nx = 400
xl = (-11, 6)
fsave = True
pz = (8,6)
dpi = 300
lw = 0.8

w = np.array([ 0.07, 0.75, 0.18 ])
m = np.array([ -8.0, -0.5, 2.5 ])
s = np.array([ 0.3, 1.75, 0.75 ])

cn = [ 'brown', 'cadetblue', 'darkgoldenrod', 'navy' ]
cs = [ 'r', 'g', 'b' ]

rc = {'font.family' : 'serif', 
      'mathtext.fontset' : 'stix'}
plt.rcParams.update(rc)
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams["font.serif"]

x = np.linspace(xl[0], xl[1], num=Nx)
w0 = np.sum(w)
m0 = np.sum(w * m) / w0
v0 = np.sum(w * (s**2 + (m - m0)**2)) / w0

Ld0 = np.zeros((Nx,Nn))
Ld2 = np.zeros((Nx,Nn))
Lg = np.zeros((Nx,Nn))
Lh = np.zeros((Nx,Nn))

q = np.zeros(K)
mn = np.zeros((K,Nn))
sn = np.zeros((K,Nn))
for j in range(Nn):
    W = w * N[j]
    for k in range(K):
        mi, vi = alg.nix_infer(alg.Cumulator(W[k], m[k], W[k] * s[k]**2),
            (m0, v0))
        mn[k,j] = mi
        sn[k,j] = vi**0.5
    for i in range(Nx):
        for k in range(K):
            q[k] = norm.cdf(x[i], loc=mn[k,j], scale=sn[k,j])
        Ld0[i,j] = pmax_loss(W, q, prior=1.0)
        Ld2[i,j] = alg.dirichlet_partition_loss(W, q, prior=1.0)
        Lg[i,j] = gini_loss(W, q)
        Lh[i,j] = entropy_loss(W, q)


blx = np.concatenate((np.array([xl[1], xl[0]]), x))
bly = np.zeros(2)
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        k = 2 * i + j
        for l in range(K):
            axs[i][j].fill(blx, np.concatenate((bly, w[l] * norm.pdf(x, loc=mn[l,k], scale=sn[l,k]))), cs[l], alpha=0.2)
        for l in range(K):
            axs[i][j].plot(x, w[l] * norm.pdf(x, loc=mn[l,k], scale=sn[l,k]), '-', c=cs[l], linewidth=0.5, alpha=0.5)
        axs[i][j].hlines(0, xl[0], xl[1], colors='black', linewidth=0.5)
        axs[i][j].plot(x, -Lh[:,k], ':', c=cn[1], linewidth=lw, label='$H(x)$')
        axs[i][j].plot(x, -Lg[:,k], '-.', c=cn[0], linewidth=lw, label='$I_G(x)$')
        axs[i][j].plot(x, -Ld0[:,k], '--', c=cn[2], linewidth=lw, label='$\mathcal{G}_0(x)$ ')
        axs[i][j].plot(x, -Ld2[:,k], '-', c=cn[3], linewidth=lw, label='$\mathcal{G}_2(x)$ ')
        axs[i][j].set_title('$N = ' + str(N[k]) + '$', fontsize='small')
        axs[i][j].xaxis.set_ticklabels([])
        axs[i][j].yaxis.set_ticklabels([])
        if (i == 1):
            axs[i][j].set_xlabel('$x$')
        if (j == 0):
            axs[i][j].set_ylabel('Gain')
        axs[i][j].set_xlim(xl)
        axs[i][j].legend(loc='upper left', fontsize='small')
if fsave:
    plt.gcf().set_size_inches(pz[0],pz[1])
    plt.tight_layout()
    plt.savefig('partition-gain.pdf', dpi=dpi, format='pdf')
    plt.close(fig)
else:
    plt.show()


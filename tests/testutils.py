# test utility routines
#
# T.Makinen terhi.makinen@fmi.fi

import time
import datetime
import warnings
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import gammaincc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import hmm.interfaces as ifc
from hmm.algorithms import dirichlet_uncertainty
from hmm.models import EigType
from hmm.lda import LDAPar
from hmm import hlda

### Configuration ##############################################################

class ImageConfig(object):
    """ Wrapper class for image options.
    """
    def __init__(self):
        self.cmap = 'jet'
        self.shared_v = False
        self.size = (1200,1200)

class MarkerConfig(object):
    """ Wrapper class for marker options.
    """
    def __init__(self):
        self.alpha = 0.5
        self.edgewidth = 0.75
        self.marker = [ 'o', 'p', 'h' ]
        self.size = [ 25.0, 25.0, 25.0 ]
        self.edgecolor = ['k', 'k', 'k']
        self.facecolor = ['r', 'g', 'b']

class FrameConfig(object):
    """ Wrapper class for frame options.
    """
    def __init__(self):
        self.do = False
        self.alpha = 0.5
        self.edgecolor = 'white'
        self.facecolor = 'black'

class SplitterConfig(object):
    """ Wrapper class for split line options.
    """
    def __init__(self):
        self.color = 'silver'
        self.style = ':'
        self.width = 0.75

class PlotOutConfig(object):
    """ Wrapper class for plot output options.
    """
    def __init__(self):
        self.save = True
        self.size = (4,4)
        self.dpi = 600
        self.title = False

class PlotConfig(object):
    """ Wrapper class for plotting options.
    """
    def __init__(self):
        self.margin = 0.3
        self.image = ImageConfig()
        self.splitter = SplitterConfig()
        self.markers = MarkerConfig()
        self.frame = FrameConfig()
        self.out = PlotOutConfig()
        self.bgcolor = (0.92, 0.92, 0.92)
        self.pgamma = 1.0

class Test1Config(object):
    """ Wrapper class for test 1 options.
    """
    def __init__(self):
        self.name = 'basic partitioning'
        self.run = True
        self.nodes = True
        self.dendro = True

class Test2Config(object):
    """ Wrapper class for test 2 options.
    """
    def __init__(self):
        self.name = '2D auto density'
        self.run = True
        self.nodes = True
        self.complement = True
        self.uncertainty = True
        self.valea = (0.0, 0.45)
        self.vepis = (0.0, 0.165)

class Test3Config(object):
    """ Wrapper class for test 3 options.
    """
    def __init__(self):
        self.name = '2D shared density'
        self.run = True

class ClassData(object):
    """ Wrapper class for data with assigned class.
    """
    def __init__(self, x, y, names, norm=True):
        self.nclass = len(names)
        self.ndata = x.shape[0]
        self.ndim = x.shape[1]
        self.x = x.copy()
        if norm:
            self.x -= np.mean(x, axis=0).reshape((1,-1))
            self.x /= np.std(x, axis=0).reshape((1,-1))
        self.y = y
        self.names = names

class BlobConfig(object):
    """ Wrapper class for blob options.
    """
    def __init__(self):
        self.n = 1
        self.r = 0.1

class TractorConfig(object):
    """ Wrapper class for tractor options.
    """
    def __init__(self):
        self.a = (0.0, 1.0)
        self.c = 2.0
        self.n = 4
        self.r = 0.5
        self.s = (1.0, 3.0)
        self.v = (0.4, 1.0)

class PerfDataConfig(object):
    """ Wrapper class for performance test data options.
    """
    def __init__(self):
        self.n_data = 6400  # per class
        self.n_classes = 3
        self.n_dim_inform = 2
        self.n_dim_random = 2
        self.distance = (1.5, 2.5)
        self.blob = BlobConfig()
        self.tractor = TractorConfig()
        self.rotate = True
        self.oom_data = True
        self.oom_extend = 1.5
        self.random_state = None

class PerfPlotConfig(object):
    """ Wrapper class for performance test plot options.
    """
    def __init__(self):
        self.data = True
        self.nmax = 256
        self.ROC = True
        self.AS = True
        self.scores = True
        self.save = True
        self.markersize = 3
        self.mew = 0.5
        self.size = (6,6)
        self.dpi = 300

class Classifier(object):
    """ Wrapper class for classifier information.
    """
    def __init__(self, algo, cid, name, c, ls, mk, dist=False):
        self.algo = algo
        self.cid = cid
        self.name = name
        self.c = c
        self.ls = ls
        self.mk = mk
        self.is_dist = dist

### Algorithms #################################################################

def orthox(D, seed=None):
    """ Return a random set of D orthogonal row vectors of size D in descending
        order of vector length.
    """
    if (seed is not None):
        np.random.seed(seed)
    x = np.random.randn(D**2).reshape((D,D))
    d = np.zeros(D)
    for i in range(D):
        while (d[i] == 0):
            for j in range(i):
                x[i,:] -= np.sum(x[i,:] * x[j,:]) * x[j,:]
            d[i] = np.sum(x[i,:]**2)**0.5
            if (d[i] == 0):
                x[i,:] = np.random.randn(D)
        x[i,:] /= d[i]
    d = np.sort(d)[::-1]
    return x * d.reshape((D,1))

def projx(D, S, seed=None):
    """ Return a random projection (column) matrix from D to S <= D dimensions.
    """
    if (seed is not None):
        np.random.seed(seed)
    x = np.random.randn(D*S).reshape((D,S))
    for i in range(S):
        for j in range(i):
            x[:,i] -= np.sum(x[:,i] * x[:,j]) * x[:,j]
        x[:,i] /= np.sum(x[:,i]**2)**0.5
    return x

def shortstr(s):
    """ Return an ordered shortcut version of a sequence of integers, indicating
        duplicates and continuous ranges.
        Input:
            s : ndarray(N) of int, sequence
        Output:
            str, human-readable sorted shortcut of the sequence
    """
    g = []
    for x in np.sort(s):
        if (len(g) == 0):
            g.append([x])
        else:
            if (x == g[-1][-1]):
                if (x == g[-1][0]):
                    g[-1].append(x)
                else:
                    g[-1].pop()
                    g.append([x, x])
            elif (x == g[-1][-1] + 1):
                if ((len(g[-1]) == 1) or (g[-1][0] < g[-1][-1])):
                    g[-1].append(x)
                else:
                    g.append([x])
            else:
                g.append([x])
    l = []
    for x in g:
        if (len(x) == 1):
            l.append(str(x[0]))
        elif (x[0] == x[-1]):
            l.append(str(x[0]) + 'x' + str(len(x)))
        else:
            l.append(str(x[0]) + '-' + str(x[-1]))
    return '[' + ', '.join(l) + ']'

def _pn_proj(p, n, m, R):
    pr = np.matmul(p - m, R)
    b = p - m - np.matmul(pr, R.T)
    nr = np.matmul(n, R)
    dnr = np.sum(nr**2)**0.5
    if (dnr == 0):
        pp = np.zeros(R.shape[1])
        nn = np.zeros(R.shape[1])
        nn[0] = 1
        if (np.sum(n * b) > 0):
            pp[0] = 1e6
        else:
            pp[0] = -1e6
        return (pp, nn)
    nr /= dnr
    db = np.sum(b**2)**0.5
    if (db == 0):
        return (pr.flatten(), nr.flatten())
    n2 = np.matmul(nr, R.T)
    n2 /= np.sum(n2**2)**0.5
    ca = np.sum(n * n2)
    ar = (db * (1.0 - ca**2)**0.5 / ca) * nr
    if (np.sum(n * b) > 0):
        qr = pr + ar
    else:
        qr = pr - ar
    return (qr.flatten(), nr.flatten())

def normalize(x):
    """ Scale multivariate data to have zero mean and unit standard deviation.
        Input:
            x : ndarray(N,D) data
        Output:
            tuple(3) scaled data, mean and standard deviation
    """
    xm = np.mean(x, axis=0).reshape((1,-1))
    xs = np.std(x, axis=0).reshape((1,-1))
    return ((x - xm) / xs, xm, xs)

def rrot(D):
    """ Create a random D-dimensional rotation matrix.
        Input:
            D : int, dimension
        Output:
            ndarray(D,D), rotation matrix
    """
    R = np.random.randn(D*D).reshape((D,D))
    for i in range(D):
        for j in range(i):
            R[:,i] -= np.sum(R[:,i] * R[:,j]) * R[:,j]
        R[:,i] /= np.sum(R[:,i]**2)**0.5
    return R

def randnball(D, num=1):
    """ Return uniformly distributed points in a D-dimensional ball with unit
        radius.
        Input:
            D   : int, dimension
            num : int, number of returned points
        Output:
            ndarray(num, D)
    """
    x = np.random.randn(num*D).reshape((num,D))
    x *= (np.random.rand(num)**(1/(1+D)) /
        np.sum(x**2, axis=1)**0.5).reshape((-1,1))
    return x

def randblobs(N, D, c=None, num=2, r=1.0):
    """ Create a random assortment of isotropic multivariate normal blobs.
        Input:
            N   : int, number of data points
            D   : int, dimensionality
            c   : ndarray(num,D) blob centers, or None
            num : int, number of blobs
            r   : float, volume ratio, or
                  ndarray(num), blob scales
    """
    if (c is None):
        cc = randnball(D, num=num)
    else:
        cc = c
    if isinstance(r, np.ndarray):
        R = r
    else:
        X = np.random.rand(num)
        R = (1 + 2 * (X - 0.5) * abs(X - 0.5)) * (r / num)**(1 / D)
    y = np.random.randint(num, size=N)
    x = np.random.randn(N*D).reshape((N,D))
    for k in range(num):
        sx = (y == k)
        if (np.count_nonzero(sx) > 0):
            x[sx,:] *= R[k]
            x[sx,:] += cc[k,:].reshape((1,D))
    x, xm, xs = normalize(x)
    return x

def mindist(x):
    """ Find the per-point minimal interpoint distance.
        Input:
            x : ndarray(N,D), data points
        Output:
            tuple(2) of minimal distance, respective point index
    """
    N, D = x.shape
    d2 = (np.sum((x.reshape((N,1,D)) - x.reshape((1,N,D)))**2, axis=2) +
        np.eye(N) * 1e8)
    ix = np.argmin(d2, axis=0)
    return (d2[np.arange(N),ix]**0.5, ix)

def centroids(N, D, dlim):
    """ Create D-dimensional data points with the given interpoint distance
        characteristics.
        Input:
            N    : int, number of points
            D    : int, number of dimensions
            dlim : tuple(2) of float, per-point minimal and maximal interpoint
                   distance
        Output:
            ndarray(N,D) set of points
    """
    dg = dlim[1] / dlim[0]
    c = randnball(D, num=N)
    d, ix = mindist(c)
    nx = np.argmin(d)
    xx = np.argmax(d)
    while (d[xx] > dg * d[nx]):
        cc = np.mean(c, axis=0)
        cm = 0.5 * (c[nx,:] + c[ix[nx],:]) - cc
        dm = np.sum(cm**2)**0.5
        cm /= dm 
        cn = c[nx,:] - c[ix[nx],:]
        cn /= np.sum(cn**2)**0.5
        dn = d[xx] / dg
        cx = 0.5 * (np.sum(cm * cn) + 1)
        c[nx,:] = cc + dm * cm + cx * dn * cn
        c[ix[nx],:] = cc + dm * cm - (1 - cx) * dn * cn
        d, ix = mindist(c)
        nx = np.argmin(d)
        xx = np.argmax(d)
    return c * (dlim[0] / d[nx])

class ConvexHull2D(object):
    """ Convex 2D boundary object.
        Input:
            lims   : arraylike(4) xmin, xmax, ymin, ymax, initial rectangle
            margin : float, margin added to the initial rectangle
    """
    def __init__(self, lims, margin=1.0):
        assert ((isinstance(lims, np.ndarray) and
                 (lims.ndim == 1) and
                 (lims.size == 4)
                ) or
                (isinstance(lims, (list, tuple)) and
                 (len(lims) == 4)
                )
               )
        assert isinstance(margin, (int, float))

        xa = lims[0] - margin
        xb = lims[1] + margin
        xm = 0.5 * (xa + xb)
        ya = lims[2] - margin
        yb = lims[3] + margin
        ym = 0.5 * (ya + yb)

        # constant outer hull

        self.oh = np.array([[xa - margin, ya - margin],
                            [xa - margin, yb + margin],
                            [xb + margin, yb + margin],
                            [xb + margin, ya - margin]])

        # variable inner hull

        self.ih = [[[xm, ya], [ 0,  1], [xa, ya]],
                   [[xb, ym], [-1,  0], [xb, ya]],
                   [[xm, yb], [ 0, -1], [xb, yb]],
                   [[xa, ym], [ 1,  0], [xa, yb]]]

    def cut(self, p, n):
        """ Partition the hull.
            Input:
                p : ndarray(2), anchor
                n : ndarray(2), normal
        """
        N = len(self.ih)
        qs = []
        nv = np.zeros(N)
        for i in range(N):
            pi = np.array(self.ih[i][0])
            ni = np.array(self.ih[i][1])
            vl = np.array(self.ih[i][2])
            vr = np.array(self.ih[(i+1) % N][2])
            nv[i] = np.sum(n * (vl - p))
            b = np.sum((p - pi) * ni) * ni
            db = np.sum(b**2)**0.5
            nr = n - np.sum(n * ni) * ni
            dnr = np.sum(nr**2)**0.5
            if (dnr > 0):
                nr /= dnr
                csa = np.sum(n * nr)
                if (sum(n * b) > 0):
                    q = p - b + (db * (1 - csa**2)**0.5 / csa) * nr
                else:
                    q = p - b - (db * (1 - csa**2)**0.5 / csa) * nr
                if (np.sum((q - pi) * (vl - pi)) > 0):
                    if (np.sum((q - pi)**2) <= np.sum((vl - pi)**2)):
                        qs.append([ i, q ])
                else:
                    if (np.sum((q - pi)**2) < np.sum((vr - pi)**2)):
                        qs.append([ i, q ])
        if (len(qs) < 2):
            if (np.amax(nv) <= 0):
                self.ih = []
            return

        pm = 0.5 * (qs[0][1] + qs[1][1])
        nr = np.array([n[1], -n[0]])
        ih = []
        if (np.sum(nr * (qs[0][1] - pm)) < 0):
            il = 0
        else:
            il = 1
        istop = qs[il][0]
        q0 = qs[il][1]
        i = qs[1-il][0]
        q1 = qs[1-il][1]
        ih.append([[pm[0], pm[1]], [n[0], n[1]], [q0[0], q0[1]]])
        v = np.array(self.ih[(i+1) % N][2])
        pm = 0.5 * (q1 + v)
        ih.append([[pm[0], pm[1]], self.ih[i][1], [q1[0], q1[1]]])
        done = False
        while not done:
            i = (i+1) % N
            if (i == istop):
                v = np.array(self.ih[i][2])
                if (np.sum((q0 - v)**2)**0.5 > 1e-6):
                    pm = 0.5 * (q0 + v)
                    ih.append([[pm[0], pm[1]], self.ih[i][1], self.ih[i][2]])
                done = True
            else:
                ih.append(self.ih[i])
        self.ih = ih

    def path(self, annulus=True):
        """ Return a path of either the convex hull or an annulus around the
            convex hull.
            Input:
                annulus : bool, if True, return an annulus, else the hull
            Output:
                matplotlib.path.Path object, or None
        """
        coh = np.ones(4, dtype=mpath.Path.code_type) * mpath.Path.LINETO
        coh[0] = mpath.Path.MOVETO
        N = len(self.ih)
        if (N == 0):        # no inner hull
            if annulus:
                return mpath.Path(self.oh, coh)
            else:
                return None
        cih = np.ones(N+1, dtype=mpath.Path.code_type) * mpath.Path.LINETO
        cih[0] = mpath.Path.MOVETO
        xy = np.zeros((N+1,2))
        for i, seg in enumerate(self.ih):
            xy[i,0] = seg[2][0]
            xy[i,1] = seg[2][1]
        xy[N,:] = xy[0,:]
        if annulus:
            return mpath.Path(np.concatenate((self.oh, xy)),
                np.concatenate((coh, cih)))
        else:
            return mpath.Path(xy, cih)

    @property
    def size(self):
        """ Return the number of hull segments.
        """
        return len(self.ih)

class Tractor(object):
    """ Data modifier that rotates and either attracts or repels points.
        Input:
            x0    : ndarray(1,D), center of the effect
            scale : float, effective radius
            alim  : float, limits of the attraction or repulsion factor
            rmax  : float, maximal rotation factor
            vlim  : tuple(2) of float, lower and upper limits for eigenvalues
                    as fractions of scale
    """
    def __init__(self, x0, scale, alim=(-0.5, 5.0), rmax=1.0, vlim=(0.2, 1.0)):
        D = x0.size
        self._x0 = x0.reshape((1,-1))
        self._s = scale
        self._a = alim[0] + (alim[1] - alim[0]) * np.random.rand()
        X = np.random.rand()
        self._ar = rmax * (0.5 + (X - 0.5) * abs(X - 0.5)**0.5)
        self._v = ((vlim[0] + (vlim[1] - vlim[0]) * np.random.rand(D))
            * scale).reshape((1,D))
        self._P = rrot(D)
        self._R = rrot(D)

    def tract(self, x):
        z = np.matmul(x - self._x0, self._P) / self._v
        r = np.sum(z**2, axis=1).reshape((-1,1))**0.5
        fr = self._ar * np.exp(-0.5 * (r / self._s)**2)
        z = (1 - fr) * z + fr * np.matmul(z, self._R)
        z *= 1 + self._a * np.exp(-r / self._s)
        return np.matmul(z * self._v, self._P.T) + self._x0

class TractorGen(object):
    """ Tractor generator.
        Input:
            D    : int, dimension
            alim : tuple(2), range of the attraction / repulsion factor
            cmax : float, maximal central offset
            rmax : float, maximal rotation factor
            slim : tuple(2), scale range
            vlim : tuple(2) of float, range of eigenvalues as fractions of scale
    """
    def __init__(self, D, alim=(-0.5, 3.0), cmax=2.0, rmax=0.5, slim=(1.0, 3.0),
        vlim=(0.2, 1.0)):
        self._d = D
        self._a = alim
        self._c = cmax
        self._r = rmax
        self._s = (slim[0], slim[1] - slim[0])
        self._v = vlim

    def get(self):
        x = self._c * randnball(self._d)
        s = self._s[0] + self._s[0] * np.random.rand()**self._d
        return Tractor(x, s, alim=self._a, rmax=self._r, vlim=self._v)

class BlobGen(object):
    """ randblobs() generator data.
        Input:
            N : int, number of blobs
            R : float, volume ratio
    """
    def __init__(self, N, R):
        self.n = N
        self.r = R

def make_class_data(n_data, n_classes=3, n_dim_inform=2, n_dim_random=0,
    random_state=None, blobgen=None, tractorgen=None, n_tractors=5,
    distance=(1.0, 2.0)):
    """ Create a multiclass data set for algorithm testing.
        Input:
            n_data         : int, number of data points per class
            n_classes      : int, number of classes
            n_dim_inform   : int, number of dimensions with class data
            n_dim_random   : int, number of dimensions filled with random normal
                             distributed values
            random_state   : int, random number generator seed
            blobgen        : BlobGen object or None
            tractorgen     : TractorGen object or None
            n_tractors     : int, number of applied tractors per class
            distance       : tuple(2) class centroid minimal distance range
        Output:
            tuple(2) of ndarray(N,D), ndarray(N) of int, data and class label
    """

    n = n_data
    N = n * n_classes
    D = n_dim_inform
    Dmax = D + n_dim_random
    K = n_classes
    if (random_state is not None):
        np.random.seed(random_state)
    if (blobgen is None):
        BG = BlobGen(10, 0.5)
    else:
        BG = blobgen
    if (n_tractors > 0):
        if (tractorgen is None):
            TG = TractorGen(D)
        else:
            TG = tractorgen

    # create all generators before the data so that the number of data points
    # does not affect class distributions

    gens = []
    for k in range(K):
        u = np.random.rand(BG.n)
        gens.append({'blobc': randnball(D, num=BG.n),
                     'blobr': ((1 + 2 * (u - 0.5) * abs(u - 0.5)) *
                              (BG.r / BG.n)**(1 / D)),
                     'tract': [ TG.get() for i in range(n_tractors) ],
                     'clasc': centroids(K, D, distance) })

    # draw the data

    X = np.zeros((N,Dmax))
    if (D < Dmax):
        X[:,D:Dmax] = np.random.randn(N*(Dmax-D)).reshape((N,Dmax-D))
    Y = np.zeros(N, dtype=np.int)
    for k in range(K):
        x = randblobs(n_data, D, c=gens[k]['blobc'], num=BG.n,
            r=gens[k]['blobr'])
        T = gens[k]['tract']
        for i in range(n_tractors):
            x = T[i].tract(x)
        x, xm, xs = normalize(x)
        x += gens[k]['clasc'][k,:].reshape((1,-1))
        X[n*k:n*(k+1),0:D] = x
        Y[n*k:n*(k+1)] = k
    X, Xm, Xs = normalize(X)
    return (X, Y)

def make_shell(N, D, r0=1.0, rt=1.0):
    """ Create data in a shell.
        Input:
            N  : int, number of data points
            D  : int, number of dimensions
            r0 : float, shell inner radius
            rt : float, shell thickness
        Output:
            ndarray(N,D)
    """
    X = np.random.randn(N*D).reshape((N,D))
    X /= np.sum(X**2, axis=1).reshape((N,1))**0.5
    X *= r0 + rt * np.random.rand(N).reshape((N,1))
    return X

def atrapez(pps, pos):
    x0 = pps[-2]
    x1 = pps[-1]
    if (x0 == x1):
        return 0.0
    y0 = pos[-2]
    y1 = pos[-1]
    a = y0 - x0
    b = y1 - x1
    if (a * b > 0):
        return 0.5 * (x1 - x0) * abs(a + b)
    return 0.5 * (x1 - x0) * (a**2 + b**2) / (abs(a) + abs(b))

def reliability(rs, K, nmin=10, MCE=True):
    pp = rs[0,:].flatten()      # mean(p)
    vp = rs[1,:].flatten()      # mean(p * q)
    nt = rs[2,:].flatten()      # N_True
    n = nt + rs[3,:].flatten()  # N_Total
    pmin = rs[4,:].flatten()    # min(p)
    pmax = rs[5,:].flatten()    # max(p)
    N = np.sum(n)
    vex = np.sum(n * vp) / N
    po = nt.copy()
    sx = (n > 0)
    if (np.count_nonzero(sx) > 0):
        po[sx] /= n[sx]
    if MCE:
        mce = 0.0
        mace = 0.0
        pps = []
        pos = []
        for i in range(pp.size):
            if (n[i] >= nmin):
                mce += nt[i] - n[i] * pp[i]
                mace += abs(nt[i] - n[i] * pp[i])
                pps.append(pp[i])
                pos.append(po[i])
        return (vex, (mce / N, mace / N), n, np.array(pps), np.array(pos))
    else:
        paw = 0.0
        pas = 0.0
        pps = [ 0.0 ]
        pos = [ 0.0 ]
        is_zero = True
        i_nozero = -1
        for i in range(pp.size):
            if (n[i] >= nmin):
                i_nozero = i
                if is_zero:
                    pps.append(pmin[i])
                    pos.append(0.0)
    #                pas += atrapez(pps, pos)
                    is_zero = False
                pps.append(pp[i])
                pos.append(po[i])
                paw += n[i]
                pas += n[i] * atrapez(pps, pos)
        if (i_nozero >= 0):
            pps.append(pmax[i_nozero])
            pos.append(1.0)
    #        pas += atrapez(pps, pos)
        pps.append(1.0)
        pos.append(1.0)
    #    pas += atrapez(pps, pos)
        a = 1 / K
        b = 1 - a
        if (paw > 0.0):
            pas /= paw
        pas = 1 - pas / (0.5 * (a**2 + b**2))
        return (vex, pas, n, np.array(pps), np.array(pos))

def f1_macro_ovr(y, y_pred, K):
    TP = np.zeros(K)
    FP = np.zeros(K)
    FN = np.zeros(K)
    for k in range(K):
        TP[k] = np.count_nonzero((y_pred == k) & (y == k))
        FP[k] = np.count_nonzero((y_pred == k) & (y != k))
        FN[k] = np.count_nonzero((y_pred != k) & (y == k))
    return np.mean(2 * TP / (2 * TP + FP + FN))

def skill_metrics(y, xy, pom, is_dst=False, n_prange=10, n_vrange=5, n_cdf=300):
    """
    """
    if is_dst:
        ay = np.matmul(np.sum(xy, axis=1).reshape((y.size,1)),
            np.ones((1,xy.shape[1])))
        py = xy / ay
    else:
        py = xy

    N = y.size
    K = xy.shape[1]
    N_noskill = N / K
    y_pred = np.argmax(py, axis=1)

    # Accuracy

    ACC = np.count_nonzero(y == y_pred) / N

    # Heidke skill score

#    HSS = (np.count_nonzero(y == y_pred) - N_noskill) / (N - N_noskill)

    # macro-averaged OvR F1 score

    F1 = f1_macro_ovr(y, y_pred, K)

    # micro-averaged OvR ROC-AUC

    label_binarizer = LabelBinarizer().fit(y)
    ytr = label_binarizer.transform(y).ravel()
    pyr = py.ravel()
    ROC = roc_curve(ytr, pyr)
    AUC = roc_auc_score(ytr, pyr)

    # calibration area score CAS and probability frequency PPN

    prng = np.linspace(0, 1, num=n_prange+1)
    w = prng[1] - prng[0]
    PRS = np.zeros((6,n_prange,1))
    if is_dst:
        PPN = np.zeros((n_prange,n_vrange))
        ayr = ay.ravel()
        ayl = np.sort(ayr)[np.linspace(0,ayr.size-1,num=n_vrange+1).astype(int)]
    else:
        PPN = np.zeros((n_prange,1))
    for i in range(n_prange):
        sx = (pyr >= prng[i]) & (pyr < prng[i+1])
        nx = np.count_nonzero(sx)
        if (nx > 0):
            pyx = pyr[sx]
            ysx = ytr[sx]
            PRS[0,i,0] = np.mean(pyx)
            PRS[1,i,0] = np.mean(pyx * (1 - pyx))
            PRS[2,i,0] = np.count_nonzero(ysx)
            PRS[3,i,0] = nx - PRS[2,i,0]
            PRS[4,i,0] = np.amin(pyx)
            PRS[5,i,0] = np.amax(pyx)
            if is_dst:
                ayx = ayr[sx]
                for j in range(n_vrange):
                    tx = (ayx >= ayl[j]) & (ayx < ayl[j+1])
                    PPN[i,j] = np.count_nonzero(tx)
            else:
                PPN[i,0] = nx
    CAS = reliability(PRS[:,:,0].reshape((6,n_prange)), K)

    # negative detection score NDS and cumulative functions PCF, NCF

    if (pom is None):
        nds = ncf = pcf = None
    else:
        posx = np.sort(np.amax(py, axis=1))
        pomx = np.sort(pom)
        pofs = 1e-8 - 1 / K
        nds = 1.0 - (np.mean(pomx) + pofs) / (np.mean(posx) + pofs)
        if (posx.size <= n_cdf):
            pcf = posx
        else:
            pcf = posx[np.linspace(0, posx.size-1, num=n_cdf, dtype=np.int)]
        if (pomx.size <= n_cdf):
            ncf = pomx
        else:
            ncf = pomx[np.linspace(0, pomx.size-1, num=n_cdf, dtype=np.int)]

    return {'ACC': ACC, 'F1': F1, 'ROC': ROC, 'AUC': AUC, 'CAS': CAS,
        'NDS': nds, 'CFS': (pcf, ncf), 'PPN': PPN}

def kfoldtest(Ncl, Xc, Y, Z, C, K=5, frac=1):
    """ Run a k-fold test set on data and out-of-model data with all the
        provided classifiers.
        Input:
            Ncl  : int, number of classes
            Xc   : ndarray(N,D), data
            Y    : ndarray(N) of int, class labels
            Z    : ndarray(M,D) or None, out-of-model data
            C    : list of Classifier objects
            K    : int, number of folds
            frac : int, number of data divisions
        Output:
            tuple(3) of
                ndarray(W) of int, true labels
                ndarray(W,Ncl*Nmd) class probabilities per class and
                    classifier
                ndarray(M,Nmd) out-of-model maximal class probability per
                    classifier
    """
    D = Xc.shape[1]
    Nmd = len(C)
    X = []
    Nx = []
    Nk = []
    Nf = []
    for c in range(Ncl):
        sx = (Y == c)
        X.append(Xc[sx,:].reshape((-1,D)))
        Nx.append(X[c].shape[0])
        Nk.append(Nx[c] // K)
        Nf.append(Nk[c] // frac)

    # NB: train (1) and test (K-1) sets are reversed relative to the usual
    #     k-fold configuration to test the classifier behavior at the small data
    #     limit

    Y_lst = []
    Yp_lst = []
    P_oomax = []

    nx_train = np.zeros(Nmd, dtype=np.int)
    tx_train = np.zeros(Nmd)
    nx_test = np.zeros(Nmd, dtype=np.int)
    tx_test = np.zeros(Nmd)

    for k in range(K):
        for j in range(frac):
            Xtr = []
            Ytr = []
            Xts = []
            Yts = []
            for c in range(Ncl):
                for u in range(K):
                    fa = u * Nk[c] + j * Nf[c]
                    fb = fa + Nf[c]
                    if (u == k):
                        Xtr.append(X[c][fa:fb,:].reshape((-1,D)))
                        Ytr.append(np.zeros(Nf[c], dtype=np.int) + c)
                    else:
                        Xts.append(X[c][fa:fb,:].reshape((-1,D)))
                        Yts.append(np.zeros(Nf[c], dtype=np.int) + c)
            X_train = np.vstack(Xtr)
            X_test = np.vstack(Xts)
            Y_train = np.hstack(Ytr)
            Y_test = np.hstack(Yts)
            Pp = np.zeros((X_test.shape[0],Ncl*Nmd))
            if (Z is not None):
                Pn = np.zeros((Z.shape[0],Nmd))
            for i, M in enumerate(C):

                tix = time.process_time()
                M.algo.fit(X_train, Y_train)
                tx_train[i] += time.process_time() - tix
                nx_train[i] += Y_train.size

                if M.is_dist:

                    tix = time.process_time()
                    Pp[:,Ncl*i:Ncl*(i+1)] = M.algo.predict_para(X_test)
                    tx_test[i] += time.process_time() - tix
                    nx_test[i] += X_test.shape[0]

                    if (Z is not None):
                        p = M.algo.predict_para(Z)
                        p /= np.sum(p, axis=1).reshape((-1,1))
                        Pn[:,i] = np.amax(p, axis=1)
                else:

                    tix = time.process_time()
                    Pp[:,Ncl*i:Ncl*(i+1)] = M.algo.predict_proba(X_test)
                    tx_test[i] += time.process_time() - tix
                    nx_test[i] += X_test.shape[0]

                    if (Z is not None):
                        Pn[:,i] = np.amax(M.algo.predict_proba(Z), axis=1)
            Y_lst.append(Y_test)
            Yp_lst.append(Pp)
            if (Z is not None):
                P_oomax.append(Pn)

    if (Z is None):
        POMAX = np.zeros(0)
    else:
        POMAX = np.vstack(P_oomax)
    tixx = (nx_train, tx_train, nx_test, tx_test)
    return (np.hstack(Y_lst), np.vstack(Yp_lst), POMAX, tixx)

class CustomGNB(object):
    def __init__(self):
        self.n_features = 0
        self.var_smoothing = 1e-9
        self.dirichlet_prior = 1.0

    def fit(self, X, Y):
        N, D = X.shape
        self.n_features = D
        self.classes = np.unique(Y)
        K = self.classes.size
        self.n = np.zeros(K)
        self.m = np.zeros((K,1,D))
        self.v = np.zeros((K,D))
        for k in range(K):
            sx = (Y == self.classes[k])
            self.n[k] = np.count_nonzero(sx)
            Xk = X[sx,:].reshape((-1,D))
            self.m[k,0,:] = np.mean(Xk, axis=0)
            self.v[k,:] = np.var(Xk, axis=0)
        self.v += self.var_smoothing * np.amax(self.v, axis=0).reshape((1,D))
        self.iv = 1.0 / self.v.reshape((K,D,1))

    def predict_para(self, X):
        N, D = X.shape
        K = self.classes.size
        return self.dirichlet_prior + self.n * gammaincc(0.5 * self.n_features,
            0.5 * np.matmul((X.reshape((1,N,D)) - self.m)**2,
            self.iv).reshape((K,N)).T)

### Graphics ###################################################################

@ticker.FuncFormatter
def remove_negative(x, pos):
    """ Remove negative tick labels.
    """
    label = ' ' if x < 0 else '{:.1f}'.format(x)
    return label

def square_lims(ax, margin=0.0):
    """ Determine new axis limits to force axes into a centered square.
        Input:
            ax : tuple (xmin, xmax, ymin, ymax), default axis limits
        Output:
            tuple (xmin, xmax, ymin, ymax), square axis limits
    """
    xmin, xmax, ymin, ymax = ax
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    dh = 0.5 * max(xmax - xmin, ymax - ymin) * (1.0 + margin)
    return (cx - dh, cx + dh, cy - dh, cy + dh)

def data_lims(x, margin=0.0):
    """ Determine axis limits from data.
        Input:
            x      : ndarray(N,2), data
            margin : float, fractional margin added to the limits
        Output:
            tuple (xmin, xmax, ymin, ymax), axis limits
    """
    xmin = np.amin(x, axis=0)
    xmax = np.amax(x, axis=0)
    cx = 0.5 * (xmax[0] + xmin[0])
    hx = 0.5 * (xmax[0] - xmin[0]) * (1.0 + margin)
    cy = 0.5 * (xmax[1] + xmin[1])
    hy = 0.5 * (xmax[1] - xmin[1]) * (1.0 + margin)
    return (cx - hx, cx + hx, cy - hy, cy + hy)

def plot_split(packet, node, proj=None, side=''):
    hdr = packet.header
    cfg, testid, saveid = hdr.get_collection('splot')[0]
    plot_comp = False
    postsplit = False
    if (testid == cfg.test2.name):
        plot_comp = cfg.test2.complement
    K = cfg.data.nclass
    if (proj is None):
        n = packet.get_attr('_model_normal')
        p = packet.get_attr('_model_pivot')
        if ((n is None) or (p is None)):
            return  # undefined projection
        z = None
    else:
        p, n, z = proj
    X = packet.get_attr('_train_data')
    if ((X is not None) and (X.n > 0)):
        x = X.x - p
        xx = np.matmul(x, n)
        if (z is None):
            xr = x - np.matmul(xx, n.T)
            if (np.sum(xr**2) > 0):
                v, P = np.linalg.eigh(np.matmul(xr.T, xr))
                z = P[:,np.argmax(v)].reshape((-1,1))
    else:
        xx = xy = xz = np.zeros(0)
    C = packet.get_attr('_complement')
    if ((C is not None) and (C.n > 0) and plot_comp):
        c = C.x - p
        cx = np.matmul(c, n)
        if (z is None):
            cr = c - np.matmul(cx, n.T)
            if (np.sum(cr**2) > 0):
                v, P = np.linalg.eigh(np.matmul(cr.T, cr))
                z = P[:,np.argmax(v)].reshape((-1,1))
            else:
                return  # unable to determine z
        cz = np.matmul(c, z)
        cy = C.yxt
        cix = np.zeros(C.n, dtype=np.int)
        for k in range(K):
            cix[cy[k]:cy[k+1]] = k
        cm = (np.hstack((cx, cz)), cix)
    else:
        cm = None
    if (z is None):
        return  # unable to determine z
    if ((X is not None) and (X.n > 0)):
        xz = np.matmul(x, z)
        xy = X.yxt
        xix = np.zeros(X.n, dtype=np.int)
        for k in range(K):
            xix[xy[k]:xy[k+1]] = k
    fname = cfg.outdir + saveid + '-node-' + str(node.nid) + side + '.pdf'
    title = 'Node ' + str(node.nid) + side
    cfg.plot_data(data=(np.hstack((xx, xz)), xix), complement=cm,
        labels=(title, '$\hat{p}_{0}$', '$\hat{p}_{1}$'), name=fname,
        split=True)
    if (postsplit and (side == '')):
        subs = packet.get_attr('_subpackets')
        if (subs[0] is not None):
            plot_split(subs[0], node, proj=(p, n, z),
                side='-' + str(node.left.nid))
        if (subs[1] is not None):
            plot_split(subs[1], node, proj=(p, n, z),
                side='-' + str(node.right.nid))

def plot_node(packet, node):

# DEBUG

#    print('PLOT NODE ' + str(node.nid))

    hdr = packet.header
    cfg, testid, saveid = hdr.get_collection('splot')[0]
    plot_comp = False
    if (testid == cfg.test2.name):
        plot_comp = cfg.test2.complement
    par = hdr.par(node)
    K = par.nclass
    lda = node.model.lda
    cw = lda.split_solver.data.cw.reshape((1,K))
    P = lda.sub.copy()

# DEBUG

#    print(P.T)
#    print(np.matmul(P.T, P))

    if node.is_split:
        mm = lda.pivot
    else:
        iwc = np.argmax(lda.split_solver.data.cn)
        mm = (lda.split_solver.data.mean +
            lda.split_solver.data.cmean[iwc,:].reshape((1,-1)))
    xlab = '$\hat{p}_{0}$'
    ylab = '$\hat{p}_{1}$'
    X = packet.get_attr('_train_data')
    if ((X is not None) and (X.n > 0)):
        x = np.matmul(X.x - mm, P)
        xlim = data_lims(x)
        yx = X.yxt
        yxx = np.zeros(x.shape[0], dtype=np.int)
        for k in range(1,yx.size-1):
            yxx[yx[k]:yx[k+1]] = k
        xyx = (x, yxx)
    else:
        xlim = [ 0, 0, 0, 0 ]
        xyx = None
    C = packet.get_attr('_complement')
    if ((C is not None) and (C.n > 0) and plot_comp):
        c = np.matmul(C.x - mm, P)
        clim = data_lims(c)
        lim = square_lims([ min(xlim[0], clim[0]),
                            max(xlim[1], clim[1]),
                            min(xlim[2], clim[2]),
                            max(xlim[3], clim[3])], margin=cfg.plot.margin)
        yc = C.yxt
        ycx = np.zeros(c.shape[0], dtype=np.int)
        for k in range(1,yc.size-1):
            ycx[yc[k]:yc[k+1]] = k
        cyc = (c, ycx)
    else:
        if (x is None):
            return
        lim = square_lims(xlim, margin=cfg.plot.margin)
        cyc = None
    if cfg.plot.frame.do:
        H = ConvexHull2D(lim)
        cutnode = node
        while (cutnode.parent is not None):
            cla = cutnode.parent.model.lda
            cn = cla.normal.T
            if (cutnode is cutnode.parent.left):
                cn = -cn
            pp, nn = _pn_proj(cla.pivot, cn, mm, P)
            H.cut(pp, nn)
            cutnode = cutnode.parent
        if (H.size > 0):
            frame = H.path()
        else:
            frame = None
    else:
        frame = None
    isz = cfg.plot.image.size
    xy = np.zeros((isz[0], isz[1], 2))
    xy[:,:,0] += np.linspace(lim[0], lim[1], isz[0]).reshape((1,isz[0]))
    xy[:,:,1] += np.linspace(lim[3], lim[2], isz[1]).reshape((isz[1],1))
    Q = np.matmul(xy.reshape((isz[0]*isz[1],2)), P.T) + mm
    a = node.model.predict_para(Q, hdr.par(node))
    if ((testid == cfg.test2.name) and cfg.test2.uncertainty):
        U = dirichlet_uncertainty(a)
        if cfg.plot.image.shared_v:
            vla = vle = (np.amin(U), np.amax(U))
        else:
            vla = (min(np.amin(U[:,0]), cfg.test2.valea[0]),
                   max(np.amax(U[:,0]), cfg.test2.valea[1]))
            vle = (min(np.amin(U[:,1]), cfg.test2.vepis[0]),
                   max(np.amax(U[:,1]), cfg.test2.vepis[1]))
        fname = cfg.outdir + saveid + '-alea-node-' + str(node.nid) + '.pdf'
        title = '$U_{al}$ node ' + str(node.nid)
        cfg.plot_data(image=U[:,0].reshape((isz[0],isz[1])),
            labels=(title, xlab, ylab), lims=lim, name=fname, v=vla)
        fname = cfg.outdir + saveid + '-epis-node-' + str(node.nid) + '.pdf'
        title = '$U_{ep}$ node ' + str(node.nid)
        cfg.plot_data(image=U[:,1].reshape((isz[0],isz[1])),
            labels=(title, xlab, ylab), lims=lim, name=fname, v=vle)
    p = a / np.sum(a, axis=1).reshape((-1,1))
    if (cfg.plot.pgamma != 1):
        t = 1.0 / K
        sx = p < t
        p[sx] = t * (1 - (1 - p[sx] / t)**cfg.plot.pgamma)
        sx = p > t
        p[sx] = t + (1 - t) * ((p[sx] - t) / (1 - t))**cfg.plot.pgamma
    fname = cfg.outdir + saveid + '-p-node-' + str(node.nid) + '.pdf'
    title = 'Node ' + str(node.nid)
    cfg.plot_data(image=p.reshape((isz[0],isz[1],3)), data=xyx, complement=cyc,
        frame=frame, labels=(title, xlab, ylab), lims=lim, name=fname,
        split=False, v=(0,1))

def fstr(x):
    if (x < 10):
        return '{:.2f}'.format(x)
    return '{:.1f}'.format(x)

def _add_branch(ax, hdr, node, L, L0, N, x, dx, s):
    """ Draw a single dendrogram branch.
        Input:
            ax   : Axes object
            hdr  : HLDA object
            node : BNode object
            L    : float, loss at the node
            L0   : float, loss at the root node
            N    : int, total number of data points
            x    : float, x coordinate of the node
            dx   : float, branch half-width
            s    : int, branch side indicator, either -1 or 1
    """
    if node.is_split:
        xa = x - dx
        cha = node.left
        wa = cha.model._lda.split_solver.data.cw
        La = cha.model.loss(hdr.par(cha)) * (np.sum(wa) / N)
        xb = x + dx
        chb = node.right
        wb = chb.model._lda.split_solver.data.cw
        Lb = chb.model.loss(hdr.par(chb)) * (np.sum(wb) / N)
        ax.plot([xa, xa, xb, xb], [La, L, L, Lb], '-', c='black')
        xmin, xmaxa = _add_branch(ax, hdr, cha, La, L0, N, xa, 0.5 * dx, -1)
        xminb, xmax = _add_branch(ax, hdr, chb, Lb, L0, N, xb, 0.5 * dx, 1)
        ncl = (0.92, 0.92, 0.92)
    else:
        xmin = xmax = x
        La = Lb = L - 1
        ncl = 'blue'
    w = node.model._lda.split_solver.data.cw
    txt = '(' + ', '.join([fstr(v) for v in w]) + ')'
    ax.text(x, L - 0.025 * L0, txt, rotation=90,
        horizontalalignment='center', verticalalignment='top',
        fontsize=7, fontstretch='condensed')
    if (node.parent is None):
        ax.plot([x, x], [L, L + 0.2 * (La + Lb)], '-', c='black')
    ax.plot([x], [L], 'o', markerfacecolor=ncl, markeredgecolor='black')
    ax.text(x + 0.075 * s, L + 0.01 * L0, str(node.nid),
        horizontalalignment='center', verticalalignment='bottom',
        fontsize=7, fontstretch='condensed')
    return (xmin, xmax)

def make_dendro(cfg, M, fname, title=None):
    """ Create a tree visualization.
        Input:
            M      : HLDA object
            fname  : str, save file name
            title  : str, plot title, or None
            cfg    : TestConfig object
    """
    psz = cfg.plot.out.size
    plt.rcParams["mathtext.fontset"] = 'cm'
    fig, ax = plt.subplots()
    if ((title is not None) and cfg.plot.out.title):
        plt.title(title)
    ax.get_xaxis().set_ticks([])
    ax.set_ylabel('Loss')
    L = M.root.model.loss(M.par(M.root))
    N = np.sum(M.root.model.ntrain)
    xmin, xmax = _add_branch(ax, M, M.root, L, L, N, 0.0, 1.0, 1)
    xext = max(-xmin, xmax) * 1.2
    ax.set_xlim(-xext, xext)
    ax.set_ylim(-0.35 * L, 1.2 * L)
    ax.yaxis.set_major_formatter(remove_negative)
    plt.gcf().set_size_inches(psz[0],psz[1])
    plt.tight_layout()
    plt.savefig(fname, dpi=cfg.plot.out.dpi, format='pdf')
    plt.close(fig)

def barstack(ax, x, y, c='Blues', lc='k', lw=0.5, margin=0.05):
    """ Add a stacked bar plot to an axis.
        Input:
            ax     : Axes object
            x      : ndarray(K), bar center x coordinates
            y      : ndarray(K,N), bar heights per component
            c      : str, colormap name, or list of colors
            lc     : color, line and edge color
            lw     : float, line and edge width
            margin : float, relative spacing between bars
        Output:
            list of colors used
    """
    K, N = y.shape
    w = (x[1] - x[0]) * (1 - 2 * margin)
    b = np.zeros(K)
    if isinstance(c, list):
        cl = c
    else:
        cmap = plt.cm.get_cmap(c)
        cl = []
        r = np.linspace(1, N, num=N) / (N + 1)
        for j in range(N):
            cl.append(cmap(r[j]))
    for j in range(N):
        ax.bar(x, y[:,j], width=w, bottom=b, color=cl[j], zorder=1)
        b += y[:,j]
    for i in range(K):
        ax.add_patch(Rectangle((x[i] - 0.5*w, 0), w, np.sum(y[i,:]), fill=False,
            edgecolor=lc, lw=lw, zorder=2))
    ax.axhline(y=0, color=lc, linewidth=lw)
    return cl

def plot_pred(x, y, a, n, f, name, is_bay):
    if is_bay:
        a0 = np.sum(a, axis=1).reshape((-1,1))
        p = a / a0
    else:
        p = a
    K = p.shape[1]
    for k in range(K):
        sx = (y == k)
        fig, ax = plt.subplots()
        ax.scatter(x[sx,0], x[sx,1], c=p[sx,:].reshape((-1,3)))
        ax.set_aspect(1.0)
        plt.savefig(name + '-' + str(n) + '-' + str(f) + '-' + str(k) + '.pdf',
            format='pdf')
        plt.close(fig)

### Interface elements #########################################################

class SplitPlot(ifc.HookCall):
    """ Hook call for visualizing data splitting during model training.
        Input:
            packet : Packet object
            node   : BNode object
    """
    def __call__(self, packet, node):
        plot_split(packet, node)

class NodePlot(ifc.HookCall):
    """ Hook call for visualizing node models during model training.
        Input:
            packet : Packet object
            node   : BNode object
    """
    def __call__(self, packet, node):
        plot_node(packet, node)

### Test #######################################################################

class HLDATest(object):
    """ Wrapper class for test routines.
    """
    def __init__(self, name, data, logfile='', outdir=''):
        self.name = name
        self.data = data
        self.seed = 0
        self.logfile = logfile
        if (logfile == ''):
            self.flog = None
        else:
            self.flog = open(logfile, 'a')
        self.outdir = outdir
        self.verbose = True
        self.plot = PlotConfig()
        self.test1 = Test1Config()
        self.test2 = Test2Config()
        self.test3 = Test3Config()

    def log(self, msg):
        if (self.logfile == ''):
            if self.verbose:
                if isinstance(msg, (list, tuple)):
                    for item in msg:
                        if isinstance(item, str):
                            print(item)
                        else:
                            print(str(item))
                else:
                    if isinstance(msg, str):
                        print(msg)
                    else:
                        print(str(msg))
        else:
            if isinstance(msg, (list, tuple)):
                for item in msg:
                    if isinstance(item, str):
                        self.flog.write(item + '\n')
                    else:
                        self.flog.write(str(item) + '\n')
            else:
                if isinstance(msg, str):
                    self.flog.write(msg + '\n')
                else:
                    self.flog.write(str(msg) + '\n')

    def plot_data(self, image=None, data=None, complement=None, frame=None,
        bar=True, labels=None, legend=None, lims=None, name=None, split=False,
        v=None):
        """ Plot data elements.
            Input:
                image      : None, or ndarray(M,N), ndarray(M,N,1) or
                             ndarray(M,N,3), raster image
                data       : None, or tuple(2) of
                             x : ndarray(N,2), data
                             y : ndarray(N) of int, class indices
                complement : None, or tuple(2) of
                             x : ndarray(N,2), data
                             y : ndarray(N) of int, class indices
                frame      : None, or matplotlib.path.Path object
                bar        : bool, if True and image is not None, plot color
                             sidebar
                labels     : None, or tuple(3) of str, title, x and y labels
                legend     : None, or tuple of list of handles, str
                lims       : None, or tuple(4) of float, x and y axis limits
                name       : None, or str, save file name
                split      : bool, if True, plot partitioning line
                v          : None, or tuple(2) of float, color value limits
            Output:
                tuple(4), x and y axis limits
        """
        if (lims is None):
            if (data is not None):
                xlim = data_lims(data[0], margin=self.plot.margin)
            elif (complement is not None):
                xlim = data_lims(complement[0], margin=self.plot.margin)
            else:
                xlim = (0, 1, 0, 1)
        else:
            xlim = lims
        plt.rcParams["mathtext.fontset"] = 'cm'
        fig, ax = plt.subplots()
        ax.set_xlim((xlim[0], xlim[1]))
        ax.set_ylim((xlim[2], xlim[3]))
        if (labels is not None):
            if ((labels[0] is not None) and self.plot.out.title):
                plt.title(labels[0])
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[2])
        else:
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
        got_bar = False
        if (image is None):
            ax.set_facecolor(self.plot.bgcolor)
        else:
            if isinstance(v, tuple):
                vmin = v[0]
                vmax = v[1]
            else:
                vmin = np.amin(image)
                vmax = np.amax(image)
            img = plt.imshow(image, aspect='equal', cmap=self.plot.image.cmap,
                extent=xlim, vmin=vmin, vmax=vmax, zorder=0)
            if (bar and ((image.ndim == 2) or (image.shape[2] == 1))):
                fig.colorbar(img, ax=ax)
                got_bar = True
        if split:
            plt.axvline(x=0, color=self.plot.splitter.color,
                linestyle=self.plot.splitter.style,
                linewidth=self.plot.splitter.width, zorder=1)
        K = self.data.nclass
        ncl = np.zeros(K, dtype=np.int)
        if (data is not None):
            x, y = data
            for k in range(K):
                sx = (y == k)
                nx = np.count_nonzero(sx)
                if (nx > 0):
                    plt.scatter(x[sx,0], x[sx,1],
                        s=self.plot.markers.size[k],
                        c=self.plot.markers.facecolor[k],
                        marker=self.plot.markers.marker[k],
                        linewidths=self.plot.markers.edgewidth,
                        edgecolors=self.plot.markers.edgecolor[k],
                        label=self.data.names[k], zorder=3)
                    ncl[k] += nx
        if (complement is not None):
            x, y = complement
            for k in range(K):
                sx = (y == k)
                nx = np.count_nonzero(sx)
                if (nx > 0):
                    if (ncl[k] > 0):
                        mlb = None
                    else:
                        mlb = self.data.names[k]
                    plt.scatter(x[sx,0], x[sx,1],
                        s=self.plot.markers.size[k],
                        c=self.plot.markers.facecolor[k],
                        marker=self.plot.markers.marker[k],
                        alpha=self.plot.markers.alpha,
                        linewidths=0, label=mlb, zorder=2)
                    ncl[k] += nx
        if (frame is not None):
            ax.add_patch(mpatches.PathPatch(frame,
                facecolor=self.plot.frame.facecolor,
                edgecolor=self.plot.frame.edgecolor,
                alpha=self.plot.frame.alpha, zorder=4))
        if (np.sum(ncl) > 0):
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles),
                key=lambda t: t[0]))
            plt.legend(handles, labels, loc='best')
        elif (legend is not None):
            plt.legend(handles=legend[0], loc=legend[1])
        if (self.plot.out.save and (name is not None)):
            psz = self.plot.out.size
            if got_bar:
                plt.gcf().set_size_inches(6*psz[0]/5,psz[1])
            else:
                plt.gcf().set_size_inches(psz[0],psz[1])
            plt.tight_layout()
            plt.savefig(name, dpi=self.plot.out.dpi, format='pdf')
            plt.close(fig)
            self.log('Saved plot ' + name)
        else:
            plt.show()
        return xlim

    def run(self):
        warnings.simplefilter('error')
        if (self.logfile != ''):
            self.flog = open(self.logfile, 'a')
        start_time = time.process_time()
        self.log(['### HLDATest ' + self.name + ' @ ' +
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ''])
        K = self.data.nclass
        ny = np.zeros(K, dtype=np.int)
        for k in range(K):
            ny[k] = np.count_nonzero(self.data.y == k)
        self.log([self.name + ' dataset:',
                  '  Number of data points : ' + str(self.data.ndata),
                  '  Number of dimensions  : ' + str(self.data.ndim),
                  '  Number of classes     : ' + str(K) + ' ' + str(ny)])

        if self.test1.run:
            self.test_1()
        if self.test2.run:
            self.test_2()
        if self.test3.run:
            self.test_3()

        self.log(['', '### Completed test ' + self.name,
                  '### Elapsed time {:.2f}'.format(time.process_time() -
                  start_time) + ' seconds', ''])
        if (self.logfile != ''):
            self.flog.close()

    def test_1(self):
        np.random.seed(self.seed)
        self.log(['', '### Starting test: ' + self.test1.name])

        K = self.data.nclass
        D = self.data.ndim
        HMM = hlda(K, D)
        if self.test1.nodes:
            methods = [ifc.HookMethod(2, 'NODE_SPLIT', SplitPlot())]
            HMM.collect('splot', (self, self.test1.name, 'HLDA'))
        else:
            methods = []
        HMM.partial_fit(self.data.x, self.data.y, methods=methods)
        self.log('Model acquired in ' + str(HMM.root.nsplits) + ' splits.')

        if self.test1.dendro:
            fname = self.outdir + self.name + '-dendro.pdf'
            title = self.name + ' HLDA'
            make_dendro(self, HMM, fname, title=title)
            self.log('Dendrogram saved as ' + fname)

    def test_2(self):
        np.random.seed(self.seed)
        self.log(['', '### Starting test: ' + self.test2.name])

        K = self.data.nclass
        D = self.data.ndim
        HMM = hlda(K, D, sub=2)
        if self.test2.nodes:
            methods = [ifc.HookMethod(2, 'NODE_OUT', NodePlot())]
            HMM.collect('splot', (self, self.test2.name, 'HLDA-2D-auto'))
        else:
            methods = []
        HMM.partial_fit(self.data.x, self.data.y, methods=methods)
        self.log('Model acquired in ' + str(HMM.root.nsplits) + ' splits.')

    def test_3(self):
        np.random.seed(self.seed)
        self.log(['', '### Starting test: ' + self.test3.name])

        K = self.data.nclass
        D = self.data.ndim
        m = np.zeros((K,D))
        for k in range(K):
            m[k,:] = np.mean(self.data.x[self.data.y == k,:].reshape((-1,D)),
            axis=0)
        P = np.zeros((D,2))
        for i in range(2):
            P[:,i] = m[i+1,:] - m[i,:]
            for j in range(i):
                P[:,i] -= np.sum(P[:,i] * P[:,j]) * P[:,j]
            P[:,i] /= np.sum(P[:,i]**2)**0.5
        HMM = hlda(K, D, sub=P)
        HMM.partial_fit(self.data.x, self.data.y, methods=[])
        self.log('Model acquired in ' + str(HMM.root.nsplits) + ' splits.')

        x = np.matmul(self.data.x, P)
        lim = square_lims(data_lims(x), margin=self.plot.margin)
        isz = self.plot.image.size
        xy = np.zeros((isz[0], isz[1], 2))
        xy[:,:,0] += np.linspace(lim[0], lim[1], isz[0]).reshape((1,isz[0]))
        xy[:,:,1] += np.linspace(lim[3], lim[2], isz[1]).reshape((isz[1],1))
        Q = np.matmul(xy.reshape((isz[0]*isz[1],2)), P.T)
        a = HMM.predict_para(Q)
        p = a / np.sum(a, axis=1).reshape((-1,1))
        U = dirichlet_uncertainty(a)
        xlab = '$\hat{c}_{0}$'
        ylab = '$\hat{c}_{1}$'
        pot = self.plot.out.title
        self.plot.out.title = True  # always show title for these plots

        nix = np.zeros((isz[0]*isz[1], 3))
        leg = []
        for ix, tk in enumerate(HMM._collect['track']):
            node, idx = tk
            nix[idx,:] = plt.cm.Pastel1(ix)[0:3]
            leg.append(Line2D([0], [0], linewidth=0, marker='o',
                label='Node ' + str(node.nid), markeredgecolor='black',
                markerfacecolor=plt.cm.Pastel1(ix)[0:3], markersize=10))
        self.plot_data(image=nix.reshape((isz[0],isz[1],3)),
            labels=('Model', xlab, ylab), legend=(leg, 'best'), lims=lim,
            name=self.outdir + 'HLDA-2D-compound-src.pdf', v=(0,1))

        self.plot_data(image=p.reshape((isz[0],isz[1],3)),
            labels=('$P_{\omega}$', xlab, ylab), lims=lim,
            name=self.outdir + 'HLDA-2D-compound-p.pdf', v=(0,1))

        vla = (min(np.amin(U[:,0]), self.test2.valea[0]),
               max(np.amax(U[:,0]), self.test2.valea[1]))
        self.plot_data(image=U[:,0].reshape((isz[0],isz[1])),
            labels=('$U_{\mathrm{al}}$', xlab, ylab), lims=lim,
            name=self.outdir + 'HLDA-2D-compound-alea.pdf', v=vla)

        vle = (min(np.amin(U[:,1]), self.test2.vepis[0]),
               max(np.amax(U[:,1]), self.test2.vepis[1]))
        self.plot_data(image=U[:,1].reshape((isz[0],isz[1])),
            labels=('$U_{\mathrm{ep}}$', xlab, ylab), lims=lim,
            name=self.outdir + 'HLDA-2D-compound-epis.pdf', v=vle)

        self.plot.out.title = pot

class PerformanceTest(object):
    """ Wrapper class for performance test routines.
    """
    def __init__(self, name):
        self.name = name
        self.outdir = 'tests/performance/'
        self.data = PerfDataConfig()
        self.plot = PerfPlotConfig()
        self.n_fold = 5
        self.r_train = [ 1, 2, 4, 8, 16, 32, 64, 128, 256 ]

        self.hlda_sub = self.data.n_dim_inform

        self.method = [
            Classifier(CustomGNB(), 'NB', 'Naive Bayes',
                'tab:gray', (0, (3, 1, 1, 1)), 's', dist=True),
            Classifier(None, 'BP', 'Bayesian Partitioning',
                'tab:red', (0, (3, 1, 1, 1, 1, 1)), 'o', dist=True),
            Classifier(RandomForestClassifier(), 'RF', 'Random Forest',
                'tab:olive', '--', '^'),
            Classifier(LogisticRegressionCV(), 'LR', 'Logistic Regression',
                'tab:cyan', '-', 'h')
            ]

    def run(self):
        D = self.data.n_dim_inform + self.data.n_dim_random
        K = self.data.n_classes

        # add the HLDA classifier to the set (cannot add it before this point
        # because the configuration can change)

        # NB: force matrix solver (LOBPCG not optimized for performance)

        hpars = LDAPar(self.data.n_classes, D, eigalg=EigType.MATRIX,
            sub=self.hlda_sub)
        self.method[0].algo = hlda(hpars)

        # create the dataset

        self.X, self.Y = make_class_data(self.data.n_data * self.n_fold,
            n_classes=K,
            n_dim_inform=self.data.n_dim_inform,
            n_dim_random=self.data.n_dim_random,
            random_state=self.data.random_state,
            blobgen=BlobGen(self.data.blob.n, self.data.blob.r),
            tractorgen=TractorGen(self.data.n_dim_inform,
                alim=self.data.tractor.a,
                cmax=self.data.tractor.c,
                rmax=self.data.tractor.r,
                slim=self.data.tractor.s,
                vlim=self.data.tractor.v),
            n_tractors=self.data.tractor.n,
            distance=self.data.distance)

        if self.data.oom_data:
            r2 = np.sum(self.X**2, axis=1)
            irx = np.argmax(r2)
            rx = r2[irx]**0.5
            N = self.data.n_data
            self.Xoom = np.random.randn(N*D).reshape((N,D))
            self.Xoom[:,0:self.data.n_dim_inform] = make_shell(self.data.n_data,
                self.data.n_dim_inform, r0=self.data.oom_extend * rx)
        else:
            self.Xoom = None

        if self.plot.data:
            Nx = self.plot.nmax
            cl = [ 'tab:red', 'tab:green', 'tab:blue',
                   'tab:orange', 'tab:purple', 'tab:brown',
                   'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan' ]
            plt.rcParams["mathtext.fontset"] = 'cm'
            fig, ax = plt.subplots()
            for k in range(self.data.n_classes):
                sx = (self.Y == k)
                nx = np.count_nonzero(sx)
                if (nx > Nx):
                    ax.plot(self.X[sx,0][0:Nx], self.X[sx,1][0:Nx], '.',
                        c=cl[k % 10], markersize=self.plot.markersize,
                        markeredgewidth=self.plot.mew)
                else:
                    ax.plot(self.X[sx,0], self.X[sx,1], '.', c=cl[k % 10],
                        markersize=self.plot.markersize,
                        markeredgewidth=self.plot.mew)
            if (self.Xoom is not None):
                Mx = self.n_fold * Nx
                if (self.Xoom.shape[0] > Mx):
                    ax.plot(self.Xoom[0:Mx,0], self.Xoom[0:Mx,1], '.',
                        c='magenta', markersize=self.plot.markersize,
                        markeredgewidth=self.plot.mew)
                else:
                    ax.plot(self.Xoom[:,0], self.Xoom[:,1], '.', c='magenta',
                        markersize=self.plot.markersize,
                        markeredgewidth=self.plot.mew)
            ax.set_aspect(1.0)
            ax.set_xlabel('$x_0$')
            ax.set_ylabel('$x_1$')
            if self.plot.save:
                psz = self.plot.size
                plt.gcf().set_size_inches(psz[0],psz[1])
                plt.savefig(self.outdir + self.name + '-data.pdf',
                    dpi=self.plot.dpi, format='pdf')
                plt.close(fig)
            else:
                plt.show()

        if self.data.rotate:
            np.random.seed(self.data.random_state)
            self.R = rrot(D)
            self.X = np.matmul(self.X, self.R)
            self.Xoom = np.matmul(self.Xoom, self.R)
        else:
            self.R = None

        # run the performance tests

        ACC = []
        F1 = []
        AUC = []
        CAS = []
        MCE = []
        NDS = []
        Ntr = []
        TTR = []
        TTS = []
        for k, Nr in enumerate(self.r_train):
            Ntr.append(self.data.n_classes * self.data.n_data // Nr)
            acc = []
            f1 = []
            roc = []
            auc = []
            cas = []
            mce = []
            nds = []
            ttr = []
            tts = []

            start_time = time.process_time()
            y, py, pom, tixx = kfoldtest(self.data.n_classes, self.X, self.Y,
                self.Xoom, self.method, K=self.n_fold, frac=Nr)
            for i, M in enumerate(self.method):
                if (pom is None):
                    pomi = None
                else:
                    pomi = pom[:,i]
                ret = skill_metrics(y, py[:,K*i:K*(i+1)], pomi, M.is_dist)
                acc.append(ret['ACC'])
                f1.append(ret['F1'])
                roc.append(ret['ROC'])
                auc.append(ret['AUC'])
                mcas = ret['CAS'][1]
                if isinstance(mcas, tuple):
                    cas.append(mcas[1])
                    mce.append(mcas[0])
                else:
                    cas.append(mcas)
                nds.append(ret['NDS'])
                if self.plot.AS:
                    self.plot_AS(ret, Ntr[k], i)
                ttr.append(tixx[1][i] / tixx[0][i])
                tts.append(tixx[3][i] / tixx[2][i])

            if self.plot.ROC:
                fig, ax = plt.subplots()
                ax.plot([0,1], [0,1], ':', c='k', lw=0.75)
                for i, M in enumerate(self.method):
                    FPR, TPR, TRH = roc[i]
                    ax.plot(FPR, TPR, color=M.c, linestyle=M.ls,
                        label=M.cid + ' AUC = {:.3f}'.format(auc[i]))
                ax.set_aspect(1.0)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                ax.legend(loc='lower right')
#                plt.tight_layout()
                if self.plot.save:
                    psz = self.plot.size
                    plt.gcf().set_size_inches(psz[0],psz[1])
                    plt.savefig(self.outdir + self.name + '-ROC-' + str(Ntr[k])
                        + '.pdf', dpi=self.plot.dpi, format='pdf')
                    plt.close(fig)
                else:
                    plt.show()

            ACC.append(acc)
            F1.append(f1)
            AUC.append(auc)
            CAS.append(cas)
            if (len(mce) > 0):
                MCE.append(mce)
            NDS.append(nds)
            TTR.append(ttr)
            TTS.append(tts)

        if self.plot.scores:
            N = np.array(Ntr)
            self.score_plot(N, np.array(ACC), 'ACC', 'Accuracy')
            self.score_plot(N, np.array(F1), 'F1', 'F1 Score')
            self.score_plot(N, np.array(AUC), 'AUC', 'ROC-AUC')
            if (len(MCE) == 0):
                self.score_plot(N, np.array(CAS), 'CAS', 'Calibration Score')
            else:
                self.score_plot(N, np.array(MCE), 'MCE', 'MCE')
                self.score_plot(N, np.array(CAS), 'MACE', 'MACE')
            self.score_plot(N, np.array(NDS), 'NDS', 'NDS')
            self.score_plot(N, np.array(TTR), 'TTR', 'Time (s)', logy=True)
            self.score_plot(N, np.array(TTS), 'TTS', 'Time (s)', logy=True)

    def plot_AS(self, ret, N, ix):
        K = self.data.n_classes
        CAS = ret['CAS']
        ppn = ret['PPN']
        pcf, ncf = ret['CFS']
        n_prange = CAS[2].size
        mw = 0.5 / n_prange
        ppx = np.linspace(mw, 1 - mw, num=n_prange)
        ppn /= np.sum(ppn)
        ppc = np.linspace(0, 1, num=pcf.size)
        npc = np.linspace(0, 1, num=ncf.size)
        fig, ax = plt.subplots(figsize=(6,7))
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_aspect(1.0)
        divider = make_axes_locatable(ax)
        hax = divider.append_axes('bottom', 1.5, pad=0.1, sharex=ax)
        cls = barstack(hax, ppx, ppn)
        hax.set_xlabel('Predicted Probability')
        hax.set_ylabel('Frequency')
        ax.plot([0,1], [0,1], '--', c='gray', lw=0.75)
        ax.plot([1/K,1/K], [0,1], '--', c='gray', lw=0.75)
        pp = CAS[3]
        po = CAS[4]
        if isinstance(CAS[1], tuple):
            ax.plot(pp, po, 'o-', c='blue', label='MACE = {:.3f}'.format(CAS[1][1]))
        else:
            ax.plot(pp, po, 'o-', c='blue', label='CAS = {:.3f}'.format(CAS[1]))
        ax.plot(pcf, ppc, '-', c='green', linewidth=2)
        ax.plot(ncf, npc, '-', c='red', label='NDS = {:.3f}'.format(ret['NDS']))
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_title(self.method[ix].name)
        ax.set_ylabel('Observed Probability')
        ax.legend(loc='lower right')
        plt.tight_layout()
        if self.plot.save:
            psz = self.plot.size
            plt.gcf().set_size_inches(psz[0],7*psz[0]/6)
            plt.savefig(self.outdir + self.name + '-CAS-NDS-' +
                self.method[ix].cid + '-' + str(N) + '.pdf',
                dpi=self.plot.dpi, format='pdf')
            plt.close(fig)
        else:
            plt.show()

    def score_plot(self, N, S, ID, name, logy=False):
        Nn = N.size
        plt.rcParams["mathtext.fontset"] = 'cm'
        fig, ax = plt.subplots()
        for i, M in enumerate(self.method):
            ax.plot(N, S[:,i], M.mk + '-', c=M.c, label=M.cid)
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('Training set size')
        ax.set_ylabel(name)
        ax.legend(loc='best')
#        plt.tight_layout()
        if self.plot.save:
            psz = self.plot.size
            plt.gcf().set_size_inches(psz[0],psz[1])
            plt.savefig(self.outdir + self.name + '-Score-' + ID + '.pdf',
                dpi=self.plot.dpi, format='pdf')
            plt.close(fig)
        else:
            plt.show()

# EOF testutils.py __________________________________________________________________

"""
FILE:                   lda.py
COPYRIGHT:              (c) 2024 Finnish Meteorological Institute
                        P.O. BOX 503
                        FI-00101 Helsinki, Finland
                        https://www.fmi.fi/
LICENCE:                MIT
AUTHOR:                 Terhi Mäkinen (terhi.makinen@fmi.fi)
DESCRIPTION:

This module provides a hierarchical linear discriminant analyser (HLDA).
"""

import numpy as np
from . import algorithms as alg
from . import models
from . import interfaces as ifc
from . import trees

class LDASplitPacket(ifc.HookCall):
    """Hook call for splitting a packet with LDA data. The partitioned subsets
    will be added as an attribute to the original packet.

    Parameters
    ----------
    packet : Packet object
        packet
    node : N/A
        retained for compatibility
    """
    def __call__(self, packet, node):
        assert isinstance(packet, ifc.Packet)

        hdr = packet.header
        X = packet.get_attr('_train_data')
        if (X is None):
            Xa = Xb = None
        else:
            n = packet.get_attr('_model_normal')
            p = packet.get_attr('_model_pivot')
            w = packet.get_attr('_weights')
            q = w[0].copy()
            ws = w[0] + w[1]
            wx = ws > 0
            if (np.count_nonzero(wx) > 0):
                q[wx] /= ws[wx]
            if (X.n > 0):
                x = np.matmul(X.x - p, n).flatten()
            else:
                x = np.zeros(0)
            s = x < 0
            ns = np.count_nonzero(s)
            if (ns > 0):
                Xa = X.subset(s, q=q)
            else:
                Xa = None
            if (X.n - ns > 0):
                Xb = X.subset(~s, q=1-q)
            else:
                Xb = None
        C = packet.get_attr('_complement')
        if (C is None):
            Ca = Cb = None
        else:
            zc = hdr.par(node).cutoff
            prj = packet.get_attr('_projection')
            yc = C.yxt
            K = yc.size - 1
            if (X is None):
                n = packet.get_attr('_model_normal')
                p = packet.get_attr('_model_pivot')
                x = np.zeros(0)
                yx = np.zeros(K+1, dtype=np.int)
            else:
                yx = X.yxt
            if (C.n > 0):
                c = np.matmul(C.x - p, n).flatten()
            else:
                c = np.zeros(0)
            if (x.size + c.size > 0):
                Ca = C.subset(np.zeros(0, dtype=np.int))
                Cb = C.subset(np.zeros(0, dtype=np.int))
                for k in range(K):
                    if ((yx[k+1] - yx[k] > 0) or (yc[k+1] - yc[k] > 0)):
                        (ixa, ica), (ixb, icb) = alg.complement_split(
                            x[yx[k]:yx[k+1]], c[yc[k]:yc[k+1]],
                            G=prj.gmm[k], zc=zc)
                        if (ixa.size > 0):
                            Ca.add(X.subset(ixa + yx[k]))
                        if (ica.size > 0):
                            Ca.add(C.subset(ica + yc[k]))
                        if (ixb.size > 0):
                            Cb.add(X.subset(ixb + yx[k]))
                        if (icb.size > 0):
                            Cb.add(C.subset(icb + yc[k]))
        if ((Xa is not None) or (Xb is not None) or
            (Ca is not None) or (Cb is not None)):
            AttrA = {}
            if (Xa is not None):
                AttrA['_train_data'] = Xa
            if (Ca is not None):
                AttrA['_complement'] = Ca
            if (len(AttrA) > 0):
                pa = ifc.Packet(hdr.get_pid(), hdr, attrs=AttrA, methods=[])
                pa.clone(packet)
            else:
                pa = None
            AttrB = {}
            if (Xb is not None):
                AttrB['_train_data'] = Xb
            if (Cb is not None):
                AttrB['_complement'] = Cb
            if (len(AttrB) > 0):
                pb = ifc.Packet(hdr.get_pid(), hdr, attrs=AttrB, methods=[])
                pb.clone(packet)
            else:
                pb = None
            packet.set_attr('_subpackets', (pa, pb))
        if (packet.is_set('_eval_data')):
            X = packet.get_attr('_eval_data')
            n = packet.get_attr('_model_normal')
            p = packet.get_attr('_model_pivot')
            s = np.matmul(X.x - p, n).flatten() < 0
            ns = np.count_nonzero(s)
            if (ns > 0):
                pa = ifc.Packet(hdr.get_pid(), hdr,
                    attrs={'_eval_data': X.subset(s)}, methods=[])
                pa.clone(packet)
            else:
                pa = None
            if (X.n - ns > 0):
                pb = ifc.Packet(hdr.get_pid(), hdr,
                    attrs={'_eval_data': X.subset(~s)}, methods=[])
                pb.clone(packet)
            else:
                pb = None
            packet.set_attr('_subpackets', (pa, pb))

class LDATrackData(ifc.HookCall):
    """Hook call for tracking LDA data.

    Parameters
    ----------
    packet : Packet object
        packet
    node : BNode object
        node
    """
    def __call__(self, packet, node):
        assert isinstance(packet, ifc.Packet)
        assert isinstance(node, trees.BNode)

        hdr = packet.header
        if (packet.is_set('_train_data')):
            hdr.collect('track', (node, packet.get_attr('_train_data').idx))
        elif (packet.is_set('_eval_data')):
            hdr.collect('track', (node, packet.get_attr('_eval_data').idx))

class PriorWeight(object):
    """Wrapper for prior weights.

    Parameters
    ----------
    cfg : dict or None
        priors, recognized keys are Beta, Dirichlet, NIX, and IWishart
    """
    def __init__(self, cfg=None):
        assert isinstance(cfg, dict) or (cfg is None)

        self.Beta = 1.0
        self.Dirichlet = 1.0
        self.NIX = (1.0, 1.0)
        self.IWishart = 1.0
        if (cfg is not None):
            if ('Beta' in cfg) and isinstance(cfg['Beta'], (int, float)):
                self.Beta = cfg['Beta']
            if ('Dirichlet' in cfg) and isinstance(cfg['Dirichlet'], (int, float)):
                self.Dirichlet = cfg['Dirichlet']
            if ('NIX' in cfg):
                if isinstance(cfg['NIX'], (int, float)):
                    self.NIX = (cfg['NIX'], cfg['NIX'])
                elif isinstance(cfg['NIX'], list):
                    self.NIX = (cfg['NIX'][0], cfg['NIX'][1])
                elif isinstance(cfg['NIX'], tuple):
                    self.NIX = cfg['NIX']
            if ('IWishart' in cfg) and isinstance(cfg['IWishart'], (int, float)):
                self.IWishart = cfg['IWishart']

class LDAPar(ifc.IDataPar):
    """LDA parameters.

    Parameters
    ----------
    nclass : int
        number of classes
    ndim : int
        number of data dimensions
    eigalg : EigType
        algorithm type for solving the generalized eigenvalue problem
    prior : dict or None
        priors, recognized keys are Beta, Dirichlet, NIX, and IWishart
    sub : ndarray(ndim,nsub)
        orthonormal base vector set of the density subspace, or
        : int
        subspace dimensionality of per-node models
    subtype : str
        density subspace type
    ngmm : int
        maximal number of Gaussian mixture model components
    cutoff : float
        complement cutoff in standard deviations
    ftol : float
        floating point tolerance
    niter : int
        maximal number of iterations
    """
    def __init__(self, nclass,
                       ndim,
                       eigalg = models.EigType.AUTO,
                       prior = None,
                       sub = 0,
                       subtype = 'auto',
                       ngmm = 8,
                       cutoff = 6.0,
                       ftol = 1e-8,
                       niter = 100
                ):
        assert isinstance(nclass, int) and (nclass > 1)
        assert isinstance(ndim, int) and (ndim > 1)
        assert isinstance(eigalg, models.EigType)
        assert isinstance(prior, dict) or (prior is None)
        assert ((isinstance(sub, int) and (sub >= 0)) or
                (isinstance(sub, np.ndarray) and
                 (sub.ndim == 2) and
                 (sub.shape[0] == ndim) and
                 (sub.shape[1] <= ndim) and
                 np.allclose(np.matmul(sub.T, sub), np.eye(sub.shape[1]))
                )
               )
        assert (subtype in ('auto', 'none', 'fixed', 'leaf', 'splitter'))
        assert isinstance(ngmm, int) and (ngmm >= 0)
        assert isinstance(cutoff, (int,float)) and (ftol > 0)
        assert isinstance(ftol, float) and (ftol > 0)
        assert isinstance(niter, int) and (niter > 0)

        self._nclass = nclass
        self._ndim = ndim
        self._eigalg = eigalg
        self._prior = PriorWeight(prior)
        self._sub = sub
        if (subtype == 'auto'):
            if isinstance(sub, np.ndarray):
                self._subtype = 'fixed'
            elif (sub == 0):
                self._subtype = 'none'
            else:
                self._subtype = 'leaf'
        else:
            self._subtype = subtype
        if (isinstance(sub, int) and (sub == 0)):
            self._ngmm = 0
        else:
            self._ngmm = max(1,ngmm)
        self._cutoff = cutoff
        self._ftol = ftol
        self._niter = niter

    @property
    def cutoff(self):
        """Return complement cutoff distance.

        Returns
        -------
        float
        """
        return self._cutoff

    @property
    def eigalg(self):
        """Return eigenvalue algorithm type.

        Returns
        -------
        EigType
        """
        return self._eigalg

    @property
    def ngmm(self):
        """Return number of Gaussian mixture model components.

        Returns
        -------
        int
        """
        return self._ngmm

    @property
    def ftol(self):
        """Return floating point tolerance.

        Returns
        -------
        float
        """
        return self._ftol

    @property
    def nclass(self):
        """Return number of classes.

        Returns
        -------
        int
        """
        return self._nclass

    @property
    def ndim(self):
        """Return number of dimensions.

        Returns
        -------
        int
        """
        return self._ndim

    @property
    def niter(self):
        """Return default maximal number of iterations.

        Returns
        -------
        int
        """
        return self._niter

    @property
    def nsub(self):
        """Return the dimensionality of the density subspace.

        Returns
        -------
        int
        """
        if isinstance(self._sub, int):
            return self._sub
        return self._sub.shape[1]

    @property
    def prior(self):
        """Return a reference to prior weights.

        Returns
        -------
        PriorWeight object
        """
        return self._prior

    @property
    def sub(self):
        """Return the projection matrix to the density subspace, or None if the
        projection is node dependent.

        Returns
        -------
        ndarray(D,M) or None
        """
        if isinstance(self._sub, int):
            return None
        return self._sub

    @property
    def subtype(self):
        """Return the density subspace type.

        Returns
        -------
        str
        """
        return self._subtype

# LDAModel declared hooks     POINT OF CALL:

ifc.hook.set('MODEL_IN')    # on entering LDAModel.send()
ifc.hook.set('MODEL_OUT')   # on leaving LDAModel.send()

class LDAModel(ifc.IDataModel):
    """Data model wrapper for Linear Discriminant Analysis.

    Parameters
    ----------
    par : LDAPar object
        model parameters
    """
    def __init__(self, par):
        assert isinstance(par, LDAPar)

        self._lda = models.LDA(par.nclass, par.ndim, solver=par.eigalg,
            ngmm=par.ngmm, sub=(par.nsub, par.sub))

    def clear(self):
        """Reset the model.
        """
        self._lda.clear()

    def is_splittable(self, par):
        """Return True if the model allows splitting.

        Returns
        -------
        bool
        """
        assert isinstance(par, LDAPar)

        return (self._lda.pivot is not None)

    @property
    def lda(self):
        """Return a reference to the LDA model.

        Returns
        -------
        LDAModel object
        """
        return self._lda

    def loss(self, par, c=None, w=None):
        """Return the internal loss function value of the model, or if either
        c, w, or both are provided, return the partition loss.

        Parameters
        ----------
        par : LDAPar object
            model parameters
        c : tuple(2) of tuple(2) of (BNode, LDAPar) or None
            child nodes and their parameters
        w : tuple(2) of ndarray(K) or None
            additional weights

        Returns
        -------
        float
            loss
        """
        assert isinstance(par, LDAPar)
        assert ((c is None) or
                (isinstance(c, (list, tuple)) and
                 (len(c) == 2) and
                 isinstance(c[0], (list, tuple)) and
                 (len(c[0]) == 2) and
                 isinstance(c[0][0], trees.BNode) and
                 isinstance(c[0][1], LDAPar) and
                 isinstance(c[1], (list, tuple)) and
                 (len(c[1]) == 2) and
                 isinstance(c[1][0], trees.BNode) and
                 isinstance(c[1][1], LDAPar)
                )
               )
        assert ((w is None) or
                isinstance(w, (list, tuple)) and
                (len(w) == 2) and
                isinstance(w[0], (int, float, np.ndarray)) and
                isinstance(w[1], (int, float, np.ndarray))
               )

        if ((c is None) and (w is None)):
            return alg.dirichlet_loss(self._lda.split_solver.data.cw,
                prior=par.prior.Dirichlet)
        K = par.nclass
        r = np.zeros(K)
        s = np.zeros(K)
        if (c is not None):
            r += c[0][0].model._lda.split_solver.data.cw
            s += c[1][0].model._lda.split_solver.data.cw
        if (w is not None):
            r += w[0]
            s += w[1]
        s += r
        sx = (s > 0)
        nx = np.count_nonzero(sx)
        if (nx > 0):
            r[sx] /= s[sx]
        if (nx < K):
            r[~sx] = 0.5
        return alg.dirichlet_partition_loss(self._lda.split_solver.data.cw, r,
            prior=par.prior.Dirichlet)

    @property
    def ntrain(self):
        """Return the number of data points per class used to train the model.

        Returns
        -------
        ndarray of int
        """
        return self._lda.split_solver.data.cn

    def predict_para(self, x, par, density=True):
        """Determine the Dirichlet parameter vector at the given points.

        Parameters
        ----------
        x : ndarray(N,D)
            evaluation points
        par : LDAPar object
            model parameters
        density : bool
            if set and density model is present, use it

        Returns
        -------
        ndarray(N,K)
            Dirichlet parameter vector a(x)
        """
        assert isinstance(x, np.ndarray) and (x.ndim == 2)
        assert isinstance(par, LDAPar)
        assert isinstance(density, bool)

        return (self._lda.predict_weight(x, R=par.sub, density=density) +
            par.prior.Dirichlet)

    def send(self, packet, node):
        """Send a packet for processing. Intended to be called by a BNode
        instance.

        Parameters
        ----------
        packet : Packet object
            packet
        node : BNode object
            node

        Attributes
        ----------
        _complement : MultiStat object
            data complement
        _eval_data : XData object
            evaluated data
        _send_count : int
            counter for calls to model.send()
        _train_data : XYData object
            training data
        _model_pending : bool
            True if packet should be sent again
        _model_normal : ndarray(D,1)
            partitioning hyperspace normal
        _model_pivot : ndarray(1,D)
            partitioning hyperspace pivot
        _projection : ProjData object
            auxiliary projection related data
        _weights : tuple(2) of ndarray(K)
            tentative split weights
        """
        assert isinstance(packet, ifc.Packet)
        assert isinstance(node, trees.BNode)

        packet.call('MODEL_IN', node)
        par = packet.header.par(node)
        send_count = packet.get_attr('_send_count')
        if (send_count == 1):

            ### training data

            if (packet.is_set('_train_data')):

                # add data and complement to the LDA model

                X = packet.get_attr('_train_data')
                C = packet.get_attr('_complement')
                q = self._lda.insert(X, C=C, R=par.sub, subtype=par.subtype,
                    prior=(par.prior.Beta, par.prior.NIX, par.prior.IWishart),
                    niter=par.niter, ftol=par.ftol #, # <<< remove ,
                    )
                if (q is None):
                    packet.set_attr('_model_pivot', None)
                else:
                    packet.set_attr('_weights', (q * X.w, (1 - q) * X.w))
                    packet.set_attr('_model_normal', self._lda.normal)
                    packet.set_attr('_model_pivot', self._lda.pivot)
                    if (par.nsub > 0):
                        packet.set_attr('_projection', self._lda.proj)

            ### evaluated data

            elif (packet.is_set('_eval_data')):

                # if the model allows splitting, add related information to the
                # packet

                if (self._lda.pivot is None):
                    packet.set_attr('_model_pivot', None)
                else:
                    packet.set_attr('_model_normal', self._lda.normal)
                    packet.set_attr('_model_pivot', self._lda.pivot)

            ### other process; not implemented

            else:
                pass

        packet.set_attr('_model_pending', False)
        packet.call('MODEL_OUT', node)

class HLDA(ifc.IDataAnalyser):
    """User interface for Hierarchical Linear Discriminant Analysis.

    Parameters
    ----------
    *args : optional arguments identified by type
        LDAModel object
            model
        LDAPar object
            model parameters
        BNodePar object
            tree parameters
    **kwargs : optional arguments identified by keyword
        arguments used to construct a LDAPar object if needed and not provided
    """
    def __init__(self, *args, **kwargs):
        self._mgen = LDAModel
        self._mpar = None
        self._tpar = None
        for arg in args:
            if (isinstance(arg, type) and issubclass(arg, LDAModel)):
                self._mgen = arg
            elif isinstance(arg, LDAPar):
                self._mpar = arg
            elif isinstance(arg, trees.BNodePar):
                self._tpar = arg
        if (self._mpar is None):
            self._mpar = LDAPar(
                *[arg for arg in args if (
                    (not isinstance(arg, type)) and
                    (not isinstance(arg, trees.BNodePar)))],
                **kwargs)
        if (self._tpar is None):
            self._tpar = trees.BNodePar()
        self._pidgen = -1
        self._collect = {}
        self._root = trees.BRoot(self.get_model(), self._tpar, self)

    def clear(self):
        """Reset the model to the initial state.
        """
        self._pidgen = -1
        self._collect = {}        
        self._root.clear()

    def clear_collection(self, key):
        """Initialize a collection.

        Parameters
        ----------
        key : immutable
            collection identifier
        """
        assert ifc.is_immutable(key)

        self._collect[key] = []

    def collect(self, key, value):
        """Add a value to a collection. If the collection does not exist, start
        a new collection.

        Parameters
        ----------
        key : immutable
            collection identifier
        value : untyped
            item added to collection
        """
        assert ifc.is_immutable(key)

        if (key in self._collect):
            self._collect[key].append(value)
        else:
            self._collect[key] = [value]

    def delete_collection(self, key):
        """Delete a collection.

        Parameters
        ----------
        key : immutable
            collection identifier
        """
        assert ifc.is_immutable(key)

        if (key in self._collect):
            del self._collect[key]

    def eval(self, X, attrs={}, methods=[]):
        """Evaluate data against the model.

        Parameters
        ----------
        X : ndarray(N,D)
            N data points of dimension D
        attrs : dict
            user-defined attributes
        methods : list of HookMethod objects
            user-defined methods

        Returns
        -------
        tuple
            tuple
                BNode
                    node
                ndarray of int
                    data tracked to node
        """
        assert (isinstance(X, np.ndarray) and
                (X.ndim == 2) and
                (X.shape[1] == self._mpar.ndim)
               )
        assert isinstance(attrs, dict)
        assert isinstance(methods, list)
        assert all([isinstance(item, ifc.HookMethod) for item in methods])

        self._eval(X, attrs, methods)
        return (*self._collect['track'], )

    def fit(self, X, Y, attrs={}, methods=[]):
        """Train model with the dataset. Use partial_fit() instead if the data
        is inserted in consecutive chunks.

        Parameters
        ----------
        X : ndarray(N,D)
            N data points of dimension D
        Y : ndarray(N) of int
            class index
        attrs : dict
            user-defined attributes
        methods : list of HookMethod objects
            user-defined methods

        Returns
        -------
        tuple
            tuple
                BNode
                    node
                ndarray of int
                    data tracked to node
        """
        assert (isinstance(X, np.ndarray) and
                (X.ndim == 2) and
                (X.shape[1] == self._mpar.ndim)
               )
        assert (isinstance(Y, np.ndarray) and
                (Y.ndim == 1) and
                (Y.size == X.shape[0])
               )
        assert isinstance(attrs, dict)
        assert isinstance(methods, list)
        assert all([isinstance(item, ifc.HookMethod) for item in methods])

        self.clear()
        A = {'model_send': True,
             'split_packet': True,
             'change_tree': True,
             '_train_data': models.XYData(X, Y)}
        if (self._mpar.nsub > 0):
            A['_complement'] = models.XYData(X, Y, idx=False)
        for k, v in attrs.items():
            if (k not in A):
                A[k] = v
        M = [
            ifc.HookMethod(0, 'NODE_SPLIT', LDASplitPacket()),
            ifc.HookMethod(1, 'NODE_TERMINATE', LDATrackData())]
        mid = 2
        for item in methods:
            item.set_mid(mid)
            mid += 1
            M.append(item)
        self.clear_collection('track')
        packet = ifc.Packet(self.get_pid(), self, attrs=A, methods=M)
        self._root.send(packet)
        return (*self._collect['track'], )

    def get_collection(self, key):
        """Return a collection, or None if the collection does not exist.

        Parameters
        ----------
        key : immutable
            collection identifier
        """
        assert ifc.is_immutable(key)

        if (key in self._collect):
            return self._collect[key]
        return None

    def get_model(self):
        """Return a new LDA model.

        Returns
        -------
        LDAModel object
        """
        return self._mgen(self._mpar)

    def get_pid(self):
        """Return a new unique packet identifier.

        Returns
        -------
        int
        """
        self._pidgen += 1
        return self._pidgen

    @property
    def ntrain(self):
        """Return the number of data points per class used to train the model.

        Returns
        -------
        int
        """
        return self._root.model.ntrain

    def par(self, node=None):
        """Return the correct LDAPar object for the LDAModel assigned to the
        given node if any.

        Parameters
        ----------
        node : BNode object or None
            controlling node
        """
        assert (node is None) or isinstance(node, trees.BNode)

        return self._mpar

    def partial_fit(self, X, Y, attrs={}, methods=[]):
        """Add data to model. Use fit() instead if the entire training dataset
        is inserted in one chunk.

        Parameters
        ----------
        X : ndarray(N,D)
            N data points of dimension D
        Y : ndarray(N) of int
            class index
        attrs : dict
            user-defined attributes
        methods : list of HookMethod objects
            user-defined methods

        Returns
        -------
        tuple
            tuple
                BNode
                    node
                ndarray of int
                    data tracked to node
        """
        assert (isinstance(X, np.ndarray) and
                (X.ndim == 2) and
                (X.shape[1] == self._mpar.ndim)
               )
        assert (isinstance(Y, np.ndarray) and
                (Y.ndim == 1) and
                (Y.size == X.shape[0])
               )
        assert isinstance(attrs, dict)
        assert isinstance(methods, list)
        assert all([isinstance(item, ifc.HookMethod) for item in methods])

        A = {'model_send': True,
             'split_packet': True,
             'change_tree': True,
             '_train_data': models.XYData(X, Y)}
        if (self._mpar.nsub > 0):
            A['_complement'] = models.XYData(X, Y, idx=False)
        for k, v in attrs.items():
            if (k not in A):
                A[k] = v
        M = [
            ifc.HookMethod(0, 'NODE_SPLIT', LDASplitPacket()),
            ifc.HookMethod(1, 'NODE_TERMINATE', LDATrackData())]
        mid = 2
        for item in methods:
            item.set_mid(mid)
            mid += 1
            M.append(item)
        self.clear_collection('track')
        packet = ifc.Packet(self.get_pid(), self, attrs=A, methods=M)
        self._root.send(packet)
        return (*self._collect['track'], )

    def predict_para(self, x, density=True):
        """Determine the Dirichlet parameter vector at the given points.

        Parameters
        ----------
        x : ndarray(N,D)
            evaluation points
        density : bool
            if set and density model is present, use it

        Returns
        -------
        ndarray(N,K)
            Dirichlet parameter vector a(x)
        """
        assert (self._mpar.nsub > 0)
        assert (isinstance(x, np.ndarray) and
                (x.ndim == 2) and
                (x.shape[1] == self._mpar.ndim)
               )
        assert isinstance(density, bool)

        N, D = x.shape
        K = self._mpar.nclass
        self._eval(x, {}, [])
        a = np.zeros((N,K))
        for tk in self._collect['track']:
            node, idx = tk
            a[idx,:] = node.model.predict_para(x[idx,:].reshape((-1,D)),
                self._mpar, density=density)
        return a

    def predict_proba(self, x, density=True):
        """Determine class probabilities at the given points.

        Parameters
        ----------
        x : ndarray(N,D)
            evaluation points
        density : bool
            if set and density model is present, use it

        Returns
        -------
        ndarray(N,K)
            class probability vector p(x)
        """
        assert (self._mpar.nsub > 0)
        assert (isinstance(x, np.ndarray) and
                (x.ndim == 2) and
                (x.shape[1] == self._mpar.ndim)
               )
        assert isinstance(density, bool)

        N, D = x.shape
        K = self._mpar.nclass
        self._eval(x, {}, [])
        a = np.zeros((N,K))
        for tk in self._collect['track']:
            node, idx = tk
            a[idx,:] = node.model.predict_para(x[idx,:].reshape((-1,D)),
                self._mpar, density=density)
        p = a / np.sum(a, axis=1).reshape((-1,1))
        return p

    def register_event(self, node, event):
        """Register a node event.

        Parameters
        ----------
        node : BNode object
            announcing node
        event : NodeEvent
            event type
        """
        assert isinstance(node, trees.BNode)
        assert isinstance(event, ifc.NodeEvent)

        pass

    @property
    def root(self):
        """Return the root node of the hierarchy.

        Returns
        -------
        BNode object
            root
        """
        return self._root

    def _eval(self, X, attrs, methods):
        """Evaluate data against the model. Internal method.

        Parameters
        ----------
        X : ndarray(N,D)
            N data points of dimension D
        attrs : dict
            user-defined attributes
        methods : list of HookMethod objects
            user-defined methods
        """

        A = {'model_send': True,
             'split_packet': True,
             '_eval_data': models.XData(X)}
        for k, v in attrs.items():
            if (k not in A):
                A[k] = v
        M = [
            ifc.HookMethod(0, 'NODE_SPLIT', LDASplitPacket()),
            ifc.HookMethod(1, 'NODE_TERMINATE', LDATrackData())]
        mid = 2
        for item in methods:
            item.set_mid(mid)
            mid += 1
            M.append(item)
        self.clear_collection('track')
        self._root.send(ifc.Packet(self.get_pid(), self, attrs=A, methods=M))

### EOF lda.py _________________________________________________________________

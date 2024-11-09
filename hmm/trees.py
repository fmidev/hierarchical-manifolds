"""
FILE:                   trees.py
COPYRIGHT:              (c) 2024 Finnish Meteorological Institute
                        P.O. BOX 503
                        FI-00101 Helsinki, Finland
                        https://www.fmi.fi/
LICENCE:                MIT
AUTHOR:                 Terhi Mäkinen (terhi.makinen@fmi.fi)
DESCRIPTION:

This module provides class definitions for a binary search tree structure used
in Hierarchical Data Analysis.
"""

from . import interfaces as ifc

class BNodePar(ifc.INodePar):
    """Binary tree parameters.

    Parameters
    ----------
    min_split : int
        minimal number of splits. Gain threshold is ignored until the minimum
        has been reached.
    max_split : int or None
        maximal allowed number of splits. If the maximal number of splits is set
        to None the size of the tree is only limited by gain. Virtual splits are
        not counted towards the total.
    epoch_split : bool
        if set, only allow node splitting at the end of an epoch. The maximal
        tree depth is thus restricted to the number of epochs.
    resplit : float
        resplitting coefficient. If the maximal number of splits has been
        reached and a leaf node requests splitting, the subleaf node with
        minimal gain is unsplit to make room if the requested split gain is
        larger than minimal gain times resplit. Value must be larger than one;
        too small a value may cause excessive resplitting, i.e., loss of
        acquired state.
    min_gain : float
        minimal splitting gain to allow node splitting.
    virtual : int
        number of successive virtual splits allowed past the first split below
        the gain threshold.
    """
    def __init__(self,
                 min_split = 0,
                 max_split = None,
                 epoch_split = False,
                 resplit = 1.05,
                 min_gain = 0.0,
                 virtual = 0
                ):
        assert (isinstance(min_split, int) and (min_split >= 0))
        assert ((max_split is None) or
                ((isinstance(max_split, int) and
                 (max_split >= min_split))
                )
               )
        assert isinstance(epoch_split, bool)
        assert isinstance(resplit, (int, float)) and (resplit > 1)
        assert (isinstance(min_gain, (int, float)) and (min_gain >= 0))
        assert isinstance(virtual, int) and (virtual >= 0)

        self._min_split = min_split
        self._max_split = max_split
        self._epoch_split = epoch_split
        self._resplit = resplit
        self._min_gain = min_gain
        self._virtual = virtual

    @property
    def epoch_split(self):
        """Return the epoch_split attribute.

        Returns
        -------
        bool
        """
        return self._epoch_split

    @property
    def max_split(self):
        """Return the max_split attribute.

        Returns
        -------
        int
        """
        return self._max_split

    @property
    def min_gain(self):
        """Return the min_gain attribute.

        Returns
        -------
        float
        """
        return self._min_gain

    @property
    def min_split(self):
        """Return the min_split attribute.

        Returns
        -------
        int
        """
        return self._min_split

    @property
    def resplit(self):
        """Return the resplit attribute.

        Returns
        -------
        float
        """
        return self._resplit

    @property
    def virtual(self):
        """Return the virtual attribute.

        Returns
        -------
        int
        """
        return self._virtual

# BNode declared hooks            POINT OF CALL:

ifc.hook.set('NODE_IN')         # on entering BNode.send()
ifc.hook.set('NODE_SEND')       # before starting model.send() iteration
ifc.hook.set('NODE_SEEK')       # request for resolving the next send nodes
ifc.hook.set('NODE_SPLIT')      # request for partitioning the packet
ifc.hook.set('NODE_TERMINATE')  # on not forwarding the packet
ifc.hook.set('NODE_OUT')        # on leaving BNode.send()

class BNode(ifc.INode):
    """Binary tree node.

    Parameters
    ----------
    nid : int
        node identifier
    parent : BNode object or None if root
        parent of this node
    model : IDataModel object
        attached data model
    virtual : int
        virtual depth
    """
    def __init__(self, nid, parent, model, virtual=0):
        assert isinstance(parent, BNode) or (parent is None)
        assert isinstance(model, ifc.IDataModel)

        self._nid = nid
        self._model = model
        self._parent = parent
        self._childa = None
        self._childb = None
        self._virtual = virtual

    def can_forward(self, gain, header, split=True):
        """Determine whether data can be forwarded to subnodes. Creates new
        subnodes if necessary and possible.

        Parameters
        ----------
        gain : float
            node splitting gain
        header : IDataAnalyser object
            top interface
        split : bool
            if True allows node splitting

        Returns
        -------
        bool
            True if data can be forwarded
        """
        assert isinstance(gain, (int, float))
        assert isinstance(header, ifc.IDataAnalyser)

        # a previously split node can always forward

        if self.is_split:
            return True

        # cannot forward if a node is unsplit and splitting is not permitted

        if (not split):
            return False

        # if a node is unsplit and split is only allowed on epoch, cannot split
        # and thus cannot forward

        root = header.root
        if root.par.epoch_split:
            return False

        # if a node is unsplit and is currently not splittable, cannot split and
        # thus cannot forward

        if (not self.is_splittable(gain, header)):
            return False

        # if a node is unsplit but splittable and tree is not full, split and
        # allow forwarding

        if (not root.is_full):
            self.split(gain, header)
            return True

        # if root is full and any splits cannot be found, cannot unsplit a node
        # to make room for splitting the current node and thus cannot forward
        # NB: this should not occur but it is covered for safety in the case
        #     that some combination of tree parameters would trigger it

        Gmin = root.min_gain
        if (Gmin is None):
            return False

        # if current gain does not reach the resplitting threshold, cannot split
        # or forward

        if (gain < root.par.resplit * Gmin):
            return False

        # resplit and allow forwarding

        root.min_node.virtualize(header)
        self.split(gain, header)
        return True

    @property
    def depth(self):
        """Return the depth of the node in the tree.

        Returns
        -------
        int
        """
        if (self._parent is None):
            return 0
        return self._parent.depth + 1

    def get_by_nid(self, nid):
        """Return node by its nid, or None if nid is not found. Since nids are
        assigned based on node creation order it is necessary to conduct a full
        tree search to locate a nid.

        Returns
        -------
        BNode object or None
            node
        """
        if (nid == self._nid):
            return self
        if (not self.is_split):
            return None
        node = self._childa.get_by_nid(nid)
        if (node is None):
            node = self._childb.get_by_nid(nid)
        return node

    def is_below(self, node):
        """Return True if this node is below the argument node in the tree.

        Returns
        -------
        bool
        """
        assert isinstance(node, BNode) or (node is None)

        if (node is None):
            return False
        pp = self._parent
        while pp is not None:
            if (pp._nid == node._nid):
                return True
            pp = pp._parent
        return False

    @property
    def is_leaf(self):
        """Return True if the node is a leaf node, i.e., non-virtual and either
        unsplit or split with virtual subnodes.

        Returns
        -------
        bool
        """
        if self.is_virtual:
            return False
        if (self._childa is None):
            return True
        return self._childa.is_virtual

    @property
    def is_root(self):
        """Return True if the node is the root node.

        Returns
        -------
        bool
        """
        return (self._parent is None)

    @property
    def is_split(self):
        """Return True if the node is a split node, i.e., has subnodes.

        Returns
        -------
        bool
        """
        return (self._childa is not None)

    def is_splittable(self, gain, header):
        """Determine whether the node is splittable.

        Parameters
        ----------
        gain : float
            split gain
        header : IDataAnalyser object
            top interface

        Returns
        -------
        bool
            True if node is splittable
        """
        assert isinstance(gain, (int, float))
        assert isinstance(header, ifc.IDataAnalyser)

        # is not splittable if the model itself cannot be split

        if (not self._model.is_splittable(header.par(self))):
            return False

        # if minimal number of splits has not yet been reached, is splittable
        # regardless of gain

        root = header.root
        par = root.par
        if (root.nsplits < par.min_split):
            return True

        # is splittable if gain is above threshold

        if (gain >= par.min_gain):
            return True

        # even if gain is below threshold is splittable if maximal virtual depth
        # has not yet been reached

        return (self._virtual < par.virtual)

    @property
    def is_subleaf(self):
        """Return True if the node is a subleaf, i.e., a node with subnodes
        that are leaves.

        Returns
        -------
        bool
        """
        return (self.is_split and
                self._childa.is_leaf and
                self._childb.is_leaf
               )

    @property
    def is_virtual(self):
        """Return True if the node is virtual.

        Returns
        -------
        bool
        """
        return (self._virtual > 0)

    @property
    def left(self):
        """Return a reference to the left child node of this node.

        Returns
        -------
        BNode object
            child node
        """
        return self._childa

    @property
    def model(self):
        """Return a reference to the attached model.

        Returns
        -------
        IDataModel object
            attached model
        """
        return self._model

    @property
    def nid(self):
        """Return node identifier.

        Returns
        -------
        int
        """
        return self._nid

    @property
    def parent(self):
        """Return a reference to the parent node of this node.

        Returns
        -------
        BNode object or None
            parent node
        """
        return self._parent

    def realize(self, header):
        """Remove one step of virtualization from the branch.

        Parameters
        ----------
        header : IDataAnalyser object
            top interface
        """
        assert isinstance(header, ifc.IDataAnalyser)

        root = header.root
        if self.is_split:
            self._childa.realize(header)
            self._childb.realize(header)
        if (self._virtual > 0):
            self._virtual -= 1
        else:
            root.register_event(self, ifc.NodeEvent.REALIZE)
            if root.is_pending:
                root.resolve()

    def remodel(self, model):
        """Replace attached model with the one given as an argument.

        Parameters
        ----------
        model : IDataModel object
            model to be attached
        """
        assert isinstance(model, ifc.IDataModel)

        self._model = model

    @property
    def right(self):
        """Return a reference to the right child node of this node.

        Returns
        -------
        BNode object
            child node
        """
        return self._childb

    @property
    def root(self):
        """Return a reference to the root node of the hierarchy.

        Returns
        -------
        BNode object
            root node
        """
        if (self._parent is None):
            return self
        return self._parent.root

    def send(self, packet):
        """Send a packet for processing. Intended to be called by an
        IDataAnalyser or another BNode instance.

        Parameters
        ----------
        packet : Packet object
            packet

        Attributes
        ----------
        change_tree : bool
            if set, allows changes in tree structure
        model_send : bool
            if set, send the packet to the model
        real_only : bool
            if set, do not send to virtual nodes
        seek : bool
            if set, forward in seek mode
        split_packet : bool
            if set, split the packet on forward
        _model_pending : bool
            True if packet should be sent again
        _model_pivot : ndarray(1,D)
            partitioning hyperspace pivot
        _send_count : int
            counter for calls to model.send()
        _send_nodes : list of BNode
            forwarding targets
        _subpackets : tuple(2) of Packet
            partitioned subpackets
        _terminate : bool
            if set, do not forward the packet
        _weights : tuple(2)
            tentative split weights
        """
        assert isinstance(packet, ifc.Packet)

        packet.call('NODE_IN', self)
        forward = True
        terminate = False
        if packet.is_set('model_send'):
            can_change = packet.is_set('change_tree')
            packet.call('NODE_SEND', self)
            header = packet.header
            review = True
            send_count = 0
            while review:
                send_count += 1
                packet.set_attr('_send_count', send_count)
                self._model.send(packet, self)
                was_split = self.is_split
                w = packet.get_attr('_weights')
                gain = self.split_gain(header, w=w)
                forward = self.can_forward(gain, header, split=can_change)
                realized = header.root.register_gain(self, gain)
                review = ((was_split is not self.is_split) or
                          realized or
                          packet.is_set('_model_pending'))
        if (forward and (not packet.is_set('_terminate'))):
            if packet.is_set('seek'):
                packet.call('NODE_SEEK', self)
                nodes = packet.get_attr('_send_nodes')
                if (len(nodes) == 0):
                    terminate = True
                else:
                    for node in nodes:
                        assert isinstance(node, BNode)
                        if (node is not self):
                            node.send(packet)
            elif (self.is_split and
                  ((self._childa._virtual == 0) or
                   (not packet.is_set('real_only'))
                  )
                 ):
                if packet.is_set('split_packet'):
                    if packet.is_set('_model_pivot'):
                        packet.call('NODE_SPLIT', self)
                        subpackets = packet.get_attr('_subpackets')
                        if (subpackets[0] is not None):
                            self._childa.send(subpackets[0])
                        if (subpackets[1] is not None):
                            self._childb.send(subpackets[1])
                    else:
                        terminate = True
                else:
                    self._childa.send(packet)
                    self._childb.send(packet)
            else:
                terminate = True
        else:
            terminate = True
        if terminate:
            packet.call('NODE_TERMINATE', self)
        packet.call('NODE_OUT', self)

    def split(self, gain, header):
        """Split a node if it is unsplit.

        Parameters
        ----------
        gain : float
            split gain
        header : IDataAnalyser object
            top interface
        """
        assert isinstance(gain, (int, float))
        assert isinstance(header, ifc.IDataAnalyser)

        if (not self.is_split):
            root = header.root
            par = root.par

            # never do a virtual split below the minimal number of splits
            # threshold

            if (root.nsplits < par.min_split):
                vstep = 0

            # do a virtual split if node is already virtual or gain is below the
            # gain threshold

            elif ((self._virtual > 0) or (gain < par.min_gain)):
                vstep = 1
            else:
                vstep = 0 
            self._childa = BNode(
                root.get_nid(),
                self, header.get_model(),
                self._virtual + vstep)
            root.register_event(self._childa, ifc.NodeEvent.NEW)
            self._childb = BNode(
                root.get_nid(),
                self, header.get_model(),
                self._virtual + vstep)
            root.register_event(self._childb, ifc.NodeEvent.NEW)
            root.register_event(self, ifc.NodeEvent.SPLIT)
            if root.is_pending:
                root.resolve()

    def split_gain(self, header, w=None):
        """Return node specific split gain.

        Parameters
        ----------
        header : IDataAnalyser object
            top interface
        w : tuple(2) or None
            unpropagated subnode weights

        Returns
        -------
        float
            split gain
        """
        assert isinstance(header, ifc.IDataAnalyser)
        assert ((w is None) or
                (isinstance(w, (list, tuple)) and
                 (len(w) == 2)
                )
               )

        if self.is_split:
            c = ((self._childa, header.par(self._childa)),
                 (self._childb, header.par(self._childb)))
            return -self._model.loss(header.par(self), c=c, w=w)
        elif (w is None):
            return 0.0
        return -self._model.loss(header.par(self), w=w)

    def t_tuplify(self, header=None, data=('nid', 'loss', 'virtual', 'children')
        ):
        """Test routine. Return a nested tuple of node attributes.

        Parameters
        ----------
        header : IDataAnalyser object or None
            top interface
        data : tuple of str
            requested attributes

        Returns
        -------
        nested tuple
            requested attributes
        """
        if (header is None):
            hdr = self.root.header
        else:
            hdr = header
        a = []
        if ('nid' in data):
            a.append(self._nid)
        if ('loss' in data):
            a.append(self._model.loss(hdr.par(self)))
        if ('virtual' in data):
            a.append(self._virtual)
        if ('children' in data):
            if self.is_split:
                a.append(self._childa.t_tuplify(header=hdr, data=data))
                a.append(self._childb.t_tuplify(header=hdr, data=data))
            else:
                a.append(None)
                a.append(None)
        return (*a, )

    def unsplit(self, header):
        """Unsplit a node if it is split.

        Parameters
        ----------
        header : IDataAnalyser object
            top interface
        """
        assert isinstance(header, ifc.IDataAnalyser)

        if self.is_split:
            self._childa.unsplit(header)
            self._childb.unsplit(header)
            root = header.root
            root.register_event(self, ifc.NodeEvent.UNSPLIT)
            root.register_event(self._childa, ifc.NodeEvent.DELETE)
            self._childa = None
            root.register_event(self._childb, ifc.NodeEvent.DELETE)
            self._childb = None
            if root.is_pending:
                root.resolve()

    def virtualize(self, header):
        """Add a step of virtualization to the branch.

        Parameters
        ----------
        header : IDataAnalyser object
            top interface
        """
        assert isinstance(header, ifc.IDataAnalyser)

        root = header.root
        if self.is_split:
            if (self._virtual == root.par.virtual):
                self.unsplit(header)
            else:
                self._childa._virtual = self._virtual + 1
                self._childb._virtual = self._virtual + 1
                self._childa.virtualize(header)
                self._childb.virtualize(header)
                if (self._virtual == 0):
                    root.register_event(self, ifc.NodeEvent.VIRTUALIZE)
                    if root.is_pending:
                        root.resolve()

class NodeMinGain(ifc.HookCall):
    """Hook call for finding minimal split gain subleaf node in the tree.

    Parameters
    ----------
    packet : Packet object
        packet
    node : BNode object
        node
    """
    def __call__(self, packet, node):
        assert isinstance(packet, ifc.Packet)
        assert isinstance(node, BNode)

        if (node.is_subleaf and (node not in packet.get_attr('exclude'))):
            min_gain = packet.get_attr('min_gain')
            gain = node.split_gain(packet.header)
            if ((min_gain is None) or (min_gain > gain)):
                packet.set_attr('min_gain', gain)
                packet.set_attr('min_node', node)

class BRoot(BNode):
    """Binary tree root node.

    Parameters
    ----------
    model : IDataModel object
        attached data model
    par : BNodePar object
        tree parameters
    header : IDataAnalyser object
        top interface
    """
    def __init__(self, model, par, header):
        assert isinstance(model, ifc.IDataModel)
        assert isinstance(par, BNodePar)
        assert isinstance(header, ifc.IDataAnalyser)
        super().__init__(0, None, model)

        self._header = header
        self._par = par
        self._nsplits = 0
        self._nidgen = 0
        self._min_gain = None
        self._min_node = None
        self._pending = False

    def clear(self):
        """Reset the tree.
        """
        self._model.clear()
        self._childa = None
        self._childb = None
        self._virtual = 0
        self._nsplits = 0
        self._nidgen = 0
        self._min_gain = None
        self._min_node = None
        self._pending = False

    def get_nid(self):
        """Return a new unique node identifier.

        Returns
        -------
        int
        """
        self._nidgen += 1
        return self._nidgen

    @property
    def header(self):
        """Return the header.

        Returns
        -------
        IDataAnalyser object
        """
        return self._header

    @property
    def is_full(self):
        """Return True if the maximal number of splits has been reached.

        Returns
        -------
        bool
        """
        return (self._nsplits == self._par.max_split)

    @property
    def is_pending(self):
        """Return True if there is a pending upkeep task in the queue.

        Returns
        -------
        bool
        """
        return self._pending

    @property
    def min_gain(self):
        """Return minimal leaf node gain.

        Returns
        -------
        float
        """
        return self._min_gain

    @property
    def min_node(self):
        """Return minimal gain leaf node.

        Returns
        -------
        BNode object
        """
        return self._min_node

    @property
    def nnodes(self):
        """Return number of nodes.

        Returns
        -------
        int
        """
        return 1 + 2 * self._nsplits

    @property
    def nsplits(self):
        """Return number of splits.

        Returns
        -------
        int
        """
        return self._nsplits

    @property
    def par(self):
        """Return a reference to BNodePar tree parameters.

        Returns
        -------
        BNodePar object
        """
        return self._par

    def pend(self):
        """Announce an upkeep task.
        """
        self._pending = True

    def regain(self, exclude=[]):
        """Conduct a full survey of node split gain.

        Parameters
        ----------
        exclude : list of BNode
            nodes to be excluded from consideration
        """
        packet = ifc.Packet(
            self._header.get_pid(),
            self._header,
            attrs={
                'min_gain':None,
                'min_node':None,
                'exclude':exclude},
            methods=[ifc.HookMethod(0, 'NODE_IN', NodeMinGain())])
        self.send(packet)
        self._min_gain = packet.get_attr('min_gain')
        self._min_node = packet.get_attr('min_node')

    def register_event(self, node, event):
        """Register node event.

        Parameters
        ----------
        node : BNode object
            node raising the event
        event : NodeEvent
            event type
        """
        assert isinstance(node, BNode)
        assert isinstance(event, ifc.NodeEvent)

        self._header.register_event(node, event)
        if (event == ifc.NodeEvent.NEW):
            pass
        elif (event == ifc.NodeEvent.DELETE):
            if (node is self._min_node):
                self.pend()
        elif (event == ifc.NodeEvent.SPLIT):
            if (node._childa._virtual == 0):
                self._nsplits += 1
        elif (event == ifc.NodeEvent.UNSPLIT):
            if (node._childa._virtual == 0):
                self._nsplits -= 1
            if (node is self._min_node):
                self.pend()
        elif (event == ifc.NodeEvent.REALIZE):
            if (node.is_split and (node._childa._virtual == 0)):
                self._nsplits += 1
                if ((node.parent is not None) and
                    (node.parent is self._min_node)
                   ):
                    self.pend()
        elif (event == ifc.NodeEvent.VIRTUALIZE):
            if (node.is_split and (node._childa._virtual == 1)):
                self._nsplits -= 1
                if (node is self._min_node):
                    self.pend()
        else:
            raise TypeError('Unknown event type')

    def register_gain(self, node, gain):
        """Register node split gain change.

        Parameters
        ----------
        node : BNode object
            node reporting gain change
        gain : float
            split gain value

        Returns
        -------
        bool
            True if node is realized by the gain
        """
        assert isinstance(node, BNode)
        assert isinstance(gain, (int, float))

        realized = False
        if (node.is_split and
            (node._childa._virtual == 1) and
            (gain >= self._par.min_gain)
           ):
            if self.is_full:
                if (self._min_gain is not None):
                    if (gain > self._par.resplit * self._min_gain):
                        self._min_node.virtualize(self._header, step=0)
                        node.realize(self._header)
                        realized = True
            else:
                node.realize(self._header)
                realized = True
        if node.is_subleaf:
            if ((self._min_gain is None) or (self._min_gain > gain)):
                self._min_gain = gain
                self._min_node = node
            elif ((node is self._min_node) and (gain > self._min_gain)):
                self.regain(exclude=[node])
        return realized

    def resolve(self):
        """Clear upkeep queue.
        """
        if self._pending:
            self.regain()
            self._pending = False

### EOF trees.py _______________________________________________________________

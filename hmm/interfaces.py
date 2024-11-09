"""
FILE:                   interfaces.py
COPYRIGHT:              (c) 2024 Finnish Meteorological Institute
                        P.O. BOX 503
                        FI-00101 Helsinki, Finland
                        https://www.fmi.fi/
LICENCE:                MIT
AUTHOR:                 Terhi Mäkinen (terhi.makinen@fmi.fi)
DESCRIPTION:

This module provides the basic communication interface between the tree and
model subsystems of Hierarchical Data Analysis.
"""

from enum import Enum

def is_immutable(x):
    return isinstance(x, (int, float, complex, str, bytes, tuple))

class Hooks(object):
    """A singleton object providing basic software hook identifier management.
    """
    _instance = None

    def __new__(cls):
        """Singleton constructor, instantiate only if not already instantiated.

        Parameters
        ----------
        cls : Hooks instance, or None
            singleton instance

        Returns
        -------
        Hooks instance
        """
        if cls._instance is None:
            cls._instance = super(Hooks, cls).__new__(cls)
            cls._instance._uid = 0
            cls._instance._hooks = {}
        return cls._instance

    def set(cls, token):
        """Bind a token to a unique hook identifier.

        Parameters
        ----------
        token : immutable
            token hook identifier

        Returns
        -------
        int
            unique hook identifier
        """
        assert is_immutable(token)
        if (token in cls._instance._hooks):
            raise ValueError('Hook ' + str(token) + ' already exists.')
        cls._instance._uid += 1
        cls._instance._hooks[token] = cls._instance._uid
        return cls._instance._uid

    def uid(cls, token, strict=False):
        """Return the unique hook identifier of a token.

        Parameters
        ----------
        token : immutable
            token hook identifier
        strict : bool
            raises an exception if set and token does not exist

        Returns
        -------
        int or None
            unique hook identifier, or None if token does not exist
        """
        assert is_immutable(token)
        if (token in cls._instance._hooks):
            return cls._instance._hooks[token]
        assert not strict, 'invalid hook ' + str(token)
        return None

    def is_uid(cls, uid):
        """Check whether a unique hook identifier is valid.

        Parameters
        ----------
        uid : int
            unique hook identifier

        Returns
        -------
        bool
            True if the identifier is reserved
        """
        if ((uid > 0) and uid <= cls._instance._uid):
            return True
        return False

hook = Hooks()  # instantiated unique identifier manager

class HookCall(object):
    """Prototype class for hook callable functions.

    Parameters
    ----------
    packet : Packet object
        packet
    node : INode object or None
        node
    """
    def __call__(self, packet, node):
        assert isinstance(packet, Packet)
        assert (node is None) or isinstance(node, INode)
        raise NotImplementedError()

class HookMethod(object):
    """Encapsulation of a hook identifier and callable function.

    Parameters
    ----------
    mid : int
        method identifier
    token : immutable
        token hook identifier
    method : HookCall object
        callable function
    """
    def __init__(self, mid, token, method):
        assert isinstance(mid, int)
        assert isinstance(method, HookCall)
        self._mid = mid
        self._uid = hook.uid(token, strict=True)
        self._func = method
        self._active = True

    def call(self, packet, uid, node):
        """Call the packet attached function if active and the hook matches.

        Parameters
        ----------
        packet : Packet object
            packet
        uid : int
            unique hook identifier
        node : INode object or None
            node
        """
        assert isinstance(packet, Packet)
        assert hook.is_uid(uid)
        assert (node is None) or isinstance(node, INode)
        if ((uid == self._uid) and self._active):
            self._func(packet, node)

    @property
    def is_active(self):
        """Return the active status of the attached function.

        Returns
        -------
        bool
        """
        return self._active

    @property
    def mid(self):
        """Return the method identifier.

        Returns
        -------
        int
        """
        return self._mid

    def set_active(self, state):
        """Set the active status of the attached function.

        Parameters
        ----------
        state : bool
            function active state
        """
        assert isinstance(state, bool)
        self._active = state

    def set_mid(self, mid):
        """Set the method identifier.

        Parameters
        ----------
        mid : int
            method identifier
        """
        assert isinstance(mid, int)
        self._mid = mid

    @property
    def uid(self):
        """ Return the respective unique hook identifier.

        Returns
        -------
        int
        """
        return self._uid

hook.set('PACKET_INIT')
hook.set('PACKET_DEL')

class Packet(object):
    """Encapsulation of data and methods for communication between subsystems.

    Parameters
    ----------
    pid : int
        packet identifier
    header : IDataAnalyser object
        top interface
    attrs : dict
        attached key and data values
    methods : list of HookMethod objects
        packet methods
    """
    def __init__(self, pid, header, attrs={}, methods=[]):
        assert isinstance(pid, int)
        assert isinstance(header, IDataAnalyser)
        assert isinstance(attrs, dict)
        assert isinstance(methods, list)
        assert all([isinstance(item, HookMethod) for item in methods])

        self._pid = pid
        self._header = header
        self._attrs = attrs
        self._methods = methods
        self.call('PACKET_INIT')

    def __del__(self):
        """Packet destructor.
        """
        self.call('PACKET_DEL')

    def add_method(self, method):
        """Add a new method to the packet if it does not exist already.

        Parameters
        ----------
        method : HookMethod object
            packet method
        """
        assert isinstance(method, HookMethod)
        for m in self._methods:
            if (m.mid == method.mid):
                return
        self._methods.append(method)

    @property
    def attrs(self):
        """Return attached attributes.

        Returns
        -------
        dict
        """
        return self._attrs

    def call(self, token, node=None):
        """Hook call.

        Parameters
        ----------
        token : immutable
            token hook identifier
        node : INode object
            node
        """
        assert is_immutable(token)
        assert (node is None) or isinstance(node, INode)
        uid = hook.uid(token, strict=True)
        for method in self._methods:
            method.call(self, uid, node)

    def clone(self, packet, full=False, overwrite=False):
        """Clone attributes and methods from another packet.

        Parameters
        ----------
        packet : Packet object
            source
        full : bool
            if set, also copy attributes with key names starting with an
            underscore
        overwrite : bool
            if set, overwrite existing keys
        """
        assert isinstance(packet, Packet)
        assert isinstance(full, bool)
        assert isinstance(overwrite, bool)

        if full:
            if overwrite:
                for k, v in packet._attrs.items():
                    self._attrs[k] = v
            else:
                for k, v in packet._attrs.items():
                    if (k not in self._attrs):
                        self._attrs[k] = v
        else:
            if overwrite:
                for k, v in packet._attrs.items():
                    if (k[0] != '_'):
                        self._attrs[k] = v
            else:
                for k, v in packet._attrs.items():
                    if ((k[0] != '_') and (k not in self._attrs)):
                        self._attrs[k] = v
        for item in packet._methods:
            isnew = True
            for m in self._methods:
                if (m.mid == item.mid):
                    isnew = False
                    break
            if isnew:
                self._methods.append(item)

    def get_attr(self, key):
        """Get attribute from the packet.

        Parameters
        ----------
        key : immutable
            token identifier

        Returns
        -------
        any
            respective data, or None if key does not exist
        """
        assert is_immutable(key)
        if (key not in self._attrs):
            return None
        return self._attrs[key]

    @property
    def header(self):
        """Return packet header.

        Returns
        -------
        IDataAnalyser object
        """
        return self._header

    def is_set(self, key):
        """Return True if key exists in attrs and the respective value is not
        None or False.

        Parameters
        ----------
        key : immutable
            token identifier

        Returns
        -------
        bool
        """
        assert is_immutable(key)
        if (key not in self._attrs):
            return False
        if (self._attrs[key] is None):
            return False
        if isinstance(self._attrs[key], bool):
            return self._attrs[key]
        return True

    @property
    def methods(self):
        """Return packet methods.

        Returns
        -------
        list of HookMethod objects
        """
        return self._methods

    @property
    def pid(self):
        """ Return the packet identifier.

        Returns
        -------
        int
        """
        return self._pid

    def remove_attr(self, key):
        """Remove an existing attribute from the packet.

        Parameters
        ----------
        key : immutable
            token identifier
        """
        assert is_immutable(key)
        if (key in self._attrs):
            del self._attrs[key]

    def set_attr(self, key, value):
        """Set an attribute to the packet.

        Parameters
        ----------
        key : immutable
            token identifier
        value : untyped
            respective data
        """
        assert is_immutable(key)
        self._attrs[key] = value

class NodeEvent(Enum):
    """Tracked node events.
    NEW         a node is instantiated
    DELETE      a node is deleted
    SPLIT       a node is split, i.e., has children added to it
    UNSPLIT     a node is unsplit, i.e., its children are removed
    REALIZE     a virtual node becomes realized
    VIRTUALIZE  a real node becomes virtualized
    """
    NEW = 1
    DELETE = 2
    SPLIT = 3
    UNSPLIT = 4
    REALIZE = 5
    VIRTUALIZE = 6

class INodePar(object):
    """Node parameter interface.

    This is the interface definition.
    """
    def __init__(self): pass

class INode(object):
    """Node interface.

    This is the interface definition.
    """
    def __init__(self): pass

    @property
    def model(self):
        """Return a reference to the attached model.

        Returns
        -------
        IDataModel object
        """
        raise NotImplementedError()

    def send(self, packet):
        """Send a packet for processing. Intended to be called by an
        IDataAnalyser or another INode instance.

        Parameters
        ----------
        packet : Packet object
            packet
        """
        assert isinstance(packet, Packet)
        raise NotImplementedError()

class IDataPar(object):
    """Data model parameter interface.

    This is the interface definition.
    """
    def __init__(self): pass

class IDataModel(object):
    """Data model interface.

    This is the interface definition.
    """
    def __init__(self): pass

    def clear(self):
        """Reset the model.
        """
        raise NotImplementedError()

    def is_splittable(self, par):
        """Return True if the model allows splitting.

        Parameters
        ----------
        par : IDataPar object
            model parameters
        """
        assert isinstance(par, IDataPar)
        raise NotImplementedError()

    def loss(self, par, c=None, w=None):
        """Return the internal loss function value of the model, or if either
        c, w, or both are provided, return the loss differential (gain).

        Parameters
        ----------
        par : IDataPar object
            model parameters
        c : tuple(2) of tuple(2) of (INode, IDataPar) or None
            child nodes and their parameters
        w : tuple(2) of untyped or None
            additional weights

        Returns
        -------
        float
            loss
        """
        assert isinstance(hdr, IDataPar)
        assert ((c is None) or
                (isinstance(c, (list, tuple)) and
                 (len(c) == 2) and
                 isinstance(c[0], (list, tuple)) and
                 (len(c[0]) == 2) and
                 isinstance(c[0][0], INode) and
                 isinstance(c[0][1], IDataPar) and
                 isinstance(c[1], (list, tuple)) and
                 (len(c[1]) == 2) and
                 isinstance(c[1][0], INode) and
                 isinstance(c[1][1], IDataPar)
                )
               )
        assert ((w is None) or
                isinstance(w, (list, tuple)) and
                (len(w) == 2)
               )
        raise NotImplementedError()

    def send(self, packet, node):
        """Send a packet for processing. Intended to be called by an INode
        instance.

        Parameters
        ----------
        packet : Packet object
            packet
        node : INode object
            sending node
        """
        assert isinstance(packet, Packet)
        assert isinstance(node, INode)
        raise NotImplementedError()

class IDataAnalyser(object):
    """ Data analyser application.

    This is the interface definition.
    """
    def __init__(self): pass

    def get_model(self):
        """Return a new instance of IDataModel.

        Returns
        -------
        IDataModel object
        """
        raise NotImplementedError()

    def get_pid(self):
        """Return a new Packet identifier.

        Returns
        -------
        int
        """
        raise NotImplementedError()

    def par(self, node):
        """Return model parameters for IDataModel.

        Parameters
        ----------
        node : INode object
            model owning node

        Returns
        -------
        IDataPar object
            model parameters
        """
        assert isinstance(node, INode)
        raise NotImplementedError()

    def register_event(self, node, event):
        """Register node event.

        Parameters
        ----------
        node : INode object
            model owning node
        event : NodeEvent
            node event
        """
        assert isinstance(node, INode)
        assert isinstance(event, NodeEvent)
        raise NotImplementedError()

    @property
    def root(self):
        """Return a reference to the root node of the model hierarchy.

        Returns
        -------
        INode object
            root node
        """
        raise NotImplementedError()

### EOF interfaces.py __________________________________________________________

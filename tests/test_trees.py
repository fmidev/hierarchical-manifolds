# test the trees.py module
#
# Usage: python -m unittest tests.test_trees

import unittest
from hmm import interfaces as ifc
from hmm import trees

# Dummy data model for testing trees

class TestPar(ifc.IDataPar):
    """ Data model parameter interface.
    """
    def __init__(self): pass

class TestModel(ifc.IDataModel):
    """ Data model interface.
        Input:
            par : TestPar object
    """
    def __init__(self, par):
        assert isinstance(par, TestPar)

        self._loss = 0
        self._splittable = False

    def is_splittable(self, par):
        """ Return True if the model allows splitting.
            Input:
                par : TestPar object
        """
        assert isinstance(par, TestPar)

        return self._splittable

    def loss(self, par, w=None):
        """ Return the internal loss function value of the model.
            Input:
                par : TestPar object
                w   : float, additional weight, or None
            Output:
                float, loss
        """
        assert isinstance(par, TestPar)
        assert ((w is None) or isinstance(w, (int, float)))

        if (w is None):
            return self._loss
        return self._loss + w

    def send(self, packet, node):
        """ Send a packet for processing. Intended to be called by a BNode
            instance.
            Input:
                packet : Packet object
                node   : BNode object
        """
        assert isinstance(packet, ifc.Packet)
        assert isinstance(node, trees.BNode)

        send_count = packet.get_attr('send_count')
        if (send_count == 1):
            header = packet.get_attr('header')
            data = packet.get_attr('train_data')
            if (data is not None):
                assert isinstance(data, tuple) and (len(data) == 3)
                packet.set_attr('model_change', True)
                self._loss += data[0]
                if (data[1] is not None):
                    self._splittable = True
                    pa = ifc.Packet(header.get_pid(),
                                    attrs={'header': header,
                                           'model_send': True,
                                           'split_packet': True,
                                           'train_data': data[1]
                                          }
                                   )
                    pb = ifc.Packet(header.get_pid(),
                                    attrs={'header': header,
                                           'model_send': True,
                                           'split_packet': True,
                                           'train_data': data[2]
                                          }
                                   )
                    packet.set_attr('subpackets', (pa, pb))
                    packet.set_attr('weights', (data[1][0], data[2][0]))
        else:
            packet.set_attr('model_change', False)

class TestAnalyser(ifc.IDataAnalyser):
    """ Data analyser (user) interface.
    """
    def __init__(self, *args, **kwargs):
        self._mgen = TestModel
        self._mpar = None
        self._tpar = None
        for arg in args:
            if (isinstance(arg, type) and issubclass(arg, TestModel)):
                self._mgen = arg
            elif isinstance(arg, TestPar):
                self._mpar = arg
            elif isinstance(arg, trees.BNodePar):
                self._tpar = arg
        if (self._mpar is None):
            self._mpar = TestPar(*[arg for arg in args
                                   if ((not isinstance(arg, type)) and
                                       (not isinstance(arg, trees.BNodePar))
                                      )
                                  ], **kwargs)
        if (self._tpar is None):
            self._tpar = trees.BNodePar()
        self._pidgen = -1
        self._root = trees.BRoot(self.get_model(), self._tpar, self)

    def get_model(self):
        """ Return a new instance of TestModel.
        """
        return self._mgen(self._mpar)

    def get_pid(self):
        """ Return a new Packet identifier.
        """
        self._pidgen += 1
        return self._pidgen

    def loss(self, w):
        """ Return the internal loss function value for the given weights.
            Input:
                w : float, weight
            Output:
                float, loss
        """
        assert isinstance(w, (int, float))

        return w

    def par(self, node):
        """ Return model parameters for IDataModel.
        """
        assert (node is None) or isinstance(node, trees.BNode)

        return self._mpar

    def register_event(self, node, event):
        """ Register node event.
        """
        assert isinstance(node, trees.BNode)
        assert isinstance(event, ifc.NodeEvent)

        pass

    @property
    def root(self):
        """ Return a reference to the root node of the model hierarchy.
        """
        return self._root

    def train(self, X):
        """ Add data to model.
            Input:
                X : nested tuple of (loss, child_a, child_b)
        """
        self._root.send(ifc.Packet(self.get_pid(),
                                   attrs={'header': self,
                                          'model_send': True,
                                          'split_packet': True,
                                          'train_data': X
                                         }
                                  )
                       )

class TestAlgorithms(unittest.TestCase):
    def test_train(self):
        """ Test basic tree structure construction.
        """
        X = (1.0,
             (0.3,
              (0.1, None, None),
              (0.1, None, None)
             ),
             (0.5,
              (0.1, None, None),
              (0.1, None, None)
             )
            )
        T = TestAnalyser()
        T.train(X)
        self.assertEqual(T.root.nsplits, 3)                                    #
        self.assertEqual(T.root.t_tuplify(data=('loss', 'children')), X)       #
        T2 = TestAnalyser(trees.BNodePar(max_split=2))
        T2.train(X)
        XO = (0, 1.0, 0,
              (1, 0.3, 0, None, None),
              (2, 0.5, 0,
               (5, 0.1, 0, None, None),
               (6, 0.1, 0, None, None)
              )
             )
        self.assertEqual(T2.root.nsplits, 2)                                   #
        self.assertEqual(T2.root.t_tuplify(), XO)                              #
        T3 = TestAnalyser(trees.BNodePar(max_split=2, virtual=1))
        T3.train(X)
        XV = (0, 1.0, 0,
              (1, 0.3, 0,
               (3, 0.1, 1, None, None),
               (4, 0.1, 1, None, None)
              ),
              (2, 0.5, 0,
               (5, 0.1, 0, None, None),
               (6, 0.1, 0, None, None)
              )
             )
        self.assertEqual(T3.root.nsplits, 2)                                   #
        self.assertEqual(T3.root.t_tuplify(), XV)                              #

if __name__ == '__main__':
    unittest.main()

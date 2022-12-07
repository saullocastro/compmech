from itertools import groupby
from collections import OrderedDict

import numpy as np


class Model(object):

    def __init__(self):
        self.els = []
        self.ncoords = {}
        self.props = {}
        self.ptypes = {} #TODO must be filled based on all props implemented

    def add_property(self, id, prop):
        self.props[id] = prop

    def add_node(self, id, coords):
        self.ncoords[id] = coords

    def add_element(self, id, nids, pid):
        elem = dict(id=id, nids=nids, pid=pid)
        self.els.append(elem)

    def assemble(self):
        for prop in self.props:
            prop.el = []
        self.nnodes = []
        self.nseq = []
        self.nids = []
        self.elids = []
        for el in self.els:
            el['ptypenum'] = self.props[el.pid].ptypenum
            self.elids.append(el['id'])
            self.nseq += [nid for nid in el['nids']]
            self.nnodes.append(len(el['nids']))
        self.nids = np.unique(self.nseq)
        self.pos = np.searchsorted(self.nids, self.nseq)

        for ptypenum, els in groupby(self.els, lambda el: el['ptypenum']):
            for el in els:
                self.ptypes[el.ptypenum].els.append(el)

        # initiating the global stiffness matrices
        approx_size = int(0.05*(sum(self.nnodes)*6)**2)
        k0r = np.zeros(approx_size, dtype=np.intc)
        k0c = np.zeros(approx_size, dtype=np.intc)
        k0v = np.zeros(approx_size, dtype=np.float64)

        # stiffness matrices in the local coordinate system
        for ptype in self.ptypes.values():
            if not ptype.els:
                continue

            els = ptype.els
            nel = len(els)
            Cs_dict = OrderedDict()
            Cind = []
            for el in els:
                prop = self.props[el.pid]
                Cs_dict[el.pid] = prop.C
                Cind.append(el.pid)
            Cind = np.searchsorted(np.unique(Cind), Cind).view(np.intc)

            psize = ptype.psize
            Cs = np.zeros(len(Cs_dict)*psize, dtype=np.float64)
            for i, C in enumerate(Cs_dict.values()):
                Cs[i*psize:(i+1)*psize] = C

            nk0 = ptype.nk0
            k0 = np.zeros(nk0*nel, dtype=np.float64)

            pnseq = []
            for el in els:
                pnseq += el['nids']
            coords = np.array([self.ncoords[nid] for nid in pnseq],
                              dtype=np.float64)

            ptype.calc_el_attr(coords)

            # transform the node coordinates to the local element
            # calculate jacobians
            # calculate other arrays required by the property type
            #  - such as rs with the radii
            #  - such as arads with the semi-vertex angles


class Property(object):
    def __init__(self):
        self.ptype = None
        self.pid = None
        self.C = None

class Shell2DFSDT(Property):

    def __init__(self, stack, plyts, laminaprops):
        from compmech.composite.laminate import read_stack
        super(Property, self).__init__()
        lam = read_stack(stack, plyts=plyts, laminaprops=laminaprops)
        self.C = lam.ABDE[np.triu_indices_from(lam.ABDE)]
        self.psize = len(self.C)


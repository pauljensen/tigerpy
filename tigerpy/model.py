
import copy
import collections

import numpy as np

import cobra
import cobra.test
from . import parsing


class TigerModel(object):
    def __init__(self, fields):
        self.fields = copy.deepcopy(fields)
        self.c = self.fields['c']
        self.S = self.fields['S']
        self.b = self.fields['b']
        self.lb = self.fields['lb']
        self.ub = self.fields['ub']
        self.genes = self.fields['genes']

        grRules = self.fields['grRules']
        self.gprs = list(map(parsing.parse_boolean_string, grRules))
        self.rules = list(map(lambda x: x.to_rule(), self.gprs))

    @classmethod
    def from_cobra(cls, cb):
        fields = cobra.io.mat.create_mat_dict(cb)
        return TigerModel(fields)

    def eval_gprs(self, gene_dict, default=True):
        x = collections.defaultdict(lambda: default, gene_dict)
        n = self.S.shape[1]
        rxn_states = np.ndarray(n, dtype=bool)
        for i in range(n):
            if self.rules[i]:
                rxn_states[i] = eval(self.rules[i])
            else:
                rxn_states[i] = True
        return rxn_states


if __name__ == '__main__':
    cb = cobra.test.create_test_model('textbook')
    tiger = TigerModel.from_cobra(cb)
    print(tiger.gprs[0].operands)
    print(tiger.rules)

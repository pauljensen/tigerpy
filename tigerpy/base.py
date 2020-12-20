
import copy
import collections

import numpy as np
import gurobi as gp
from gurobi import GRB

import cobra
import cobra.test
from . import parsing


def get_vars_by_name(model, names, container="list"):
    v = [model.getVarByName(name) for name in names]
    if container.lower() == "list":
        return v
    elif container.lower() == "dict":
        return dict(zip(names, v))
    elif container.lower() == "mvar":
        return gp.MVar(v)


class TigerModel(object):
    def __init__(self, fields):
        self.fields = copy.deepcopy(fields)
        self.c = self.fields['c']
        self.S = self.fields['S']
        self.b = self.fields['b']
        self.lb = self.fields['lb']
        self.ub = self.fields['ub']
        self.genes = list(self.fields['genes'])
        self.rxns = list(self.fields['rxns'])

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
    
    def get_exchange_rxns(self, names=False):
        idxs = (self.S != 0).sum(0) == 1
        if names:
            return self.rxns[idxs]
        else:
            return idxs
    
    def make_base_model(self, add_gpr=True, genes="continuous", bounds="weak", **kwargs):
        model = gp.Model()
        n = self.S.shape[1]
        for i in range(n):
            model.addVar(lb=self.lb[i], ub=self.ub[i], name=self.rxns[i])
        model.update()
        
        v = get_vars_by_name(model, self.rxns, "MVar")
        model.setObjective(self.c @ v, GRB.MAXIMIZE)
        model.addConstr(self.S @ v == self.b, name="Sv=b")
        model.update()
        
        if add_gpr:
            if genes == "continuous" and bounds == "weak":
                add_gprs_cnf_weak_(self, model, **kwargs)
            # elif genes == "continuous" and bounds == "strong":
            #     # FALCON
            # elif genes == "discrete" and bounds == "weak":
            #     # CNF with binary variables
            # elif genes == "discrete" and bounds == "strong":
            #     # SR-FBA
                
            g = get_vars_by_name(model, self.genes, "MVar")
        else:
            g = None
            
        return model, v, g


def add_gprs_cnf_weak_(tiger, model, gene_ub=None, gpr_suffix="__GPR", **kwargs):
    rxns = get_vars_by_name(model, tiger.rxns)
    
    if gene_ub is None:
        ub = GRB.INFINITY
    else:
        ub = gene_ub
    for gene in tiger.genes:
        model.addVar(lb=0.0, ub=ub, name=gene)
    model.update()
    g = get_vars_by_name(model, tiger.genes, "dict")
    
    for v, gpr in zip(rxns, tiger.gprs):
        if gpr.is_empty():
            continue
        
        cnfs = parsing.make_cnf(gpr, names_only=True)
        for i, cnf in enumerate(cnfs):
            genes = [g[name] for name in cnf]
            if gene_ub is None:
                model.addConstr(
                    v <= gp.quicksum(genes),
                    name = v.varname + gpr_suffix + '-UB[' + str(i) + ']')
                model.addConstr(
                    v >= gp.quicksum(genes),
                    name = v.varname + gpr_suffix + '-LB[' + str(i) + ']')
            else:
                model.addConstr(
                    v <= v.ub/ub * gp.quicksum(genes),
                    name = v.varname + gpr_suffix + '-UB[' + str(i) + ']')
                model.addConstr(
                    v >= v.lb/ub * gp.quicksum(genes),
                    name = v.varname + gpr_suffix + '-LB[' + str(i) + ']')
    
    model.update()



if __name__ == '__main__':
    cb = cobra.test.create_test_model('textbook')
    tiger = TigerModel.from_cobra(cb)
    print(tiger.gprs[0].operands)
    print(tiger.rules)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:28:32 2020

@author: jensen
"""


import cobra
import model
import base

from gurobi import GRB
import numpy as np

smu_cobra = cobra.io.read_sbml_model("iSMUv01_CDM.xml")
smu = model.TigerModel.from_cobra(smu_cobra)

# open the exchanges
excs = (smu.S != 0).sum(0) == 1
smu.lb[excs] = -100
smu.ub[excs] = 100

# git rid of NGAM
smu.lb[smu.lb > 0] = 0

#idxs = range(100)
#simple = base.simple_flux_coupling(smu, idxs)
#coupled = base.flux_coupling(smu)


def min_rxn_binary(tiger):
    model, v = base.make_base_model(tiger)
    base.add_objective_constraint(model)
    x = model.addMVar(v.shape[0], name="x", vtype=GRB.BINARY)
    model.addConstr(v <= np.diag(v.UB) @ x)
    model.addConstr(v >= np.diag(v.LB) @ x)
    model.setObjective(sum(x), sense=GRB.MINIMIZE)
    model.optimize()
    return model, x

def min_rxn_l1(tiger):
    model, v = base.make_base_model(tiger)
    base.add_objective_constraint(model)
    x = model.addMVar(v.shape[0], name="x", lb=0.0, ub=1.0)
    model.addConstr(v <= np.diag(v.UB) @ x)
    model.addConstr(v >= np.diag(v.LB) @ x)
    model.setObjective(sum(x), sense=GRB.MINIMIZE)
    model.optimize()
    return model, x

m, bx = min_rxn_binary(smu)
m, lx = min_rxn_l1(smu)
    
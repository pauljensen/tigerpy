#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:28:04 2020

@author: jensen
"""

import cobra
import model
import base

import gurobi as gp
from gurobi import GRB
import numpy as np
import matplotlib.pyplot as plt


## ------------ loading iSMU model ------------

smu_cobra = cobra.io.read_sbml_model("iSMUv01_CDM.xml")
smu = model.TigerModel.from_cobra(smu_cobra)

# open the exchanges
#excs = (smu.S != 0).sum(0) == 1
#smu.lb[excs] = -100
#smu.ub[excs] = 100

# git rid of NGAM
#smu.lb[smu.lb > 0] = 0

model, v = base.make_base_model(smu)

def make_dual_model(tiger):
    model = gp.Model()
    n = tiger.S.shape[1]
    m = tiger.S.shape[0]
    yS = model.addMVar(shape=m, lb=-GRB.INFINITY, name="yS")
    yL = model.addMVar(shape=n, lb=-GRB.INFINITY, ub=0.0, name="yL")
    yU = model.addMVar(shape=n, name="yU")
    model.addConstr(tiger.S.T @ yS + np.eye(n) @ yL + np.eye(n) @ yU == tiger.c)
    model.setObjective(tiger.b @ yS + tiger.lb @ yL + tiger.ub @ yU)
    model.update()
    return model, (yS, yL, yU)

dual, ys = make_dual_model(smu)
dual.optimize()

# fix the dual objective
dual.addConstr(dual.getObjective() == dual.getObjective().getValue(), name="duality")
dual.setObjective(ys[0]@ys[0] + ys[1]@ys[1] + ys[2]@ys[2])
dual.update()
dual.optimize()

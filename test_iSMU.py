#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:28:32 2020

@author: jensen
"""


import cobra
import tigerpy as tiger

from gurobi import GRB
import numpy as np

smu_cobra = cobra.io.read_sbml_model("test/iSMUv01_CDM.xml")
smu = tiger.TigerModel.from_cobra(smu_cobra)

# open the exchanges
excs = (smu.S != 0).sum(0) == 1
smu.lb[excs] = -100
smu.ub[excs] = 100

# git rid of NGAM
smu.lb[smu.lb > 0] = 0

#idxs = range(100)
#simple = base.simple_flux_coupling(smu, idxs)
#coupled = base.flux_coupling(smu)

model, v, g = smu.make_base_model()

    
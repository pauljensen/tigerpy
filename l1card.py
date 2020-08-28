#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cobra
import model
import base

from gurobi import GRB
import numpy as np
import matplotlib.pyplot as plt


## ------------ loading iSMU model ------------

smu_cobra = cobra.io.read_sbml_model("iSMUv01_CDM.xml")
smu = model.TigerModel.from_cobra(smu_cobra)

# open the exchanges
excs = (smu.S != 0).sum(0) == 1
smu.lb[excs] = -100
smu.ub[excs] = 100

# git rid of NGAM
smu.lb[smu.lb > 0] = 0

model, v = base.make_base_model(smu)

## ------------ helper functions ------------

INT_TOL = 1e-8

def polish(x, tol=INT_TOL):
    x[np.abs(x) <= tol] = 0
    x[np.abs(x) >= 1 - tol] = 1
    return x

def binarize(x, tol=INT_TOL):
    x = polish(x, tol)
    x[x > 0] = 1
    return x

def card(x, tol=None):
    if tol is not None:
        x = polish(x, tol)
    return np.sum(x != 0)

def splits(x, tol=1e-16, uppertol=0.1):
    ones = np.where(np.abs(x) >= uppertol)[0]
    zeros = np.where(np.abs(x) <= tol)[0]
    indeterminate = np.where(np.logical_and(np.abs(x) > tol, np.abs(x) < uppertol))[0]
    return zeros, indeterminate, ones

def splitcounts(x, **kwargs):
    s = splits(x, **kwargs)
    return list(map(lambda y: y.shape[0], s))

def add_fva_bounds(model, v, frac=1.0):
    model.Params.OutputFlag = 0
    
    if frac is not None:
        base.add_objective_constraint(model, frac)
    
    orig_obj = model.getObjective()
    orig_sense = model.ModelSense
    
    for i in range(v.shape[0]):
        model.setObjective(v[i] + 0, GRB.MINIMIZE)
        model.optimize()
        v[i].LB = v[i].X
        
        model.setObjective(v[i] + 0, GRB.MAXIMIZE)
        model.optimize()
        v[i].UB = v[i].X
        
    model.setObjective(orig_obj, orig_sense)
    model.update()
    
def is_valid_solution(model, x, sol):
    prev_ub = np.copy(x.UB)
    x.UB = binarize(sol)
    model.optimize()
    status = model.status
    x.UB = prev_ub
    model.update()
    return status == GRB.OPTIMAL
    

## ------------ min card(v) ------------

def min_rxn_binary(model, v, niter=5, fixiter=True):
    base.add_objective_constraint(model)
    x = model.addMVar(v.shape[0], name="x", vtype=GRB.BINARY)
    U = np.copy(v.ub)
    L = np.copy(v.lb)
    model.setObjective(sum(x), sense=GRB.MINIMIZE)
    X = np.zeros((x.shape[0],niter))
    sol = np.ones(x.shape)
    for i in range(niter):
        model.addConstr(v <= np.diag(U) @ x)
        model.addConstr(v >= np.diag(L) @ x)
        model.optimize()
        sol = binarize(np.copy(x.X), tol=1e-16)
        X[:,i] = x.X
        print(card(sol))
        U[sol==0] = U[sol==0] / 10.0
        L[sol==0] = L[sol==0] / 10.0
        if fixiter:
            zeros, indeter, ones = splits(X[:,i])
            x[zeros].lb = 0
            x[zeros].ub = 0
            x[ones].lb = 1
            x[ones].ub = 1
            x[indeter].lb = 0
            x[indeter].ub = 1
    return model, x, X

def min_rxn_l1_abs(model, v, niter=5, gamma=0.001):
    base.add_objective_constraint(model)
    vf = model.addMVar(v.shape[0], name="vf", lb=0.0, ub=v.ub)
    vr = model.addMVar(v.shape[0], name="vr", lb=0.0, ub=-v.lb)
    model.addConstr(v == vf - vr)
    V = np.ones((v.shape[0],niter))
    wf = np.ones(vf.shape)
    wr = np.ones(vr.shape)
    print("starting iterations")
    for i in range(niter):
        model.setObjective(wf @ vf + wr @ vr , sense=GRB.MINIMIZE)
        model.optimize()
        V[:,i] = v.X
        wf = 1.0 / (gamma + np.abs(vf.X))
        wr = 1.0 / (gamma + np.abs(vr.X))
    return model, v, V

## ------------ running with iSMU ------------

bmod1, bv1 = base.make_base_model(smu)
bmod1.Params.OutputFlag = 0
_, bx1, X = min_rxn_binary(bmod1, bv1)
#print(is_valid_solution(bmod1, bx1, bx1.X))
print(splitcounts(X[:,0]))
print(splitcounts(X[:,-1]))



#bmod2, bv2 = base.make_base_model(smu)
#add_fva_bounds(bmod2, bv2, 0.8)
#_, bx2 = min_rxn_binary(bmod2, bv2)
#print(is_valid_solution(bmod2, bx2, bx2.X))

lmod, lv = base.make_base_model(smu)
_, lx, lX = min_rxn_l1_abs(lmod, lv)

#print(card(bx1.X), card(bx2.X), card(lx.X))

#plt.ylim([1e-18,1000])
#plt.ylim([0,0.01])
#for i in range(2):
#    #plt.semilogy(range(lX.shape[0]), lX[:,i], '.')
#    plt.plot(range(lX.shape[0]), lX[:,i], '.')

aslog = True

if not aslog:
    plt.ylim([-0.1,0.1])
    plt.plot(range(lX.shape[0]), lX[:,0], '.')
    plt.plot(range(lX.shape[0]), lX[:,-1], '.')
    
if aslog:
    plt.ylim([1e-18,1000])
    plt.semilogy(range(lX.shape[0]), np.abs(lX[:,0] + 1e-17), '.')
    plt.semilogy(range(lX.shape[0]), np.abs(lX[:,-1] + 1e-17), '.')
    
if True:
    print(splitcounts(lX[:,0],tol=1e-16))
    print(splitcounts(lX[:,1],tol=1e-16))
    print(splitcounts(lX[:,-1],tol=1e-16))
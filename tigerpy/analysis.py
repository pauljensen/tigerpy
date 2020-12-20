

import gurobi as gp
from gurobi import GRB

import numpy as np

from . import utils


def add_objective_constraint_(model, frac=1.0, absolute=None, solved=False):
    if frac is None and absolute is None:
        raise ValueError('Either `frac` or `absolute` required.')
    
    if absolute is not None:
        cutoff = absolute
    else:
        if not solved:
            model.optimize()
        cutoff = frac * model.ObjVal
    
    if model.ModelSense == GRB.MAXIMIZE:
        constr = model.addConstr(model.getObjective() >= cutoff)
    else:
        constr = model.addConstr(model.getObjective() <= cutoff)
    model.update()
    return constr


def get_obj(model):
    return model.getObjective(), model.ModelSense


def set_obj_(model, obj):
    model.setObjective(obj[0], obj[1])
    

def pFBA(model, x, norm="l1", solved=False):
    L1 = norm.lower() == "l1"
    L2 = norm.lower() == "l2"
    
    if not solved:
        model.optimize()
    objval = model.ObjVal
        
    obj_constr = add_objective_constraint_(model, solved=True)
    orig_obj = get_obj(model)
    
    if L1:
        # WARNING: this assumes x >= 0
        model.setObjective(x.sum(), GRB.MINIMIZE)
    elif L2:
        model.setObjective(x @ x, GRB.MINIMIZE)
    
    model.optimize()
    X = np.copy(x.X)
    
    set_obj_(model, orig_obj)
    model.remove(obj_constr)
    model.update()
    
    return X, objval


def pFBA_sensitivity(model, x, delta=0.1, norm="l1", obj_tol=1e-6):
    center_X, center_obj = pFBA(model, x, norm)
    
    ub = np.copy(x.ub)
    
    x.ub = (1+delta) * ub
    upper_X, upper_obj = pFBA(model, x, norm)
    
    if center_obj > obj_tol:
        x.ub = (1-delta) * ub
        lower_X, lower_obj = pFBA(model, x, norm)
    else:
        lower_X = np.zeros(upper_X.shape)
        lower_obj = center_obj
    
    x.ub = ub
    
    grad = np.zeros(center_X.shape)
    if abs(upper_obj - center_obj) > obj_tol:
        grad = (upper_X - center_X) / (upper_obj - center_obj)
        
    if abs(center_obj - lower_obj) > obj_tol:
        lower = (center_X - lower_X) / (center_obj - lower_obj)
        grad = np.where(np.abs(grad) > np.abs(lower), grad, lower)
        
    nonzero = np.abs(grad) > obj_tol
    grad[nonzero] = 1 / grad[nonzero]
        
    return grad, [lower_obj, center_obj, upper_obj]
    
    

# ================ Previous functions for TIGER models ================

def get_bounds(x):
    return np.copy(x.lb), np.copy(x.ub)

def reset_bounds(x, old_bounds):
    x.lb = np.copy(old_bounds[0])
    x.ub = np.copy(old_bounds[1])


def make_base_model(tiger):
    model = gp.Model()
    n = tiger.S.shape[1]
    v = model.addMVar(shape=n, lb=tiger.lb, ub=tiger.ub, name="v")
    model.setObjective(tiger.c @ v, GRB.MAXIMIZE)
    model.addConstr(tiger.S @ v == tiger.b, name="Sv=b")
    model.update()
    return model, v


def add_objective_constraint(model, frac=1.0):
    model.optimize()
    objval = model.ObjVal
    print(model.ModelSense)
    if model.ModelSense == GRB.MAXIMIZE:
        model.addConstr(model.getObjective() >= frac*objval)
    else:
        model.addConstr(model.getObjective() <= frac*objval)
    model.update()


def fba(tiger):
    m, v = make_base_model(tiger)
    m.optimize()
    return m, v


def fva(tiger, fraction=None):
    m, v = make_base_model(tiger)
    m.Params.OutputFlag = 0

    # TODO: add fraction support

    n = v.shape[0]
    minflux = np.ndarray(n)
    maxflux = np.ndarray(n)

    for i in range(n):
        m.setObjective(v[i] + 0, GRB.MAXIMIZE)
        m.optimize()
        maxflux[i] = v[i].X

    for i in range(n):
        m.setObjective(v[i] + 0, GRB.MINIMIZE)
        m.optimize()
        minflux[i] = v[i].X

    return minflux, maxflux


def single_gene_ko(tiger):
    genes = utils.unique(utils.flatten([x.atoms() for x in tiger.gprs]))
    n = len(genes)

    m, v = make_base_model(tiger)
    m.Params.OutputFlag = 0
    m.optimize()
    wt_rate = m.ObjVal
    old = get_bounds(v)

    ko_rate = np.ndarray(n)
    for i in range(n):
        to_remove = np.logical_not(tiger.eval_gprs({genes[i]: False}))
        v[to_remove].lb = 0
        v[to_remove].ub = 0
        m.optimize()
        ko_rate[i] = m.ObjVal
        reset_bounds(v, old)

    return ko_rate / wt_rate


def simple_flux_coupling(tiger, idxs=None, fixed_tol=1e-5):
    isfixed = lambda x,y: np.all(np.isclose(x, y, atol=fixed_tol))

    m, v = make_base_model(tiger)
    m.Params.OutputFlag = 0
    if idxs is not None:
        v = v[idxs]
    n = v.shape[0]

    coupled = np.zeros((n,n), dtype=bool)
    active = np.ones((n,), dtype=bool)

    old = get_bounds(v)
    lps = 0

    for i in range(n):
        if not active[i]:
            continue

        coupled[i,i] = True
        m.setObjective(1.0*v[i], sense=GRB.MAXIMIZE)
        m.optimize()

        v[i].lb = v[i].X
        v[i].ub = v[i].X

        for j in range(i+1, n):
            if not active[j]:
                continue

            m.setObjective(1.0*v[j], sense=GRB.MAXIMIZE)
            m.optimize()
            lps += 1
            vmax = v[j].X

            m.setObjective(1.0*v[j], sense=GRB.MINIMIZE)
            m.optimize()
            lps += 1
            vmin = v[j].X

            if isfixed(vmax, vmin):
                coupled[i,j] = True
                active[j] = False

        reset_bounds(v, old)
        active[i] = False
        print("Var = ", i, "LPs = ", lps)

    return coupled



def flux_coupling(tiger, idxs=None, fixed_tol=1e-5):
    isfixed = lambda x,y: np.all(np.isclose(x, y, atol=fixed_tol))

    m, v = make_base_model(tiger)
    m.Params.OutputFlag = 0
    if idxs is not None:
        v = v[idxs]
    n = v.shape[0]

    coupled = np.zeros((n,n), dtype=bool)
    active = np.ones((n,), dtype=bool)

    global_max = np.full((n,), -np.inf)
    global_min = np.full((n,), np.inf)

    #blocked = np.zeros(n, dtype=bool)
    #blocked_known = np.zeros(n, dtype=bool)

    old = get_bounds(v)
    lps = 0

    for i in range(n):
        if not active[i]:
            continue
        local_max = np.full((n,), -np.inf)
        local_min = np.full((n,), np.inf)

        coupled[i,i] = True
        m.setObjective(1.0*v[i], sense=GRB.MAXIMIZE)
        m.optimize()

        v[i].lb = v[i].X
        v[i].ub = v[i].X
        local_max = np.maximum(local_max, v.X)
        local_min = np.minimum(local_min, v.X)
        global_max = np.maximum(global_max, v.X)
        global_min = np.minimum(global_min, v.X)

        for j in range(i+1, n):
            if not active[j]:
                continue
            if not isfixed(local_max[j], local_min[j]):
                continue
            dist_ub = np.abs(v.ub[j] - local_max[j])
            dist_lb = np.abs(v.lb[j] - local_min[j])
            if dist_ub > dist_lb:
                sense = GRB.MAXIMIZE
            else:
                sense = GRB.MINIMIZE
            m.setObjective(1.0*v[j], sense=sense)
            m.optimize()
            lps += 1
            local_max = np.maximum(local_max, v.X)
            local_min = np.minimum(local_min, v.X)
            global_max = np.maximum(global_max, v.X)
            global_min = np.minimum(global_min, v.X)

            if not isfixed(local_max[j], local_min[j]):
                continue
            m.setObjective(1.0*v[j], sense=-sense)
            m.optimize()
            lps += 1
            local_max = np.maximum(local_max, v.X)
            local_min = np.minimum(local_min, v.X)
            global_max = np.maximum(global_max, v.X)
            global_min = np.minimum(global_min, v.X)

            if isfixed(local_max[j], local_min[j]):
                coupled[i,j] = True
                active[j] = False

        reset_bounds(v, old)
        active[i] = False
        print("Var = ", i, "LPs = ", lps)

    blocked = np.abs(global_max - global_min) <= fixed_tol
    coupled[:,blocked] = False
    print(sum(blocked))
    return coupled

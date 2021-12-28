import os
import copy
import numpy as np
import gurobipy as gp

from gurobipy import GRB
from typing import Tuple, List

from single_integrator_settings import *

###############################################################################
################################## Constants ##################################
###############################################################################
global MODEL, DVARS, TT0
MODEL    = None
DVARS    = None
TT0      = None

###############################################################################
################################## Functions ##################################
###############################################################################

def solve_qp(data):
    """ Solves the cascaded set of QPs in order to compute control input.

    INPUTS
    ------
    data: (dict) - contains time (t), state (x), unknown parameter estimate
                   vector (th), maximum allowable parameter estimation error
                   vector (err), and nth order eta terms for CBF derivative

    OUTPUTS
    -------
    sol: (np.ndarray) - solution to cascaded quadratic program

    """
    global MODEL

    # Update model
    MODEL,feedback = update_model(MODEL,data)






    # Update Force Model with nominal input
    F_MODEL,feedback = update_model(F_MODEL,'F',F_nom,data)

    # Compute safe F control input
    try:
        F             = compute_safe_force()
        data['F_sol'] = F
    except Exception as e:
        print(feedback)
        # raise e

    # Update Torque Model with nominal input
    TAU_MODEL,feedback = update_model(TAU_MODEL,'tau',tau_nom,data)

    # Compute safe tau1, tau2, tau3 control inputs
    try:
        tau1,tau2,tau3 = compute_safe_torques()
    except Exception as e:
        print(feedback)
        # raise e

    return np.array([F,tau1,tau2,tau3]),np.array([F_nom,tau_nom[0],tau_nom[1],tau_nom[2]])

def update_model(model:   gp.Model,
                 version: str,
                 u_nom:   float or np.ndarray,
                 data:    dict) -> gp.Model:
    """ Updates the specified QP model with the computed nominal control input.

    INPUTS
    ------
    model:   (gp.model) - QP model
    data:    (dict) - data for updating QP constraints

    OUTPUTS
    -------
    model: (gp.model) - updated model
    """
    global DVARS
    if model is None:
        # Create the model
        model = gp.Model("qp")
        model.setParam('OutputFlag',0)
        d_vars = DECISION_VARS
        u_nom  = np.array([0.0,0.0]) # Minimum norm controller

        # Create the decision variables
        decision_variables = np.zeros((len(d_vars),),dtype='object')
        decision_vars_copy = copy.deepcopy(d_vars)
        for i,v in enumerate(decision_vars_copy):
            lb   = v['lb']
            ub   = v['ub']
            name = v['name']

            decision_variables[i] = model.addVar(lb=lb,ub=ub,name=name)

        # Assign decision variables to global variable
        DVARS = decision_variables

    # Modify objective function according to new nominal control input
    model.setObjective(objective(DVARS,u_nom),GRB.MINIMIZE)
    model,feedback,e = update_constraints(model,data,DVARS)
    model.update()

    return model,feedback

############################# Safe Control Inputs #############################

def compute_safe_force():
    global F_MODEL

    # Compute solution to QP
    optimize_qp(F_MODEL)

    # Debug solution to QP
    sol,success,code = check_qp_sol(F_MODEL)
    if not success:
        return code

    # Extract control solutions
    F   = sol[0]
    # pf1 = sol[1]

    return F#,pf1

def compute_safe_torques():
    global TAU_MODEL

    # Compute solution to QP
    optimize_qp(TAU_MODEL)

    # Debug solution to QP
    sol,success,code = check_qp_sol(TAU_MODEL)
    if not success:
        return code

    # Extract control solutions
    tau1 = sol[0]
    tau2 = sol[1]
    tau3 = sol[2]
    # pt1  = sol[3]
    # pt2  = sol[4]
    # pt3  = sol[5]

    return tau1,tau2,tau3#,pt1,pt2,pt3

def update_constraints(model:  gp.Model,
                       data:   dict,
                       d_vars: np.ndarray) -> Tuple:
    """ Computes the safety-compensating force control input for the 6-DOF
    dynamic quadrotor model.

    INPUTS
    ------
    data:   (dict) - contains time (t), state (x), unknown parameter estimate
                     vector (th), maximum allowable parameter estimation error
                     vector (err), and nth order eta terms for CBF derivative
    d_vars: (np.ndarray) - decision variables

    OUTPUTS
    -------
    F:   (float) - safe force control input [0,F_MAX]
    pf1: (np.ndarray) - 'adaptive' feasibility gains from Belta paper

    """
    # Unpack data dict
    tt       = data['t']
    state    = data['x']
    thetaHat = data['th']
    eta      = data['err']
    etaTerms = data['etaTerms']
    Gamma    = data['Gamma']
    feedback = ""
    e        = None

    # Remove old constraints
    model.remove(model.getConstrs())

    # Relative-Degree 1 Safety Constraint - Exclusion Region 1
    Lfh   = cbfdot_region1(state)# - etaTerms[0]
    h     = cbf_region1(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # K-Coefficients for HO-CBF Terms
    K0 = 1.0e3
    K1 = 1.0

    # Formalized Exponential CBF Condition (with feasibility parameter on h)
    cbf1 = Lfh + Lgh @ d_vars[0:2] + K0*h*d_vars[2]

    # Add New Constraint
    model.addConstr(cbf1 >= 0)

    # Relative-Degree 1 Safety Constraint - Exclusion Region 2
    Lfh   = cbfdot_region2(state)# - etaTerms[0]
    h     = cbf_region2(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # K-Coefficients for HO-CBF Terms
    K0 = 1.0e3
    K1 = 1.0

    # Formalized Exponential CBF Condition (with feasibility parameter on h)
    cbf2 = Lfh + Lgh @ d_vars[0:2] + K0*h*d_vars[3]

    # Add New Constraint
    model.addConstr(cbf2 >= 0)

    # Update Model
    model.update()

    return model,feedback,e

###############################################################################
################################# Controller ##################################
###############################################################################

def optimize_qp(model: gp.Model) -> None:
    """ Reverts the Model settings to the defaults and then calls the
    gp.Model.optimize method to solve the optimization problem.

    INPUTS
    ------
    None

    RETURNS
    ------
    None

    """
    # Revert to default settings
    model.setParam('BarHomogeneous',-1)
    model.setParam('NumericFocus',0)

    # Solve
    model.optimize()

def check_qp_sol(model: gp.Model,
                 level: int = 0,
                 multiplier: int = 10):
    """
    Processes the status flag associated with the Model in order to perform
    error handling. If necessary, this will make adjustments to the solver
    settings and attempt to re-solve the optimization problem to obtain an
    accurate, feasible solution.

    INPUTS
    ------
    level: (int, optional) - current level of recursion in solution attempt

    RETURNS
    ------
    sol  : (np.ndarray)    - decision variables which solve the opt. prob.
    T/F  : (bool)          - pure boolean value denoting success or failure
    ERROR: (np.ndarray)    - error code for loop-breaking at higher level
    """
    # Define Error Checking Parameters
    status  = model.status
    epsilon = 0.1
    success = 2

    # Obtain solution
    sol = model.getVars()

    # Check status
    if status == success:
        sol = saturate_solution(sol)
        return sol,True,0

    else:
        model.write('diagnostic.lp')

        if status == 3:
            msg = "INFEASIBLE"
        elif status == 4:
            msg = "INFEASIBLE_OR_UNBOUNDED"
        elif status == 5:
            msg = "UNBOUNDED"
        elif status == 6:
            msg = "CUTOFF"
        elif status == 7:
            msg = "ITERATION_LIMIT"
        elif status == 8:
            msg = "NODE_LIMIT"
        elif status == 9:
            msg = "TIME_LIMIT"
        elif status == 10:
            msg = "SOLUTION_LIMIT"
        elif status == 11:
            msg = "INTERRUPTED"
        elif status == 12:
            msg = "NUMERIC"
        elif status == 13:
            msg = "SUBOPTIMAL"
        elif status == 14:
            msg = "INPROGRESS"
        elif status == 15:
            msg = "USER_OBJ_LIMIT"

        if status == 13:
            sol = saturate_solution(sol)
            return sol,True,0

        print("Solver Returned Code: {}".format(msg))

        return sol,False,ERROR

def saturate_solution(sol):
    saturated_sol = np.zeros((len(sol),))
    for ii,s in enumerate(sol):
        saturated_sol[ii] = np.min([np.max([s.x,s.lb]),s.ub])
    return saturated_sol
"""
black2020nonvanishingQP.py

Implementation of the controller proposed by:
M. Black,  K. Garg,  and  D. Panagou, “A quadratic program based control
synthesis under spatiotemporal constraints and non-vanishing disturbances,”
59th IEEE Conference on Decision and Control, 2020.

"""

__author__  = "Mitchell Black"
__email__   = "mblackjr@umich.edu"
__date__    = "November 20, 2020"
__version__ = 1.0

import os
import sys
import copy
import time
import pickle
# import winsound
import numpy as np
import gurobipy as gp

from gurobipy import GRB
from scipy import sparse
from clfcbfcontroller import ClfCbfController

# Flexible use for different problems
if sys.platform == 'darwin':
    folder_code = '/'
elif sys.platform == 'win32':
    folder_code = '\\'
else:
    folder_code = '/'

FOLDER = os.getcwd().split(folder_code)[-1]
if FOLDER == 'overtake':
    from overtake_settings import *
elif FOLDER == 'simple':
    TIME_HEADWAY       = 0
    ONCOMING_FREQUENCY = 0
    from simple_settings import *

###############################################################################
#################################### Class ####################################
###############################################################################

class BlackController(ClfCbfController):

    @property
    def safe_to_pass(self) -> bool:
        """ Evaluates whether ego vehicle can safely overtake lead vehicle
        given the bound on model uncertainty.

        INPUTS:
        -------
        None

        OUTPUTS:
        -------
        bool - True if safe, otherwise False

        """

        time_needed = 20.735 # Theta_Max = 1
        time_needed = 21.312 # Theta_Max = 2
        time_needed = 22.669 # Theta_Max = 4
        time_needed = 24.415 # Theta_Max = 6
        time_needed = 26.800 # Theta_Max = 8
        time_needed = 30.384 # Theta_Max = 10

        if time_needed < TIME_HEADWAY or self.safe:
            return True
        elif time_needed > ONCOMING_FREQUENCY:
            return False
        elif self.t > TIME_HEADWAY:
            self.setpoint = self.setpoint + 1
            self.safe = True
            print("Safe to Pass! New Goal: ",self.setpoint)
            return True

    def __init__(self,u_max):
        """ Instantiates BlackController. Sets the objective function for the
        optimization problem computing the control input, and defines the
        decision variables accordingly.

        INPUTS
        -------
        u_max: np.ndarray - contains the min/max control input constraints

        OUTPUTS
        -------
        None

        """
        # Call to parent class (ClfCbfController)
        super().__init__(u_max)

        # Create the decision variables
        self.decision_variables = np.zeros((len(DECISION_VARS),),dtype='object')
        decision_vars_copy      = copy.deepcopy(DECISION_VARS)
        for i,v in enumerate(decision_vars_copy):
            lb   = v['lb']   # lower bound
            ub   = v['ub']   # upper bound
            name = v['name'] # variable name

            # Add decision variables to Gurobi model
            self.decision_variables[i] = self.m.addVar(lb=lb,ub=ub,name=name)

        # Set model objective function
        self.m.setObjective(OBJ(self.decision_variables),GRB.MINIMIZE)

    def set_initial_conditions(self,
                               **kwargs: dict) -> None:
        """ Assigns values to the initial conditions, including min/max on the
        parametric uncertainty.

        INPUTS
        -------
        kwargs: dict - contains keyword arguments to parent method
                       _set_initial_conditions

        OUTPUTS
        -------
        None

        """
        self._set_initial_conditions(**kwargs)

        self.setpoint   = 0
        self.name       = "BLA"
        self.safe       = False

        # For data visualization
        self.dV         = None
        self.clf        = None
        self.td         = None

    def update_qp(self,
                  x: np.ndarray,
                  t: float) -> None:
        """ Configures the optimization problem (quadratic program, qp) to be
        solved at the current timestep. Updates the constraints on safety and
        performance objectives.

        INPUTS
        -------
        x: np.ndarray - state vector
        t: float - time in sec

        OUTPUTS
        -------
        None

        """
        # Remove old constraints
        if self.performance is not None:
            self.m.remove(self.performance)
        if self.safety is not None:
            for s in self.safety:
                self.m.remove(s)

        # Update Goal
        reached = False
        if self.safe_to_pass or FOLDER == 'simple':
            if clf(x,self.xd) < 0.0:
                print("Goal {} Reached! Time = {:.3f}".format(self.setpoint,self.t))
                self.setpoint = self.setpoint + 1

                # Beep to notify user of progress
                frequency = 1500  # Set Frequency To 2500 Hertz
                duration  = 100   # Set Duration To 1000 ms == 1 second
                # winsound.Beep(frequency, duration)
                # time.sleep(0.1)
                # winsound.Beep(frequency, duration)

        # Assign c1, c2 for FxTS conditions
        try:
            c1   = c1_c[self.setpoint]
            c2   = c2_c[self.setpoint]
        except TypeError:
            c1   = c1_c
            c2   = c1_c

        # Update CLF and CBF
        self.V   = V   = np.max([clf(x,self.xd),0]) # Max for when unsafe to pass
        self.B   = B   = cbf(x,t)

        # Update Partial Derivatives of CLF and CBF
        self.dV  = dV  = clf_partial(x,self.xd)
        self.dVd = dVd = clf_partiald(x,self.xd)
        self.dBx = dBx = cbf_partialx(x,t)
        self.dBt = dBt = cbf_partialt(x,t)

        # Evaluate Lie Derivatives
        phi_max              = self.get_phi_max()
        LfV1                 = np.dot(dV,f(x))
        self.LgV    = LgV    = np.dot(dV,g(x))
        self.LdVmax = LdVmax = self.max_LdV(phi_max,dV)
        self.LfB    = LfB    = np.dot(dBx,f(x))
        self.LgB    = LgB    = np.dot(dBx,g(x))
        self.LdBmin = LdBmin = self.min_LdB(phi_max,dBx)

        # Lie Derivative wrt Time-Varying Goal
        LfVd           = np.dot(dVd,fd(x,self.xd))
        self.LfV = LfV = LfV1 + LfVd

        # Configure CLF and CBF constraints
        try:

            # CLF (FxTS) Conditions: LfV + LgV*u + LdV*theta <= -c1c*Vc^gamma1c - c2c*Vc^gamma2c + delta0
            if np.sum(LgV.shape) > 1:
                p = self.m.addLConstr(LfV + np.sum(np.array([gg*self.decision_variables[i] for i,gg in enumerate(LgV)]))
                                      <=
                                      - c1*V**(1 - 1/mu_c) - c2*V**(1 + 1/mu_c) - LdVmax + self.decision_variables[2]
                )
                self.performance = p
            else:
                p = self.m.addLConstr(LfV + LgV*self.decision_variables[0]
                                      <=
                                      - c1*V**(1 - 1/mu_c) - c2*V**(1 + 1/mu_c) - LdVmax + self.decision_variables[1]
                )
                self.performance = p

            # CBF (Safety) Conditions: LfrB + LgrB*u + LdB*theta + >= -delta1(rB)
            self.safety  = []
            for ii,(ff,gg,dd,bb) in enumerate(zip(LfB,LgB,LdBmin,B)):

                # Multiple control inputs
                if np.sum(LgV.shape) > 1:
                    s = self.m.addLConstr(ff + np.sum(np.array([g*self.decision_variables[i] for i,g in enumerate(gg)])) + dd
                                          >=
                                          -self.decision_variables[3+ii]*bb**POWER
                    )
                    self.safety.append(s)

                # Single control input
                else:
                    s = self.m.addLConstr(ff + gg*self.decision_variables[0] + dd
                                          >=
                                          -self.decision_variables[2+ii]*bb**POWER
                    )
                    self.safety.append(s)

        except:
            self.report_msg()
            raise ValueError("GurobiError")

    def compute(self,
                not_skip: bool = True) -> np.ndarray:
        """ Computes the solution to the Quadratic Program by first updating
        the unknown parameter estimates and then by calling the parent
        Controller._compute method.

        INPUTS
        ------
        not_skip: (bool) - True -> update parameters, False -> do not update

        RETURNS
        ------
        return parameter for Controller._compute()

        """
        # Compute Solution
        self._compute()

        return self.sol

    def max_LdV(self,
                pmax: np.ndarray,
                dV:   np.ndarray) -> float:
        """ Computes maximum of (dV/dx)(Delta)(theta)

        INPUTS
        -------
        pmax: array (n,p) - zeros, will be populated within this function
        dV:   array (1,n) - partial derivative of V wrt x

        OUTPUTS
        -------
        m: float - maximum of LdV

        """

        # Determine maximum of LdV*(theta_hat + theta_err)
        for i,v in enumerate(dV):
            pmax[i]  = -pmax[i]*(v*-pmax[i] > v*pmax[i]) + pmax[i]*(v*-pmax[i] <= v*pmax[i])

        # Column-wise maximum
        m = np.dot(dV,pmax)
        return m


    def min_LdB(self,
                pmax: np.ndarray,
                dBx:  np.ndarray) -> np.ndarray:
        """ Computes supremum of (dV/dx)(Delta)(theta)

        INPUTS
        -------
        pmax: array (n,p) - max/min allowable values of theta
        dV:   array (m,n) - partial derivative of m CBFs wrt x

        OUTPUTS
        -------
        m: array (m,1) - minimum of LdB

        """

        # Initialize phi array
        if np.sum(dBx.shape) > dBx.shape[0]:
            phi_rB = np.zeros((dBx.shape[0],pmax.shape[0]))
        else:
            phi_rB = np.zeros(pmax.shape)

        # Determine minimum of LdB*(theta_hat + theta_err) for each CBF
        for i,b in enumerate(dBx):
            if np.sum(b.shape) > 0:
                for j,bb in enumerate(b):
                    phi_rB[i,j] = -pmax[j]*(bb*pmax[j] > bb*-pmax[j]) + pmax[j]*(bb*pmax[j] <= bb*-pmax[j])
            else:
                phi_rB[i] = -pmax[i]*(b*pmax[i] > b*pmax[i]) + pmax[i]*(b*pmax[i] <= b*-pmax[i])

        # Column-wise minimum
        m = np.einsum('ij,ij->i',dBx,phi_rB)
        return m

    def report_msg(self) -> None:
        """ Prints out important state and controller information.

        INPUTS:
        None

        OUTPUTS:
        None

        """
        print("X:   {}".format(self.x))
        print("LfV: {}".format(self.LfV))
        print("LgV: {}".format(self.LgV))
        print("LdV: {}".format(self.LdVmax))
        print("V:   {}".format(self.V))

    def get_phi_max(self) -> np.ndarray:
        """ Obtains the worst-case effect of uncertainty.

        INPUTS:
        None

        OUTPUTS:
        phi_max: array (1,p): worst-case disturbance action

        """

        theta_max  = self.theta_max[0]

        if FOLDER == 'overtake':
            scale1     = regressor(self.x,0)[0,0]
            scale2     = regressor(self.x,0)[1,1]
            phi_max    = np.array([0,0,0,0,scale1*theta_max, scale2*theta_max,0,0,0,0,0,0])
        elif FOLDER == 'simple':
            phi_max = regressor(self.x,0) @ (theta_max * np.ones((self.x.shape[0],)))

        return phi_max

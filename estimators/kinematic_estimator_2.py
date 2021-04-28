import os
import copy
import time
import numpy as np

from estimator import *

###############################################################################
##################### FxT Parameter Estimation Parameters #####################
###############################################################################

mu_e      = 2
c1_e      = 4
c2_e      = 4
k_e       = 0.0002
l_e       = 25.0
gamma1_e  = 1 - 1/mu_e
gamma2_e  = 1 + 1/mu_e
T_e       = 1 / (c1_e * (1 - gamma1_e)) + 1 / (c2_e * (gamma2_e - 1))
kG        = 1.1 # Must be greater than 1
Kw        = 165
Kw        = 2.0


###############################################################################
################################## Functions ##################################
###############################################################################

def adaptation_law(dt,tt,x,u):
    global ESTIMATOR
    if ESTIMATOR is None:
        settings = {'dt':dt,'x':x,'tt':tt}
        ESTIMATOR = KinematicEstimator()
        ESTIMATOR.set_initial_conditions(**settings)

    return ESTIMATOR.update(tt,x,u)

###############################################################################
#################################### Class ####################################
###############################################################################

class KinematicEstimator(Estimator):

    @property
    def T_e(self):
        arg1 = np.sqrt(c2_e) * self.V0_T**(1/mu_e)
        arg2 = np.sqrt(c1_e)
        return mu_e / np.sqrt(c1_e*c2_e) * np.arctan2(arg1,arg2)

    @property
    def e(self):
        """ State Prediction Error """
        return (self.x - self.xHat)[0:3]

    def __init__(self):
        super().__init__()
    """
    def set_initial_conditions(self,**settings):
        self._set_initial_conditions(settings)
    """
    def update(self,
               t: float,
               x: np.ndarray,
               u: np.ndarray):
        """ Updates the parameter estimates and the corresponding error bounds. """
        return self._update(t,x,u,law=1,mode=1)
    """
        self.t  = t
        self.x = x
        self.u  = u

        if t == 0:
            return self.thetaHat,self.errMax,self.etaTerms,self.Gamma,self.xHat,self.theta,self.minSigma

        # Update Unknown Parameter Estimates
        self.update_unknown_parameter_estimates(law=2)

        # Update state estimate using observer dynamics
        self.update_observer()

        # Update theta_tilde upper/lower bounds
        self.update_error_bounds(self.t)

        return self.thetaHat,self.errMax,self.etaTerms,self.Gamma,self.xHat,self.theta,self.minSigma
    """
    """
    def update_unknown_parameter_estimates(self,
                                            law:  int = 1,
                                            mode: int = 1):
        """
        """
        tol = 0#1e-15
        # self.update_filter_2ndOrder()

        eig = self.update_auxiliaries(mode=mode)
        if eig <= 0:
            # return
            raise ValueError('PE Condition Not Met: Eig(P) = {:.3f}'.format(eig))

        if mode == 1:
            vec = self.e
            Mat = self.W
        elif mode == 2:
            Mat = -self.P
            vec = self.P @ self.thetaHat - self.Q

        # Minimum singular value
        u,s,v = np.linalg.svd(Mat)
        self.minSigma = np.min(s)

        if law == 0:
            MatInv = np.linalg.inv(-Mat)
            
            # Adaptation Law 1
            pre  = self.Gamma @ vec / (vec.T @ MatInv.T @ vec)
            V    = (1/2 * vec.T @ MatInv.T @ np.linalg.inv(self.Gamma) @ MatInv @ vec)
            self.thetaHatDot = pre * (-c1_e * V**gamma1_e - c2_e * V**gamma2_e)

        elif law == 1:
            if np.linalg.norm(vec) < 1e-15:
                self.thetaHatDot = 0.0 * self.thetaHat
                return

            # Adaptation Law 2
            pre  = -self.Gamma / (np.linalg.norm(vec))
            prepreXi = self.Gamma @ regressor(self.x).T @ vec
            self.thetaHatDot = prepreXi + pre @ (Mat.T @ vec + Mat.T @ vec * (vec.T @ vec))

            norm_chk = np.linalg.norm(vec)
            if np.isnan(norm_chk) or np.isinf(norm_chk) or norm_chk == 0:
                print("ThetaHat: {}".format(self.thetaHat))
                raise ValueError("Vec Norm Zero")

        thd_norm      = np.linalg.norm(self.thetaHatDot)
        if thd_norm >= tol and not (np.isnan(thd_norm) or np.isinf(thd_norm)):
            self.thetaHat = self.thetaHat + (self.dt * self.thetaHatDot)
            #self.bigTheta = self.bigTheta + (self.dt * self.thetaHat)
            #self.bigTheta = np.clip(self.bigTheta,self.thetaMin,self.thetaMax)

        else:
            print("P: {}".format(self.P))
            print("Q: {}".format(self.Q))
            print("W: {}".format(W))
            print("No Theta updated")
            print("Pre  = {}\nTime = {}sec".format(pre,self.t))

        # self.theta     = Pinv @ self.Q
    """
    def update_observer(self):
        """ Updates the state estimate according to the observer (xhat) dynamics.
        Overwrites the parent method.

        INPUTS
        ------
        None

        OUTPUTS
        -------
        None

        """
        xHatDot = f(self.x)[0:3] + regressor(self.x)@self.thetaHat + Kw*self.e + self.W@self.thetaHatDot
        self.xHat[0:3] = self.xHat[0:3] + (self.dt * xHatDot)

    """
    def update_error_bounds(self,t):
        """
        """
        # Update Max Error Quantities
        arc_tan      = np.arctan2(np.sqrt(c2_e) * self.V0**(1/mu_e),np.sqrt(c1_e))
        tan_arg      = -np.min([t,self.T_e]) * np.sqrt(c1_e * c2_e) / mu_e + arc_tan
        Vmax         = (np.sqrt(c1_e / c2_e) * np.tan(np.max([tan_arg,0]))) ** mu_e
        self.Vmax    = np.clip(Vmax,0,np.inf)

        # Update eta
        self.eta     = self.Vmax

        # Define constants
        M = 2 * np.max(self.Gamma)
        N = np.sqrt(c2_e/c1_e)
        X = np.arctan2(N * self.V0,1)
        c = c1_e
        u = mu_e
        A = N*c/u
        B = np.sqrt(M * 1 / N**u)
        tan_arg = np.min([A*t - X,0.0])

        # Update eta derivatives
        etadot  = -A*B*(np.tan(tan_arg)**2 + 1)
        eta2dot = -2*A**2*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)
        eta3dot = -2*A**3*B*(np.tan(tan_arg)**2 + 1)**2 - 4*A**3*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)**2
        eta4dot = -16*A**4*B*(np.tan(tan_arg)**2 + 1)**2*np.tan(tan_arg) - 8*A**4*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)**3

        # Update etadot terms
        self.eta_order1 = np.trace(np.linalg.inv(self.Gamma)) * (self.eta * etadot)
        self.eta_order2 = np.trace(np.linalg.inv(self.Gamma)) * (self.eta * eta2dot + etadot**2)
        self.eta_order3 = np.trace(np.linalg.inv(self.Gamma)) * (self.eta * eta3dot + 3 * etadot * eta2dot)
        self.eta_order4 = np.trace(np.linalg.inv(self.Gamma)) * (self.eta * eta4dot + 4 * etadot * eta3dot + 3 * eta2dot**2)

        if self.t < T_e:
            self.etaTerms = np.array([self.eta_order1,self.eta_order2,self.eta_order3,self.eta_order4])
        else:
            self.etaTerms = np.zeros((4,))

        # Update max theta_tilde
        self.errMax = np.clip(np.sqrt(2*np.diagonal(self.Gamma)*self.Vmax),0,self.thetaMax-self.thetaMin)
    """

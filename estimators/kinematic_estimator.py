import os
import copy
import time
import numpy as np

from estimator import *

import builtins
if hasattr(builtins,"ecc_MODEL_CONFIG"):
    FOLDER = builtins.ecc_MODEL_CONFIG
else:
    FOLDER = 'simple'

if FOLDER == 'overtake':
    from overtake_settings import *
elif FOLDER == 'simple':
    from simple_settings import *
elif FOLDER == 'simple2ndorder':
    from simple2ndorder_settings import *
elif FOLDER == 'quadrotor':
    from quadrotor_settings import *

    def f_kin(x):
        return f(x)[0:3]

###############################################################################
##################### FxT Parameter Estimation Parameters #####################
###############################################################################

global ESTIMATOR
ESTIMATOR = None


mu_e      = 5
c1_e      = 50
c2_e      = 50
k_e       = 0.001
l_e       = 100
gamma1_e  = 1 - 1/mu_e
gamma2_e  = 1 + 1/mu_e
T_e       = 1 / (c1_e * (1 - gamma1_e)) + 1 / (c2_e * (gamma2_e - 1))
kG        = 1.2 # Must be greater than 1

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
# kW        = 500
# Kw        = 10.0
Kw        = 2.0

###############################################################################
################################## Functions ##################################
###############################################################################

def adaptation_law(dt,tt,x,u):
    global ESTIMATOR
    if ESTIMATOR is None:
        settings = {'dt':dt,'x':x,'tt':tt}
        ESTIMATOR = Estimator()
        ESTIMATOR.set_initial_conditions(**settings)

    return ESTIMATOR.update(tt,x,u)

###############################################################################
#################################### Class ####################################
###############################################################################

class Estimator():

    @property
    def T_e(self):
        arg1 = np.sqrt(c2_e) * self.V0_T**(1/mu_e)
        arg2 = np.sqrt(c1_e)
        return mu_e / np.sqrt(c1_e*c2_e) * np.arctan2(arg1,arg2)

    @property
    def e(self):
        """ State Prediction Error """
        return self.x - self.xHat

    @property
    def x(self):
        return self._x[0:3]

    def __init__(self):
        self.dt       = None
        self._x       = None
        self.xHat     = None
        self.thetaHat = None
        self.theta    = None
        self.thetaMax = None
        self.thetaMin = None
        self.minSigma = 0

    def set_initial_conditions(self,**settings):
        """ """
        # Set Default Initial Conditions
        self.dt       = dt
        self._x       = x0
        self.xHat     = self.x
        self.thetaHat = thetaHat
        self.thetaMax = thetaMax
        self.thetaMin = thetaMin

        if 'x0' in settings.keys():
            assert type(settings['x0']) == np.ndarray
            self._x   = settings['x0']
            self.xHat = self.x

        if 'x' in settings.keys():
            assert type(settings['x']) == np.ndarray
            self._x   = settings['x']
            self.xHat = self.x

        if 'thetaHat0' in settings.keys():
            assert type(settings['thetaHat0']) == np.ndarray
            self.thetaHat = settings['thetaHat0']
            self.theta    = settings['thetaHat0']
            if 'psi_hat0' not in settings.keys() and 'psi_est0' not in settings.keys():
                self.psi_hat = settings['thetaHat0']

        if 'th' in settings.keys():
            assert type(settings['thetaHat0']) == np.ndarray
            self.thetaHat = settings['th']
            self.theta    = settings['th']
            if 'psi_hat0' not in settings.keys() and 'psi_est0' not in settings.keys():
                self.psi_hat = settings['th']

        if 'theta_est0' in settings.keys():
            assert type(settings['thetaHat0']) == np.ndarray
            self.thetaHat = settings['theta_est0']
            self.theta    = settings['theta_est0']
            if 'psi_hat0' not in settings.keys() and 'psi_est0' not in settings.keys():
                self.psi_hat = settings['theta_est0']

        if 'psi_hat0' in settings.keys():
            assert type(settings['psi_hat0']) == np.ndarray
            self.psi_hat = settings['psi_hat0']

        if 'psi_est0' in settings.keys():
            assert type(settings['psi_est0']) == np.ndarray
            self.psi_hat = settings['psi_est0']

        if 'thetaMax' in settings.keys():
            assert type(settings['thetaMax']) == np.ndarray
            self.thetaMax = settings['thetaMax']
            if 'psi_max' not in settings.keys():
                self.psi_max = settings['thetaMax']
            if 'thetaMin' not in settings.keys():
                self.thetaMin = -1*self.thetaMax

        if 'thetaMin' in settings.keys():
            assert type(settings['thetaMin']) == np.ndarray
            self.thetaMin = settings['thetaMin']
            if 'psi_min' not in settings.keys():
                self.psi_min = settings['thetaMin']

        if 'psi_max' in settings.keys():
            assert type(settings['psi_max']) == np.ndarray
            self.psi_max = settings['psi_max']
            if 'thetaMax' not in settings.keys():
                self.thetaMax = settings['psi_max']

        if 'psi_min' in settings.keys():
            assert type(settings['psi_min']) == np.ndarray
            self.psi_min = settings['psi_min']
            if 'thetaMin' not in settings.keys():
                self.thetaMin = settings['psi_min']

        if 'dt' in settings.keys():
            assert type(settings['dt']) == float
            self.dt = settings['dt']

        if 'tt' in settings.keys():
            assert type(settings['tt']) == float or type(settings['tt']) == np.float64
            self.t = settings['tt']

        # This is more conservative than necessary -- could be fcn of initial estimate
        self.errMax    = self.thetaMax - self.thetaMin
        c              = np.linalg.norm(self.errMax)
        self.Gamma     = kG * c**2 / (2 * np.min(cbf(self._x))) * np.eye(self.thetaHat.shape[0])
        self.V0_T      = np.inf
        self.V0        = 1/2 * self.errMax.T @ np.linalg.inv(self.Gamma) @ self.errMax
        self.Vmax      = self.V0
        self.eta       = self.Vmax

        self.W         = np.zeros(regressor(self._x).shape)
        self.xi        = self.e

        self.update_error_bounds(0)

    def update(self,
               t: float,
               x: np.ndarray,
               u: np.ndarray):
        """ Updates the parameter estimates and the corresponding error bounds. """
        self.t  = t
        self._x = x
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

    def update_unknown_parameter_estimates(self,
                                            law    = 1,
                                            scheme = 1):
        """
        """
        tol = 0#1e-15
        # self.update_filter_2ndOrder()

        eig = self.update_auxiliaries()
        if eig <= 0:
            # return
            raise ValueError('PE Condition Not Met: Eig(P) = {:.3f}'.format(eig))

        if scheme == 1:
            vec = self.e
            Mat = self.W
        else:
            Mat = -self.P
            vec = self.P @ self.thetaHat - self.Q

        # Minimum singular value
        u,s,v = np.linalg.svd(Mat)
        self.minSigma = np.min(s)

        if law == 1:
            MatInv = np.linalg.inv(-Mat)
            # Adaptation Law 1
            pre  = self.Gamma @ vec / (vec.T @ MatInv.T @ vec)
            V    = (1/2 * vec.T @ MatInv.T @ np.linalg.inv(self.Gamma) @ MatInv @ vec)
            self.thetaHatDot = pre * (-c1_e * V**gamma1_e - c2_e * V**gamma2_e)

        elif law == 2:
            if np.linalg.norm(vec) < 1e-15:
                self.thetaHatDot = 0.0 * self.thetaHat
                return

            # Adaptation Law 2
            pre  = -self.Gamma / (np.linalg.norm(vec))
            prepreXi = self.Gamma @ regressor(self._x).T @ vec
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

    def update_auxiliaries(self):
        """ Updates the auxiliary matrix and vector for the filtering scheme.

        INPUTS:
        None

        OUTPUTS:
        float -- minimum eigenvalue of P matrix

        """
        Wdot   = -Kw * self.W + regressor(self._x)
        # xi_dot = -Kw * self.xi

        self.W  = self.W  + (self.dt * Wdot)
        # self.xi = self.xi + (self.dt * xi_dot)

        # Pdot = -l_e * self.P + np.dot(self.W.T,self.W)
        # Qdot = -l_e * self.Q + np.dot(self.W.T,(self.W@self.thetaHat + self.e - self.xi))

        # Pdot = np.dot(self.W.T,self.W)
        # Qdot = np.dot(self.W.T,(self.W@self.thetaHat + self.e - self.xi))

        # self.P = self.P + (self.dt * Pdot)
        # self.Q = self.Q + (self.dt * Qdot)

        # norm_chk = np.linalg.norm(self.P)
        # if np.isnan(norm_chk) or np.isinf(norm_chk):
        #     raise ValueError("P Norm out of bounds")

        # norm_chk = np.linalg.norm(self.Q)
        # if np.isnan(norm_chk) or np.isinf(norm_chk):
        #     raise ValueError("Q Norm out of bounds")

        return 1#np.min(np.linalg.eig(self.P)[0])

    def update_observer(self):
        """ Updates the state estimate according to the observer (xhat) dynamics.

        INPUTS
        ------
        None

        OUTPUTS
        -------
        None

        """
        # xHatDot = f(self.x) + g(self.x)@self.u + regressor(self.x)@self.thetaHat + Kw*self.e + self.W@self.thetaHatDot
        xHatDot = f_kin(self._x) + regressor(self._x)@self.thetaHat + Kw*self.e + self.W@self.thetaHatDot
        # print("ThetaHatDot: {}".format(self.thetaHatDot))
        # print("xHatDot: {}".format(xHatDot))
        self.xHat = self.xHat + (self.dt * xHatDot)

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


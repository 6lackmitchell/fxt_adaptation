import os
import copy
import time
import numpy as np

from icecream import ic

# from quadrotor_settings import *

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

###############################################################################
##################### FxT Parameter Estimation Parameters #####################
###############################################################################

global ESTIMATOR
ESTIMATOR = None

modes = {'estimator':1,
         'filter':   2,
         'rate':     3}

mu_e      = 2
c1_e      = 4
c2_e      = 4
pp        = 0.25
pp        = 0.5
qq        = 1.0
k_e       = 0.0002
k_e       = 0.001
l_e       = 25.0
l_e       = 500.0
l_e       = 10000.0
l_e       = 1000.0

gamma1_e  = 1 - 1/mu_e
gamma2_e  = 1 + 1/mu_e
T_e       = 1 / (c1_e * (1 - gamma1_e)) + 1 / (c2_e * (gamma2_e - 1))
kGs       = 1.1 # Must be greater than 1, Used for Rate and Estimator
kGe       = 2.0 # Must be greater than 0
#kG        = 2.0 # Used for Filtering
Kw        = 400
Kw        = 1

Kw        = 200
pp        = 0.5
qq        = 10.0
kGe       = 0.25 # Must be greater than 0
kGe       = 0.1  # Must be greater than 0
pp        = 0.5
qq        = 1.0

##########################
# FINAL ESTIMATOR VALUES
Kw        = 0.5
Kw        = 0.0
kGe       = 0.001 # Must be greater than 0
pp        = 0.1
qq        = 1.0
rr        = 15.0
mu        = 1.5
##########################

##########################
# TESTING RATE VALUES
kGe       = 0.0001 # Must be greater than 0
pp        = 0.1
qq        = 1.0
rr        = 70 / np.sqrt(2 * (13900 * kGe))
ic(rr)
mu        = 1.5
##########################

###############################################################################
################################## Functions ##################################
###############################################################################

def GammaS(dt:   float,
           tt:   float,
           x:    np.ndarray,
           u:    np.ndarray,
           law:  int = 1,
           mode: int = 1):
    global ESTIMATOR
    if ESTIMATOR is None:
        settings = {'dt':dt,'x':x,'tt':tt}
        ESTIMATOR = Estimator()
        ESTIMATOR.set_initial_conditions(**settings)

    return ESTIMATOR.GammaS

def GammaE(dt:   float,
           tt:   float,
           x:    np.ndarray,
           u:    np.ndarray,
           law:  int = 1,
           mode: int = 1):
    global ESTIMATOR
    if ESTIMATOR is None:
        settings = {'dt':dt,'x':x,'tt':tt}
        ESTIMATOR = Estimator()
        ESTIMATOR.set_initial_conditions(**settings)

    return ESTIMATOR.GammaE

def adaptation_law(dt:   float,
                   tt:   float,
                   x:    np.ndarray,
                   u:    np.ndarray,
                   law:  int = 1,
                   mode: int = 1,
                   time_varying: bool = False):
    global ESTIMATOR
    if ESTIMATOR is None:
        settings = {'dt':dt,'x':x,'tt':tt}
        ESTIMATOR = Estimator()
        ESTIMATOR.set_initial_conditions(**settings)

    return ESTIMATOR.update(tt,x,u,law,mode,time_varying)

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

    def __init__(self):
        self.dt          = None
        self.x           = None
        self.xHat        = None
        self.xLast       = None
        self.thetaHat    = None
        self.theta       = None
        self.thetaMax    = None
        self.thetaMin    = None
        self.thetaHatDot = None
        self.Mat         = None
        self.vec         = None
        self.minSigma    = 0

    def set_initial_conditions(self,**settings):
        """ """
        # Set Default Initial Conditions
        self.dt          = dt
        self.x           = x0
        self.xHat        = x0
        self.thetaHat    = thetaHat
        self.thetaMax    = thetaMax
        self.thetaMin    = thetaMin
        self.thetaHatDot = 0*thetaHat
        self.Mat         = 0*regressor(self.x)
        self.vec         = 0*x0

        if 'x0' in settings.keys():
            assert type(settings['x0']) == np.ndarray
            self.x    = settings['x0']
            self.xHat = settings['x0']

        if 'x' in settings.keys():
            assert type(settings['x']) == np.ndarray
            self.x    = settings['x']
            self.xHat = settings['x']

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
        self.GammaS    = kGs * c**2 / (2 * np.min(cbf(self.x))) * np.eye(self.thetaHat.shape[0])
        self.GammaE    = self.GammaS * kGe
        ic(self.GammaE)
        ic(self.GammaS)
        self.V0_T      = np.inf
        self.V0        = 1/2 * self.errMax.T @ np.linalg.inv(self.GammaS) @ self.errMax
        self.Vmax      = self.V0
        self.eta       = self.Vmax

        self.W         = np.zeros(regressor(self.x).shape)
        self.xi        = self.e

        # Alternate technique when unknown parameters appear in multiple stages of dynamics
        # reg_depth      = 3
        # reg            = reg_est(self.x,self.thetaHat)[:reg_depth]
        # self.xf        = np.zeros(self.x[:reg_depth].shape)#self.x
        # self.xf_dot    = np.zeros(self.x[:reg_depth].shape)
        # self.phif      = np.zeros(f(self.x)[:reg_depth].shape)
        # self.phif_dot  = np.zeros(f(self.x)[:reg_depth].shape)
        # self.Phif      = np.zeros(reg.shape)
        # self.Phif_dot  = np.zeros(reg.shape)
        # self.P         = np.zeros(np.dot(reg.T,reg).shape)
        # self.Q         = np.zeros(self.thetaHat.shape)

        reg_depth      = 3
        reg            = regressor(self.x)
        self.xf        = np.zeros(self.x.shape)
        self.xf_dot    = np.zeros(self.x.shape)
        self.phif      = np.zeros(f(self.x).shape)
        self.phif_dot  = np.zeros(f(self.x).shape)
        self.Phif      = np.zeros(reg.shape)
        self.Phif_dot  = np.zeros(reg.shape)
        self.P         = np.zeros(np.dot(reg.T,reg).shape)
        self.Q         = np.zeros(self.thetaHat.shape)

        self.update_error_bounds(0)

    def update(self,
               t:    float,
               x:    np.ndarray,
               u:    np.ndarray,
               law:  int = 1,
               mode: int = 1,
               time_varying: bool = False,
               Kw:   float = Kw):
        """ Public update method -- overwritten by child. """
        if t > 0:
            self.xLast = self.x

        return self._update(t,x,u,law,mode,time_varying,Kw)

    def _update(self,
               t:            float,
               x:            np.ndarray,
               u:            np.ndarray,
               law:          int = 1,
               mode:         int = 1,
               time_varying: bool = False,
               Kw:           float = Kw):
        """ Updates the parameter estimates and the corresponding error bounds. """
        self.t = t
        self.x = x
        self.u = u

        if t == 0:
            return self.thetaHat,self.errMax,self.etaTerms,self.xHat,self.theta,self.minSigma,self.Mat,self.vec

        # Update Unknown Parameter Estimates
        self.update_unknown_parameter_estimates(law=law,mode=mode,time_varying=time_varying,Kw=Kw)

        # Update theta_tilde upper/lower bounds
        self.update_error_bounds(self.t)

        return self.thetaHat,self.errMax,self.etaTerms,self.xHat,self.theta,self.minSigma,self.Mat,self.vec

    def update_unknown_parameter_estimates(self,
                                           law:          int   = 1,
                                           mode:         int   = 1,
                                           time_varying: bool  = False,
                                           Kw:           float = Kw):
        """
        """
        tol = 0#1e-15
        # self.update_filter_2ndOrder()

        # Update required variables and check for persistent excitation (PE)
        eig = self.update_auxiliaries(mode=mode,Kw=Kw)
        if eig <= 0:
            raise ValueError('PE Condition Not Met: Eig(P) = {:.3f}'.format(eig))

        # 'Estimator'
        if mode == 1:
            Mat = self.W
            vec = self.e
        
        # 'Filter'
        elif mode == 2:
            Mat = self.P
            vec = self.Q - self.P @ self.thetaHat
        
        # 'Rate Measurements'
        elif mode == 3:
            if self.xLast is None:
                return
            xdot = (self.x - self.xLast)/self.dt
            Mat  = regressor(self.x)
            vec  = xdot - f(self.x) - g(self.x)@self.u - regressor(self.x)@self.thetaHat

        # Minimum singular value
        u,s,v = np.linalg.svd(Mat)
        self.minSigma = np.min(s)
        self.Mat = Mat
        self.vec = vec

        # Adaptation Law from ECC2021 -- requires multiple matrix inversions
        if law == 0:
            MatInv = np.linalg.inv(-Mat)
            
            # Adaptation Law 1
            pre  = self.GammaE @ vec / (vec.T @ MatInv.T @ vec)
            V    = (1/2 * vec.T @ MatInv.T @ np.linalg.inv(self.GammaE) @ MatInv @ vec)
            self.thetaHatDot = pre * (-c1_e * V**gamma1_e - c2_e * V**gamma2_e)

        # Newly proposed Adaptation Law -- Static Parameters
        elif law == 1:
            if np.linalg.norm(vec) < 1e-15:
                self.thetaHatDot = 0.0 * self.thetaHat
                return

            # Lower adaptation rate for rate measurements
            if False:#mode == 3:
                pre  = 0.02*np.eye(3) / (np.linalg.norm(vec)) # Mode == 3
                pre  = 10.00*np.eye(3) / (np.linalg.norm(vec)) # Mode == 1
            else:
                pre  = self.GammaE / (np.linalg.norm(vec))
            
            prepreXi = 0#self.GammaE @ regressor(self.x).T @ vec
            exp1     = 2.0 / mu
            exp2     = 2.0 - 2.0 / mu
            term1    = pp * self.GammaE / (np.linalg.norm(vec) ** exp1) @ Mat.T @ vec
            term2    = qq * self.GammaE / (np.linalg.norm(vec) ** exp2) @ Mat.T @ vec * (vec.T @ vec)
            term3    = 0.0
            if time_varying == True:
                term3 = rr * self.GammaE / (np.linalg.norm(vec)) @ Mat.T @ vec

            self.thetaHatDot = prepreXi + term1 + term2 + term3

            #ic(pre)
            #ic(Mat)
            #ic(vec)
            #ic(self.GammaE)
            #ic(self.thetaHatDot)
        #thd_max = (self.thetaMax - self.thetaMin) / self.dt
        #self.thetaHatDot = np.clip(self.thetaHatDot,-thd_max,thd_max)

        norm_chk = np.linalg.norm(vec)
        if np.isnan(norm_chk) or np.isinf(norm_chk):
            print("ThetaHat: {}".format(self.thetaHat))
            print("x:    {}".format(self.x))
            print("xhat: {}".format(self.xHat))
            print("e:    {}".format(self.e))
            raise ValueError("Vec Norm Error: {}".format(norm_chk))

        thd_norm = np.linalg.norm(self.thetaHatDot)
        if thd_norm >= tol and not (np.isnan(thd_norm) or np.isinf(thd_norm)):
            self.thetaHat = self.thetaHat + (self.dt * self.thetaHatDot)
            #self.bigTheta = self.bigTheta + (self.dt * self.thetaHat)
            #self.bigTheta = np.clip(self.bigTheta,self.thetaMin,self.thetaMax)

        else:
            print("Mat: {}".format(Mat))
            print("Vec: {}".format(vec))
            print("No Theta updated")
            print("Pre  = {}\nTime = {}sec".format(pre,self.t))

        # self.theta     = Pinv @ self.Q

    def update_auxiliaries(self,
                           mode: int = 1,
                           Kw:   float = Kw):
        """ Updates the auxiliary matrix and vector for the filtering scheme.

        INPUTS:
        None

        OUTPUTS:
        float -- minimum eigenvalue of P matrix

        """
        if mode == 1:
            self.update_observer()

            Wdot   = -Kw * self.W + regressor(self.x)
            self.W = self.W  + (self.dt * Wdot)

        elif mode == 2:
            self.update_filter_2ndOrder()

            Pdot = -l_e * self.P + np.dot(self.Phif.T,self.Phif)
            Qdot = -l_e * self.Q + np.dot(self.Phif.T,(self.xf_dot - self.phif))

            self.P = self.P + (self.dt * Pdot)
            self.Q = self.Q + (self.dt * Qdot)

            norm_chk = np.linalg.norm(self.P)
            if np.isnan(norm_chk) or np.isinf(norm_chk):
                raise ValueError("P Norm out of bounds")

            norm_chk = np.linalg.norm(self.Q)
            if np.isnan(norm_chk) or np.isinf(norm_chk):
                raise ValueError("Q Norm out of bounds")

        return 1#np.min(np.linalg.eig(self.P)[0])

    def update_filter_2ndOrder(self):
        """
        Updates 2nd-order state filtering scheme according to 1st-order Euler
        derivative approximations.
        INPUTS:
        None
        OUTPUTS:
        None
        """
        # Second-Order Filter
        self.xf_dot   = self.xf_dot   + (self.dt * xf_2dot(self.x,self.xf,self.xf_dot,k_e))
        self.phif_dot = self.phif_dot + (self.dt * phif_2dot(self.x,self.u,self.phif,self.phif_dot,k_e))
        self.Phif_dot = self.Phif_dot + (self.dt * Phif_2dot(regressor(self.x),self.Phif,self.Phif_dot,k_e))

        self.xf   = self.xf   + (self.dt * self.xf_dot)
        self.phif = self.phif + (self.dt * self.phif_dot)
        self.Phif = self.Phif + (self.dt * self.Phif_dot)

    def update_observer(self):
        """ Updates the state estimate according to the observer (xhat) dynamics.

        INPUTS
        ------
        None

        OUTPUTS
        -------
        None

        """
        xHatDot = f(self.x) + g(self.x)@self.u + regressor(self.x)@self.thetaHat + Kw*self.e + self.W@self.thetaHatDot
        self.xHat = self.xHat + (self.dt * xHatDot)

    def update_error_bounds(self,t):
        """
        """
        # Update Max Error Quantities
        arc_tan      = np.arctan2(np.sqrt(c2_e) * self.V0**(1/mu_e),np.sqrt(c1_e))
        tan_arg      = -np.min([t,self.T_e]) * np.sqrt(c1_e * c2_e) / mu_e + arc_tan

        # DELETE THIS
        tan_arg      = arc_tan
        
        Vmax         = (np.sqrt(c1_e / c2_e) * np.tan(np.max([tan_arg,0]))) ** mu_e
        self.Vmax    = np.clip(Vmax,0,np.inf)

        # Update eta
        self.eta     = self.Vmax

        # Define constants
        M = 2 * np.max(self.GammaS)
        N = np.sqrt(c2_e/c1_e)
        X = np.arctan2(N * self.V0,1)
        c = c1_e
        u = mu_e
        A = N*c/u
        B = np.sqrt(M * 1 / N**u)
        tan_arg = np.min([A*t - X,0.0])

        # print("TanArg: {}".format(tan_arg))
        # print("M:      {}".format(M))
        # print("N:      {}".format(N))
        # print("X:      {}".format(X))
        # print("c:      {}".format(c))
        # print("u:      {}".format(u))


        # Update eta derivatives
        # etadot  = N*c*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)/(2*np.tan(tan_arg))
        # eta2dot = N**2*c**2*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/(4*np.tan(tan_arg)**2) - N**2*c**2*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/(2*u*np.tan(tan_arg)**2) + N**2*c**2*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)/u
        # eta3dot = N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(8*np.tan(tan_arg)**3) - 3*N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(4*u*np.tan(tan_arg)**3) + 3*N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/(2*u*np.tan(tan_arg)) + N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(u**2*np.tan(tan_arg)**3) - 2*N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/(u**2*np.tan(tan_arg)) + 2*N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)/u**2
        # eta4dot = N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**4/(16*np.tan(tan_arg)**4) - 3*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**4/(4*u*np.tan(tan_arg)**4) + 3*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(2*u*np.tan(tan_arg)**2) + 11*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**4/(4*u**2*np.tan(tan_arg)**4) - 7*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(u**2*np.tan(tan_arg)**2) + 7*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/u**2 - 3*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**4/(u**3*np.tan(tan_arg)**4) + 8*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(u**3*np.tan(tan_arg)**2) - 6*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/u**3 + 4*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)**2/u**3

        etadot  = -A*B*(np.tan(tan_arg)**2 + 1)
        eta2dot = -2*A**2*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)
        eta3dot = -2*A**3*B*(np.tan(tan_arg)**2 + 1)**2 - 4*A**3*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)**2
        eta4dot = -16*A**4*B*(np.tan(tan_arg)**2 + 1)**2*np.tan(tan_arg) - 8*A**4*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)**3
        # print(N*c*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)/(2*np.tan(tan_arg)))
        # print(N*c*np.sqrt(M*(np.tan(tan_arg)/N)**u))
        # print((np.tan(tan_arg)**2 + 1))
        # print((2*np.tan(tan_arg)))
        # print(M*(np.tan(tan_arg)/N)**u)
        # print(etadot)
        # print(eta2dot)
        # print(eta3dot)
        # print(eta4dot)

        # Update etadot terms
        self.eta_order1 = np.trace(np.linalg.inv(self.GammaS)) * (self.eta * etadot)
        self.eta_order2 = np.trace(np.linalg.inv(self.GammaS)) * (self.eta * eta2dot + etadot**2)
        self.eta_order3 = np.trace(np.linalg.inv(self.GammaS)) * (self.eta * eta3dot + 3 * etadot * eta2dot)
        self.eta_order4 = np.trace(np.linalg.inv(self.GammaS)) * (self.eta * eta4dot + 4 * etadot * eta3dot + 3 * eta2dot**2)

        if False:#self.t < T_e:
            self.etaTerms = np.array([self.eta_order1,self.eta_order2,self.eta_order3,self.eta_order4])
        else:
            self.etaTerms = np.zeros((4,))
        # print(self.etaTerms)

        # # edot_coeff   = -np.sqrt(2 * np.max(self.GammaS) * c1_e **(mu_e/2 + 1) / c2_e **(mu_e/2 - 1))
        # edot_coeff   = -np.sqrt(2 * np.max(self.GammaS) * c1_e **(1 + mu_e/2) / c2_e **(1 - mu_e/2))
        # self.etadot  = edot_coeff * np.tan(tan_arg)**(mu_e/2 - 1) / np.cos(tan_arg)**2

        # # Update eta2dot
        # eddot_coeff  = -edot_coeff * np.sqrt(c2_e / c1_e) * c1_e / mu_e
        # self.eta2dot = eddot_coeff * np.tan(tan_arg)**(mu_e/2 - 2) / np.cos(tan_arg)**2 * \
        #                ((mu_e/2 - 1) / np.cos(tan_arg)**2 + 2 * np.tan(tan_arg)**(2))

        # # Update eta3dot


        # Update max theta_tilde
        self.errMax = np.clip(np.sqrt(2*np.diagonal(self.GammaS)*self.Vmax),0,self.thetaMax-self.thetaMin)


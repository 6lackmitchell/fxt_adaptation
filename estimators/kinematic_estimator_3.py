import os
import copy
import time
import numpy as np

from .estimator import *

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
Kw        = 100
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

    def update(self,
               t: float,
               x: np.ndarray,
               u: np.ndarray):
        """ Updates the parameter estimates and the corresponding error bounds. """
        return self._update(t,x,u,law=1,mode=1,Kw=Kw)

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

    
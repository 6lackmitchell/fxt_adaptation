#!/usr/bin/env python

""" simulate.py: simulation environment for python controls/dynamics tests. """

### Import Statements ###
# General imports
import os
import sys
import pickle
import builtins
import traceback

# Authorship information
__author__     = "Mitchell Black"
__copyright__  = "Open Education - Creative Commons"
__version__    = "0.0.1"
__maintainer__ = "Mitchell Black"
__email__      = "mblackjr@umich.edu"
__status__     = "Development"

# Determine which problem we are solving
DEFAULT  = 'single_integrator'
filepath = '/home/dasc/MB/datastore/fxt_adaptation/' + DEFAULT + '/'
filepath = '/home/dasc/MB/datastore/fxt_adaptation/quadrotor/'
args = sys.argv
if len(args) > 1:
    config = str(args[1])
    if len(args) > 2:
        save_file = str(args[2])
    else:
        save_file = 'TESTING'
else:
    config = DEFAULT

PARAMETER_ADAPTATION = False

# Make problem config available to other modules
builtins.ecc_MODEL_CONFIG = config
builtins.SAVE_FILE        = filepath + save_file + '.pkl'

# Problem specific imports
if config == DEFAULT:
    from single_integrator_settings import *
    from simple_qp_controller import solve_qp
elif config == 'quadrotor':
    from quadrotor_settings import *
    from quadrotor_qp_controller import solve_qp
else:
    raise ValueError("Improper Problem Config.")

# from estimators.estimator import adaptation_law
# from estimators.nonlinear_estimator import adaptation_law
from estimators.estimator import adaptation_law, modes, pp, qq, rr, mu, GammaS, GammaE

# Logging variables
x             = np.zeros((nTimesteps,nStates))
u             = np.zeros((nTimesteps,nControls))
# p             = np.zeros((nTimesteps,nFeasibilityParams))
xhat          = np.zeros((nTimesteps,nStates))
error         = np.zeros((nTimesteps,nStates))
bigtheta      = np.zeros((nTimesteps,nParams))
thetahat      = np.zeros((nTimesteps,nParams))
thetainv      = np.zeros((nTimesteps,nParams))
minSigma      = np.zeros((nTimesteps,))
M_matrix      = np.zeros((nTimesteps,nStates,nParams))
qp_sol        = np.zeros((nTimesteps,nSols))
nominal_sol   = np.zeros((nTimesteps,nSols))
cbfs          = np.zeros((nTimesteps,nCBFs))
xf            = np.zeros((nTimesteps,nStates))
clf           = np.zeros((nTimesteps,))
true_disturb  = np.zeros((nTimesteps,nStates))

# Set initial parameters
x[0,:]        = x0
bigtheta[0,:] = bigTheta
thetahat[0,:] = thetaHat
u             = np.zeros((nControls,))
law           = 1
mode          = modes['rate']
time_varying  = False

# More initial parameters
errMax        = thetaMax - thetaMin
etaTerms      = np.zeros((4,))
GammaS        = GammaS(dt,ts,x0,u,law=law,mode=mode)
GammaE        = GammaE(dt,ts,x0,u,law=law,mode=mode)

try:
    for ii,tt in enumerate(np.linspace(ts,tf,nTimesteps)):
        if round(tt,4) % 1 == 0:
            print("Time: {} sec".format(tt))

        # Compute new parameter estimate
        #bigtheta[ii],thetahat[ii],errMax,etaTerms,Gamma,state_f,thetainv[ii],minSigma[ii] = adaptation_law(dt,tt,x[ii],u)
        #thetahat[ii],errMax,etaTerms,Gamma,xf[ii],thetainv[ii],minSigma[ii],M_matrix[ii] = adaptation_law(dt,tt,x[ii],u,law=law,mode=mode)

        # Compute Control Input
        qp_sol[ii,:],nominal_sol[ii,:] = solve_qp({'t':        tt,
                                                   'x':        x[ii],
                                                   #'bigTh':    bigtheta[ii],
                                                   'th':       thetahat[ii],
                                                   'err':      errMax,
                                                   'etaTerms': etaTerms,
                                                   'Gamma':    GammaS})
        u = qp_sol[ii,:nControls]

        # Advance Dynamics
        x[ii+1,:],true_disturb[ii+1,:] = step_dynamics(dt,x[ii],u)

        # Compute new parameter estimate
        if PARAMETER_ADAPTATION:
            thetahat[ii+1],errMax,etaTerms,xf[ii+1],thetainv[ii+1],minSigma[ii+1],M_matrix[ii+1],error[ii+1] = adaptation_law(dt,tt,x[ii+1],u,law=law,mode=mode,time_varying=time_varying)

        # Log Updates
        cbfs[ii+1] = cbf(x[ii+1])

except Exception as e:
    traceback.print_exc()

else:
    pass

finally:
    print("SIMULATION TERMINATED AT T = {:.4f} sec".format(tt))
    print("State: {}".format(x[ii]))

    # Format data here
    data = {'x':x,
            # 'p':p,
            'theta':theta,
            'sols':qp_sol,
            'sols_nom':nominal_sol,
            #'bigtheta':bigtheta,
            'thetahat':thetahat,
            'thetainv':thetainv,
            'minSigma':minSigma,
            'M_matrix':M_matrix,
            'Gamma':GammaE,
            'cbf':cbfs,
            'clf':clf,
            'disturb':true_disturb,
            'xf':xf,
            'error':error,
            'fxt_params':{'pp':pp,'qq':qq,'rr':rr,'gamma1':1 - 1/mu,'gamma2':1 + 1/mu,'mu':mu},
            'ii':ii}

    # Write data to file
    with open(builtins.SAVE_FILE,'wb') as f:
        pickle.dump(data,f)

print('\a')

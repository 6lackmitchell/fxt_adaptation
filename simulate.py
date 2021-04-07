#!/usr/bin/env python

""" simulate.py: simulation environment for python controls/dynamics tests. """

### Import Statements ###
# General imports
import os
import pickle
import traceback

# Problem specific imports
from quadrotor_settings import *
# from estimator import adaptation_law
from estimator_04072021 import adaptation_law
from quadrotor_qp_controller import solve_qp

# Authorship information
__author__     = "Mitchell Black"
__copyright__  = "Open Education - Creative Commons"
__version__    = "0.0.1"
__maintainer__ = "Mitchell Black"
__email__      = "mblackjr@umich.edu"
__status__     = "Development"

# Output file
filepath = '/home/dasc/MB/sim_data/fxt_adaptation/quadrotor/'
filename = filepath + 'NEW_ESTIMATOR.pkl'

# Logging variables
x             = np.zeros((nTimesteps,nStates))
u             = np.zeros((nTimesteps,nControls))
# p             = np.zeros((nTimesteps,nFeasibilityParams))
xhat          = np.zeros((nTimesteps,nStates))
thetahat      = np.zeros((nTimesteps,nParams))
qp_sol        = np.zeros((nTimesteps,nSols))
nominal_sol   = np.zeros((nTimesteps,nSols))
cbfs          = np.zeros((nTimesteps,nCBFs))
xf            = np.zeros((nTimesteps,nStates))
clf           = np.zeros((nTimesteps,))

# Set initial parameters
x[0,:]        = x0
thetahat[0,:] = thetaHat
u             = np.zeros((nControls,))

try:
    for ii,tt in enumerate(np.linspace(ts,tf,nTimesteps)):
        if round(tt,4) % 1 == 0:
            print("Time: {} sec".format(tt))

        # Compute new parameter estimate
        thetahat[ii],errMax,etaTerms,Gamma,state_f = adaptation_law(dt,tt,x[ii],u)

        # Compute Control Input
        qp_sol[ii,:],nominal_sol[ii,:] = solve_qp({'t':        tt,
                                                   'x':        x[ii],
                                                   'th':       thetahat[ii],
                                                   'err':      errMax,
                                                   'etaTerms': etaTerms,
                                                   'Gamma':    Gamma})
        u = qp_sol[ii,:nControls]

        # Log Updates
        cbfs[ii] = cbf(x[ii])
        xf[ii]   = state_f

        # Advance Dynamics
        x[ii+1,:] = step_dynamics(dt,x[ii],u)

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
            'thetahat':thetahat,
            'cbf':cbfs,
            'clf':clf,
            'xf':xf,
            'ii':ii}

    # Write data to file
    with open(filename,'wb') as f:
        pickle.dump(data,f)

print('\a')

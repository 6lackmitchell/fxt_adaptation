import numpy as np
from .physical_params import M, G, ELLIPSE_AX
from .control_params import f_max
# from .cbf_dynamics import *
filepath = '/home/dasc/MB/Code/sim_env/simdata/quadrotor/'
x0       = np.array([  0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0])      # Initial State
# 2 Seconds in
# x0 = np.array([4.35418928,  3.57415983,  1.99397206, -1.81182911,  1.68551367,  0.07719446,-0.07376968,  0.01424057, -0.46589271,  0.01367535, -0.04593021, -0.01919791])
# z0       = np.array([Z00(x0),
#                      0.0,
#                      Z10(x0),
#                      0.0,
#                      0.0,
#                      0.0,
#                      Z20(x0),
#                      0.0,
#                      0.0,
#                      0.0,
#                      Z30(x0),
#                      0.0])
# p0         = np.array([1.0,1.0])  # Initial State
theta    = np.array([0.85,-0.9,1.0])  # True Theta parameters
thetaHat = np.array([5.0,-5.0,-5.0]) # Initial estimates of Theta parameters
thetaMax = np.array([5.0,5.0,5.0])
# theta    = np.array([0.0,0.0,0.0])  # True Theta parameters
# thetaHat = np.array([1.0,-1.0,-1.0]) # Initial estimates of Theta parameters
# thetaMax = np.array([1.0,1.0,1.0])
thetaMin = -thetaMax
nStates  = len(x0)
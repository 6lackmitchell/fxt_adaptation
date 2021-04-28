import numpy as np
from .physical_params import *
from .initial_conditions import theta

# Define Global trig shortcuts
c = np.cos; s = np.sin; t = np.tan; se = lambda y: 1 / np.cos(y)

def R_body_to_inertial(x):
    """ Rotation matrix from body-fixed frame to inertial frame.

    INPUTS
    ------
    x: np.ndarray (nx1) - state vector

    OUTPUTS
    -------
    np.ndarray (3x3) - rotation matrix

    """
    c = np.cos; s = np.sin
    if np.sum(x.shape) == x.shape[0]:
        a = np.array([[c(x[7])*c(x[8]), s(x[6])*s(x[7])*c(x[8]) - c(x[6])*s(x[8]), c(x[6])*s(x[7])*c(x[8]) + s(x[6])*s(x[8])],
                      [c(x[7])*s(x[8]), s(x[6])*s(x[7])*s(x[8]) + c(x[6])*c(x[8]), c(x[6])*s(x[7])*s(x[8]) - s(x[6])*c(x[8])],
                      [        s(x[7]),                          -s(x[6])*c(x[7]),                          -c(x[6])*c(x[7])]])
    else:
        a = np.array([[[c(xx[7])*c(xx[8]), s(xx[6])*s(xx[7])*c(xx[8]) - c(xx[6])*s(xx[8]), c(xx[6])*s(xx[7])*c(xx[8]) + s(xx[6])*s(xx[8])],
                       [c(xx[7])*s(xx[8]), s(xx[6])*s(xx[7])*s(xx[8]) + c(xx[6])*c(xx[8]), c(xx[6])*s(xx[7])*s(xx[8]) - s(xx[6])*c(xx[8])],
                       [         s(xx[7]),                             -s(xx[6])*c(xx[7]),                             -c(xx[6])*c(xx[7])]] for xx in x])

    return a

def T_angular_rate(x):
    """ Transformation matrix from body-fixed rotation rates to Euler angle
    rotation rates:

    [phi_dot, theta_dot, psi_dot] = T_angular_rate(x) @ [p, q, r]

    INPUTS
    ------
    x: np.ndarray (nx1) - state vector

    OUTPUTS
    -------
    np.ndarray (3x3) - transformation matrix

    """
    if np.sum(x.shape) == x.shape[0]:
        a = np.array([[1, s(x[6])*t(x[7]), c(x[6])*t(x[7])],
                      [0,         c(x[6]),        -s(x[6])],
                      [0, s(x[6])/c(x[7]), c(x[6])/c(x[7])]])

    else:
        a = np.array([[[1, s(xx[6])*t(xx[7]), c(xx[6])*t(xx[7])],
                       [0,          c(xx[6]),         -s(xx[6])],
                       [0, s(xx[6])/c(xx[7]), c(xx[6])/c(xx[7])]] for xx in x])

    return a


def f(x):
    """ Autonomous component of quadrotor dynamics:

    INPUTS
    ------
    x: state vector (nx1)

    OUTPUTS
    -------
    f(x): (nx1)

    States
    ------
    0:  (pn)    inertial (north) position
    1:  (pe)    inertial (east) position
    2:  (h)     altitude
    3:  (u)     body-frame velocity in i_b direction
    4:  (v)     body-frame velocity in j_b direction
    5:  (w)     body-frame velocity in k_b direction
    6:  (phi)   roll angle defined as 3rd angle in Euler rotation (ib)
    7:  (theta) pitch angle defined as 2nd angle in Euler rotation (jb)
    8:  (psi)   yaw angle defined as 1st angle in Euler rotation (kb)
    9:  (p)     roll rate as measured along i_b
    10: (q)     pitch rate as measured along j_b
    11: (r)     yaw rate as measured along k_b

    """

    Rp = R_body_to_inertial(x)
    To = T_angular_rate(x)

    if np.sum(x.shape) == x.shape[0]:
        f = np.array([(Rp @ x[3:6])[0],
                      (Rp @ x[3:6])[1],
                      (Rp @ x[3:6])[2],
                      x[4]*x[11] - x[5]*x[10] - G*s(x[7]),
                      x[5]*x[9]  - x[3]*x[11] + G*c(x[7])*s(x[6]),
                      x[3]*x[10] - x[4]*x[9]  + G*c(x[7])*c(x[6]),
                      (To @ x[9:12])[0],
                      (To @ x[9:12])[1],
                      (To @ x[9:12])[2],
                      (Jy - Jz)/Jx * x[10]* x[11],
                      (Jz - Jx)/Jy * x[9] * x[11],
                      (Jx - Jy)/Jz * x[9] * x[10]])
    else:
        f = np.array([[(Rp[i] @ xx[3:6])[0],
                       (Rp[i] @ xx[3:6])[1],
                       (Rp[i] @ xx[3:6])[2],
                       xx[4]*xx[11] - xx[5]*xx[10] - G*s(xx[7]),
                       xx[5]*xx[9]  - xx[3]*xx[11] + G*c(xx[7])*s(xx[6]),
                       xx[3]*xx[10] - xx[4]*xx[9]  + G*c(xx[7])*c(xx[6]),
                       (To[i] @ xx[9:12])[0],
                       (To[i] @ xx[9:12])[1],
                       (To[i] @ xx[9:12])[2],
                       (Jy - Jz)/Jx * xx[10]* xx[11],
                       (Jz - Jx)/Jy * xx[9] * xx[11],
                       (Jx - Jy)/Jz * xx[9] * xx[10]] for i,xx in enumerate(x)])

    return f

def g(x):
    g = np.array([[    0,    0,    0,    0],
                  [    0,    0,    0,    0],
                  [    0,    0,    0,    0],
                  [    0,    0,    0,    0],
                  [    0,    0,    0,    0],
                  [ -1/M,    0,    0,    0],
                  [    0,    0,    0,    0],
                  [    0,    0,    0,    0],
                  [    0,    0,    0,    0],
                  [    0, 1/Jx,    0,    0],
                  [    0,    0, 1/Jy,    0],
                  [    0,    0,    0, 1/Jz]])

    if np.sum(x.shape) == x.shape[0]:
        return g
    else:
        return np.array(x.shape[0]*[g])
    return g

def regressor(x):
    if np.sum(x.shape) == x.shape[0]:

        shape_x = 0.1
        shape_y = 0.1
        shape_z = 0.1

        a11 = -(shape_x*x[0])**2
        a12 = -(shape_x*x[0] * shape_y*x[1])
        a13 = -(shape_x*x[0] * shape_z*x[2])

        a21 = -(shape_y*x[1] * shape_x*x[0])
        a22 = -(shape_y*x[1])**2
        a23 = -(shape_y*x[1] * shape_z*x[2])

        a31 = -(shape_z*x[2] * shape_x*x[0])
        a32 = -(shape_z*x[2] * shape_y*x[1])
        a33 = -(shape_z*x[2])**2

        reg = np.array([[np.exp(a11),np.exp(a12),np.exp(a13)],
                        [np.exp(a21),np.exp(a22),np.exp(a23)],
                        [np.exp(a31),np.exp(a32),np.exp(a33)]])

        reg = np.array([[np.exp(a11),          0,          0],
                        [          0,np.exp(a22),          0],
                        [          0,          0,np.exp(a33)]]) * 10

        reg = np.concatenate([np.zeros((3,3)),reg,np.zeros((6,3))])

    else:
        pass
    return reg

# Wind Regressor
def true_disturbance(x):
    RR = R_body_to_inertial(x)
    CD = 0.5 # drag coefficient

    if np.sum(x.shape) == x.shape[0]:
        # Effect of Wind on Velocity
        wx   = windu_interp(np.array([x[0],x[1],x[2]]))[0]
        wy   = windv_interp(np.array([x[0],x[1],x[2]]))[0]
        wz   = windw_interp(np.array([x[0],x[1],x[2]]))[0]

        wbx,wby,wbz = RR.T @ np.array([wx,wy,wz])
        vbx,vby,vbz = np.array([x[3],x[4],x[5]])

        mag_rel_wind = np.linalg.norm(np.array([wbx - vbx, wby - vby, wbz - vbz]))
        R1 = CD * mag_rel_wind * (wbx - vbx)
        R2 = CD * mag_rel_wind * (wby - vby)
        R3 = CD * mag_rel_wind * (wbz - vbz)

        reg = np.concatenate([np.zeros((3,)),np.array([R1,R2,R3]),np.zeros((6,))])

    else:
        pass
    return reg

def system_dynamics(t,x,u):
    # reg = regressor(x) @ theta
    disturbance = true_disturbance(x)
    try:
        xdot = f(x) + np.einsum('ijk,ik->ij',g(x),u) + disturbance
    except ValueError:
        xdot = f(x) + np.dot(g(x),u) + disturbance

    return xdot,disturbance

def feasibility_dynamics(t,p,v):
    return v

def step_dynamics(dt,x,u):
    xdot,disturbance = system_dynamics(0,x,u)
    return x + dt*xdot,disturbance

nControls = g(np.zeros((1,))).shape[1]
nParams   = regressor(np.zeros((12,))).shape[1]
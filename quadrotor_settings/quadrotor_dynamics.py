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

# Wind Regressor
def regressor(x):
    RR = R_body_to_inertial(x)
    if np.sum(x.shape) == x.shape[0]:
        xx,yy,zz = RR @ np.array([x[3],x[4],x[5]])
        # Effect of Wind on Velocity
        wu   = windu_interp(np.array([x[0],x[1],x[2]]))[0]
        wv   = windv_interp(np.array([x[0],x[1],x[2]]))[0]
        ww   = windw_interp(np.array([x[0],x[1],x[2]]))[0]
        RegA = np.array([[wu - xx,       0,       0],
                         [      0, wv - yy,       0],
                         [      0,       0, ww - zz]])

        reg = np.concatenate([np.zeros((3,3)),RegA,np.zeros((6,3))])

    else:
        pass
    return reg

# # Sinusoidal Regressor
# def regressor(x):
#     RR = R_body_to_inertial(x)
#     if np.sum(x.shape) == x.shape[0]:
#         xx,yy,zz = RR @ np.array([x[3],x[4],x[5]])
#         # Effect of Wind on Velocity
#         wu   = 1 + np.sin(zz)**2
#         wv   = 1 + np.sin(xx)**2
#         ww   = 1 + np.sin(yy)**2
#         RegA = np.array([[wu,       0,       0],
#                          [      0, wv,       0],
#                          [      0,       0, ww]])

#         reg = np.concatenate([np.zeros((3,3)),RegA,np.zeros((6,3))])

#     else:
#         pass
#     return reg

# def reg_est(x,thetaHat):
#     RR = R_body_to_inertial(x)
#     if np.sum(x.shape) == x.shape[0]:

#         # Effect of Wind on Velocity
#         wu   = windu_interp(np.array([x[0],x[1],x[2]]))[0]
#         wv   = windv_interp(np.array([x[0],x[1],x[2]]))[0]
#         ww   = windw_interp(np.array([x[0],x[1],x[2]]))[0]
#         RegV = np.array([[ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0]])

#         # # Effect of Wind on Acceleration
#         # dux = windu_diffx(np.array([x[0],x[1],x[2]]))[0]
#         # duy = windu_diffy(np.array([x[0],x[1],x[2]]))[0]
#         # duz = windu_diffz(np.array([x[0],x[1],x[2]]))[0]
#         # dvx = windv_diffx(np.array([x[0],x[1],x[2]]))[0]
#         # dvy = windv_diffy(np.array([x[0],x[1],x[2]]))[0]
#         # dvz = windv_diffz(np.array([x[0],x[1],x[2]]))[0]
#         # dwx = windw_diffx(np.array([x[0],x[1],x[2]]))[0]
#         # dwy = windw_diffy(np.array([x[0],x[1],x[2]]))[0]
#         # dwz = windw_diffz(np.array([x[0],x[1],x[2]]))[0]

#         # dDdPx = np.array([[dux,duy,duz],
#         #                  [0,0,0],
#         #                  [0,0,0]])
#         # dDdPy = np.array([[0,0,0],
#         #                  [dvx,dvy,dvz],
#         #                  [0,0,0]])
#         # dDdPz = np.array([[0,0,0],
#         #                  [0,0,0],
#         #                  [dwx,dwy,dwz]])
#         # dDdP = np.array([dDdPx,dDdPy,dDdPz])
#         # RegA = -RR.T @ (dDdP @ RR @ np.array([x[3],x[4],x[5]]) + 2 * thetaHat @ dDdP @ RegV[0:3])
#         # RegA = np.concatenate([np.zeros((3,3)),RegA,np.zeros((6,3))])

#         reg  = RegV# + RegA

#     return reg

# # Wind Regressor
# def regressor(x,theta):
#     RR = R_body_to_inertial(x)
#     if np.sum(x.shape) == x.shape[0]:
#         u,v,w = np.array([x[3],x[4],x[5]])

#         # Effect of Wind on Velocity
#         wu   = windu_interp(np.array([x[0],x[1],x[2]]))[0]
#         wv   = windv_interp(np.array([x[0],x[1],x[2]]))[0]
#         ww   = windw_interp(np.array([x[0],x[1],x[2]]))[0]
#         RegV = np.array([[wu,  0,  0],
#                          [ 0, wv,  0],
#                          [ 0,  0, ww]]) @ theta

#         # Effect of Wind on Acceleration
#         dux = windu_diffx(np.array([x[0],x[1],x[2]]))[0]
#         duy = windu_diffy(np.array([x[0],x[1],x[2]]))[0]
#         duz = windu_diffz(np.array([x[0],x[1],x[2]]))[0]
#         dvx = windv_diffx(np.array([x[0],x[1],x[2]]))[0]
#         dvy = windv_diffy(np.array([x[0],x[1],x[2]]))[0]
#         dvz = windv_diffz(np.array([x[0],x[1],x[2]]))[0]
#         dwx = windw_diffx(np.array([x[0],x[1],x[2]]))[0]
#         dwy = windw_diffy(np.array([x[0],x[1],x[2]]))[0]
#         dwz = windw_diffz(np.array([x[0],x[1],x[2]]))[0]

#         dDdPx = np.array([[dux,duy,duz],
#                          [0,0,0],
#                          [0,0,0]])
#         dDdPy = np.array([[0,0,0],
#                          [dvx,dvy,dvz],
#                          [0,0,0]])
#         dDdPz = np.array([[0,0,0],
#                          [0,0,0],
#                          [dwx,dwy,dwz]])
#         dDdP = np.array([dDdPx,dDdPy,dDdPz])
#         RegA = -RR.T @ dDdP @ (RR @ np.array([x[3],x[4],x[5]]) + RegV) @ theta

#         reg = np.concatenate([RegV,RegA,np.zeros((6,))])

#     else:
#         pass
#         # dWU = windu_interp([x[:,0],x[:,1],x[:,2]])
#         # dWV = windv_interp([x[:,0],x[:,1],x[:,2]])
#         # dWW = windw_interp([x[:,0],x[:,1],x[:,2]])
#         # a = np.array([[[ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [dwu,  0,  0],
#         #                [ 0, dwv,  0],
#         #                [ 0,   0,dww],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0]] for dwu,dwv,dww in zip(dWU,dWV,dWW)])
#     return reg

# def reg_est(x,thetaHat):
#     RR = R_body_to_inertial(x)
#     if np.sum(x.shape) == x.shape[0]:
#         # Effect of Wind on Velocity
#         wu   = windu_interp(np.array([x[0],x[1],x[2]]))[0]
#         wv   = windv_interp(np.array([x[0],x[1],x[2]]))[0]
#         ww   = windw_interp(np.array([x[0],x[1],x[2]]))[0]
#         RegV = np.array([[wu,  0,  0],
#                          [ 0, wv,  0],
#                          [ 0,  0, ww],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0]])

#         # # Effect of Wind on Acceleration
#         # dux = windu_diffx(np.array([x[0],x[1],x[2]]))[0]
#         # duy = windu_diffy(np.array([x[0],x[1],x[2]]))[0]
#         # duz = windu_diffz(np.array([x[0],x[1],x[2]]))[0]
#         # dvx = windv_diffx(np.array([x[0],x[1],x[2]]))[0]
#         # dvy = windv_diffy(np.array([x[0],x[1],x[2]]))[0]
#         # dvz = windv_diffz(np.array([x[0],x[1],x[2]]))[0]
#         # dwx = windw_diffx(np.array([x[0],x[1],x[2]]))[0]
#         # dwy = windw_diffy(np.array([x[0],x[1],x[2]]))[0]
#         # dwz = windw_diffz(np.array([x[0],x[1],x[2]]))[0]

#         # dDdPx = np.array([[dux,duy,duz],
#         #                  [0,0,0],
#         #                  [0,0,0]])
#         # dDdPy = np.array([[0,0,0],
#         #                  [dvx,dvy,dvz],
#         #                  [0,0,0]])
#         # dDdPz = np.array([[0,0,0],
#         #                  [0,0,0],
#         #                  [dwx,dwy,dwz]])
#         # dDdP = np.array([dDdPx,dDdPy,dDdPz])
#         # RegA = -RR.T @ (dDdP @ RR @ np.array([x[3],x[4],x[5]]) + 2 * thetaHat @ dDdP @ RegV[0:3])
#         # RegA = np.concatenate([np.zeros((3,3)),RegA,np.zeros((6,3))])

#         reg  = RegV# + RegA

#     return reg

# def reg_const(x,thetaHat):
#     RR = R_body_to_inertial(x)
#     if np.sum(x.shape) == x.shape[0]:
#         # Effect of Wind on Velocity
#         wu   = windu_interp(np.array([x[0],x[1],x[2]]))[0]
#         wv   = windv_interp(np.array([x[0],x[1],x[2]]))[0]
#         ww   = windw_interp(np.array([x[0],x[1],x[2]]))[0]
#         RegV = np.array([[wu,  0,  0],
#                          [ 0, wv,  0],
#                          [ 0,  0, ww],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0]])

#         # Effect of Wind on Acceleration
#         dux = windu_diffx(np.array([x[0],x[1],x[2]]))[0]
#         duy = windu_diffy(np.array([x[0],x[1],x[2]]))[0]
#         duz = windu_diffz(np.array([x[0],x[1],x[2]]))[0]
#         dvx = windv_diffx(np.array([x[0],x[1],x[2]]))[0]
#         dvy = windv_diffy(np.array([x[0],x[1],x[2]]))[0]
#         dvz = windv_diffz(np.array([x[0],x[1],x[2]]))[0]
#         dwx = windw_diffx(np.array([x[0],x[1],x[2]]))[0]
#         dwy = windw_diffy(np.array([x[0],x[1],x[2]]))[0]
#         dwz = windw_diffz(np.array([x[0],x[1],x[2]]))[0]

#         dDdPx = np.array([[dux,duy,duz],
#                          [0,0,0],
#                          [0,0,0]])
#         dDdPy = np.array([[0,0,0],
#                          [dvx,dvy,dvz],
#                          [0,0,0]])
#         dDdPz = np.array([[0,0,0],
#                          [0,0,0],
#                          [dwx,dwy,dwz]])
#         dDdP = np.array([dDdPx,dDdPy,dDdPz])
#         RegConst = RR.T @ thetaHat.T @ dDdP @ RegV[0:3] @ thetaHat
#         RegConst = np.concatenate([np.zeros((3,)),RegConst,np.zeros((6,))])

#         return RegConst

# # Wind Regressor
# def regressor(x,theta):
#     RR = R_body_to_inertial(x)
#     if np.sum(x.shape) == x.shape[0]:
#         # Effect of Wind on Velocity
#         wu   = 1 + np.sin(x[0])**2
#         wv   = 1 + np.cos(x[1])**2
#         ww   = 1 + np.sin(x[2])**2
#         RegV = np.array([[wu,  0,  0],
#                          [ 0, wv,  0],
#                          [ 0,  0, ww]]) @ theta

#         # Effect of Wind on Acceleration
#         dux = 2 * np.sin(x[0]) * np.cos(x[0])
#         duy = 0
#         duz = 0
#         dvx = 0
#         dvy = 2 * np.cos(x[1]) * -np.sin(x[1])
#         dvz = 0
#         dwx = 0
#         dwy = 0
#         dwz = 2 * np.sin(x[2]) * np.cos(x[2])

#         dDdPx = np.array([[dux,duy,duz],
#                          [0,0,0],
#                          [0,0,0]])
#         dDdPy = np.array([[0,0,0],
#                          [dvx,dvy,dvz],
#                          [0,0,0]])
#         dDdPz = np.array([[0,0,0],
#                          [0,0,0],
#                          [dwx,dwy,dwz]])
#         dDdP = np.array([dDdPx,dDdPy,dDdPz])
#         RegA = -RR.T @ dDdP @ (RR @ np.array([x[3],x[4],x[5]]) + RegV) @ theta

#         reg = np.concatenate([RegV,RegA,np.zeros((6,))])

#     else:
#         pass
#         # dWU = windu_interp([x[:,0],x[:,1],x[:,2]])
#         # dWV = windv_interp([x[:,0],x[:,1],x[:,2]])
#         # dWW = windw_interp([x[:,0],x[:,1],x[:,2]])
#         # a = np.array([[[ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [dwu,  0,  0],
#         #                [ 0, dwv,  0],
#         #                [ 0,   0,dww],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0],
#         #                [ 0,   0,  0]] for dwu,dwv,dww in zip(dWU,dWV,dWW)])
#     return reg

# def reg_est(x,thetaHat):
#     scale = 0.2
#     RR = R_body_to_inertial(x)
#     if np.sum(x.shape) == x.shape[0]:
#         scale = 0.2
#         # Effect of Wind on Velocity
#         wu   = 1 + np.sin(x[0])**2
#         wv   = 1 + np.cos(x[1])**2
#         ww   = 1 + np.sin(x[2])**2
#         RegV = np.array([[wu,  0,  0],
#                          [ 0, wv,  0],
#                          [ 0,  0, ww],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0],
#                          [ 0,  0,  0]])

#         # # Effect of Wind on Acceleration
#         # dux = 2 * np.sin(x[0]) * np.cos(x[0])
#         # duy = 0
#         # duz = 0
#         # dvx = 0
#         # dvy = 2 * np.cos(x[1]) * -np.sin(x[1])
#         # dvz = 0
#         # dwx = 0
#         # dwy = 0
#         # dwz = 2 * np.sin(x[2]) * np.cos(x[2])

#         # dDdPx = np.array([[dux,duy,duz],
#         #                  [0,0,0],
#         #                  [0,0,0]])
#         # dDdPy = np.array([[0,0,0],
#         #                  [dvx,dvy,dvz],
#         #                  [0,0,0]])
#         # dDdPz = np.array([[0,0,0],
#         #                  [0,0,0],
#         #                  [dwx,dwy,dwz]])
#         # dDdP = np.array([dDdPx,dDdPy,dDdPz])
#         # RegA = -RR.T @ (dDdP @ RR @ np.array([x[3],x[4],x[5]]) + 2 * thetaHat @ dDdP @ RegV[0:3])
#         # RegA = np.concatenate([np.zeros((3,3)),RegA,np.zeros((6,3))])

#         reg  = RegV #+ RegA

#     return reg

# # def reg_const(x,thetaHat):
# #     RR = R_body_to_inertial(x)
# #     if np.sum(x.shape) == x.shape[0]:
# #         # Effect of Wind on Velocity
# #         wu   = 1 + np.sin(x[0])**2
# #         wv   = 1 + np.cos(x[1])**2
# #         ww   = 1 + np.sin(x[2])**2
# #         RegV = np.array([[wu,  0,  0],
# #                          [ 0, wv,  0],
# #                          [ 0,  0, ww],
# #                          [ 0,  0,  0],
# #                          [ 0,  0,  0],
# #                          [ 0,  0,  0],
# #                          [ 0,  0,  0],
# #                          [ 0,  0,  0],
# #                          [ 0,  0,  0],
# #                          [ 0,  0,  0],
# #                          [ 0,  0,  0],
# #                          [ 0,  0,  0]])

# #         # Effect of Wind on Acceleration
# #         dux = 2 * np.sin(x[0]) * np.cos(x[0])
# #         duy = 0
# #         duz = 0
# #         dvx = 0
# #         dvy = 2 * np.cos(x[1]) * -np.sin(x[1])
# #         dvz = 0
# #         dwx = 0
# #         dwy = 0
# #         dwz = 2 * np.sin(x[2]) * np.cos(x[2])

# #         dDdPx = np.array([[dux,duy,duz],
# #                          [0,0,0],
# #                          [0,0,0]])
# #         dDdPy = np.array([[0,0,0],
# #                          [dvx,dvy,dvz],
# #                          [0,0,0]])
# #         dDdPz = np.array([[0,0,0],
# #                          [0,0,0],
# #                          [dwx,dwy,dwz]])
# #         dDdP = np.array([dDdPx,dDdPy,dDdPz])
# #         RegConst = RR.T @ thetaHat.T @ dDdP @ RegV[0:3] @ thetaHat
# #         RegConst = np.concatenate([np.zeros((3,)),RegConst,np.zeros((6,))])

# #         return RegConst

def system_dynamics(t,x,u,theta):
    reg = regressor(x) @ theta
    try:
        xdot = f(x) + np.einsum('ijk,ik->ij',g(x),u) + reg
    except ValueError:
        xdot = f(x) + np.dot(g(x),u) + reg

    return xdot

def feasibility_dynamics(t,p,v):
    return v

def step_dynamics(dt,x,u):
    return x + dt*system_dynamics(0,x,u,theta)

nControls = g(np.zeros((1,))).shape[1]
nParams   = regressor(np.zeros((12,))).shape[1]
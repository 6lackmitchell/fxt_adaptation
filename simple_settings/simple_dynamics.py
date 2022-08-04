import numpy as np
from .control_params import U_MAX

def f(x):
    a = np.array([0,0])
    if np.sum(x.shape) == x.shape[0]:
        return a
    else:
        return np.tile(a,(x.shape[0],1))

def g(x):
    a = np.array([[1,0],
                  [0,1]])
    if np.sum(x.shape) == a.shape[0]:
        return a
    else:
        return np.array(x.shape[0]*[a])

def regressor(x,t):
    # Function of t
    # a = np.array([[1 + np.sin(np.pi*t)**2, 0],[0, 1 + np.cos(np.pi*t)**2]])
    # if np.sum(x.shape) != x.shape[0]:
    #     a = np.array(x.shape[0]*[a])
    r1 = lambda s: 1 + np.sin(2*np.pi*s)**2
    r2 = lambda s: 1 + np.cos(8*np.pi*s)**2

    # Rank-Deficient Case
    r1 = lambda s: -0.5*s
    r2 = lambda s: -1*s
    scale = 0.5

    if np.sum(x.shape) == x.shape[0]:

        # # Full-rank Regressor
        # a = np.array([[r1(x[0]), 0],[0, r2(x[1])]])

        # Rank-deficient Regressor
        a = np.array([[r1(x[0]),r2(x[0])],[scale*r1(x[0]),scale*r2(x[0])]])

    else:
        # # Full-rank Regressor
        # a = [np.array([[r1(xx[0]), 0],[0, r2(xx[1])]]) for xx in x]

        # Rank-deficient Regressor
        a = [np.array([[r1(xx[0]), r2(xx[0])],[scale*r1(xx[0]), scale*r2(xx[0])]]) for xx in x]

        # Resolve dimensions
        a = np.array(a)

    return U_MAX[0] / 3.0 * a

def system_dynamics(t,x,u,theta):
    black_reg = regressor(x,t)#; black_reg[-1,:,:] = 0

    # 1D Control Problem
    # xdot = f(x) + np.einsum('ij,i->ij',g(x),u) + np.dot(black_reg,theta)

    # 2D Control Problem
    # print(black_reg.shape,theta.shape)
    # print("U: {}".format(u))
    # print("Uncertain Term: {}".format(np.dot(black_reg,theta)))
    
    xdot = f(x) + np.einsum('ijk,ik->ij',g(x),u) + np.dot(black_reg,theta) #np.einsum('ijk,k->ij',black_reg,theta) #
    return xdot

def xf_2dot(x,xf,xf_dot,k):
    return (x - xf - 2*k*xf_dot) / (k**2)

def phif_2dot(x,u,phif,phif_dot,k):
    return (f(x) + np.dot(g(x),u) - phif - 2*k*phif_dot) / (k**2)

def Phif_2dot(reg,Phif,Phif_dot,k):
    return (reg - Phif - 2*k*Phif_dot) / (k**2)

def fd(x,xd):
    return np.array([0,0])
    



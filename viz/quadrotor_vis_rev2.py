import matplotlib
matplotlib.use("Qt5Agg")

# %matplotlib notebook
import project_path
import pickle
import traceback
import numpy as np
import gurobipy as gp
# import latex

import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator as rgi
from icecream import ic


from quadrotor_settings import tf, dt, k1, k2, arm_length, f_max, tx_max, ty_max, tz_max, G, M, F_GERONO, A_GERONO, thetaMax, regressor
# from ecc_controller_test import T_e

# with plt.style.context('seaborn-colorblind'):
#     plt.rcParams["axes.edgecolor"] = "1"
#     plt.rcParams["axes.linewidth"] = "2"

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'figure.autolayout': True})
plt.style.use(['fivethirtyeight','seaborn-colorblind'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

filepath = '/home/dasc/MB/datastore/fxt_adaptation/quadrotor/'

nControl = 1
ecc      = 0
bla      = 1
tay      = 2
lop      = 3
lsm      = 4
zha      = 5
fxt      = 6
t        = np.linspace(dt,tf,int(tf/dt))

### Define Recording Variables ###
t         = np.linspace(dt,tf,int(tf/dt))
x         = np.zeros(((int(tf/dt)+1),nControl,2))
sols      = np.zeros(((int(tf/dt)+1),nControl,5))
theta_hat = np.zeros(((int(tf/dt)+1),nControl,2))
cbf_val   = np.zeros(((int(tf/dt)+1),nControl,2))

import builtins
if hasattr(builtins,"VIS_FILE"):
    filename = filepath + builtins.VIS_FILE + '.pkl'
else:
    filename = filepath + 'money_set/MONEY_DATASET.pkl'

with open(filename,'rb') as f:
    try:
        data       = pickle.load(f)
        x          = data['x']
        theta      = data['theta']
        sols       = data['sols']
        sols_nom   = data['sols_nom']
        #theta_hat  = data['bigtheta']
        theta_hat  = data['thetahat']
        #d_theta    = data['thetahat']
        theta_inv  = data['thetainv']
        cbf_val    = data['cbf']
        xf         = data['xf']
        minSigma   = data['minSigma']
        #M_matrix   = data['M_matrix']
        Gamma      = data['Gamma']
        disturb    = data['disturb']
        error      = data['error']
        ii         = data['ii']
        try:
            fxt_params = data['fxt_params']
        except:
            fxt_params = {'pp':1,'qq':1,'gamma1':0.5,'gamma2':1.5}
    except:
        traceback.print_exc()

# Compute FxT Bound
try:
    mu = fxt_params['mu']
except:
    mu = 1.5
pp = fxt_params['pp']
qq = fxt_params['qq']
rr = fxt_params['rr']
g1 = fxt_params['gamma1']
g2 = fxt_params['gamma2']
warmup = 0.5
warmup = 0.001
warmup = 2*dt
sig = np.min(minSigma[int(warmup/dt):ii])
lam = 1 / Gamma[0,0]
G = sig * np.sqrt(2 / lam)
ic(G)

# Compute Max Derivative
delta_disturb = []
flag = True
max_d = 0
for jj,dd in enumerate(disturb[1:ii]):
    if flag:
        old_disturb = dd
        flag = False
        continue
    ddh = np.amax(dd - old_disturb)
    ddl = abs(np.amin(dd - old_disturb))
    ddd = np.max([ddh,ddl])
    if ddd > max_d:
        max_d = ddd
    old_disturb = dd

ic(max_d / dt)

# Compute T_Fixed: mu * (1 / bG^() + 1 / aG^())
c1 = pp * G**(2.0 - 2.0/mu)
c2 = qq * G**(2.0 + 2.0/mu)
#c1 = pp * sig * (2 / lam) ** g1
#c2 = qq * sig**3 * (2 / lam) ** g2
#T_fixed = 2 * (c1 + c2)/(c1*c2) + warmup
T_fixed = 1 / (c1 * (1 - g1)) + 1 / (c2 * (g2 - 1))
ic(Gamma)
ic(sig)
ic(lam)
ic(c1,c2)
ic(T_fixed)

lwidth = 2
dash = [5,2]

"""
# Generate True Theta params
wind_file = loadmat('/home/dasc/MB/datastore/fxt_adaptation/quadrotor/wind_data/wind_velocity_data.mat')
wind_u    = wind_file['u2']
wind_v    = wind_file['v2']
wind_w    = wind_file['w2']

# Scale the winds
SCALE = 1.

# Configure wind mesh
xlim      = 5.0
ylim      = 5.0
zlim      = 5.0
xx        = np.linspace(-xlim,xlim,wind_u.shape[0])
yy        = np.linspace(-ylim,ylim,wind_v.shape[1])
zz        = np.linspace( -0.2,zlim,wind_w.shape[2])

# Configure Wind Interpolating Functions
windu_interp = rgi((xx,yy,zz),wind_u / SCALE)
windv_interp = rgi((xx,yy,zz),wind_v / SCALE)
windw_interp = rgi((xx,yy,zz),wind_w / SCALE)

ii = np.min([int(tf/dt),ii])

true_thetas = np.zeros((ii,3))
for jj in range(ii):
    break
    try:
        true_thetas[jj,0] = windu_interp(np.array([x[jj,0],x[jj,1],x[jj,2]]))[0]
        true_thetas[jj,1] = windv_interp(np.array([x[jj,0],x[jj,1],x[jj,2]]))[0]
        true_thetas[jj,2] = windw_interp(np.array([x[jj,0],x[jj,1],x[jj,2]]))[0]
    except ValueError:
        true_thetas[jj,0] = 0.0
        true_thetas[jj,1] = 0.0
        true_thetas[jj,2] = 0.0
"""

#force_est = np.zeros(theta_hat.shape)
#for jj,th in enumerate(theta_hat):
#    force_est[jj] = regressor(x[jj])[3:6] @ theta_hat[jj]

#force_est = np.zeros(theta_hat.shape)
#for jj,th in enumerate(theta_hat):
#    force_est[jj] = regressor(th)[3:6] @ theta_hat[jj]

# ii = np.min([int(5.418/dt),ii])

def set_edges_black(ax):
    ax.spines['bottom'].set_color('#000000')
    ax.spines['top'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_color('#000000')

plt.close('all')


############################################
### Control Trajectories ###
fig4 = plt.figure(figsize=(8,8))
ax2a  = fig4.add_subplot(411)
ax2b  = fig4.add_subplot(412)
ax2c  = fig4.add_subplot(413)
ax2d  = fig4.add_subplot(414)
set_edges_black(ax2a)
set_edges_black(ax2b)
set_edges_black(ax2c)
set_edges_black(ax2d)

# # Control via Motor Commands
# ax2a.plot(t[1:ii],4*k1*U_MAX[0]*np.ones(t[1:ii].shape),label=r'$F_{max}$',linewidth=lwidth,color='k')
# ax2a.plot(t[1:ii],0.0*np.ones(t[1:ii].shape),linewidth=lwidth,color='k',label=r'$F_{min}$')
# ax2a.plot(t[:ii],k1*(sols[:ii,ecc,0]+sols[:ii,ecc,1]+sols[:ii,ecc,2]+sols[:ii,ecc,3]),label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2a.set(ylabel=r'$F$',ylim=[-0.5,4*k1*U_MAX[0]+0.5])#,title='Control Inputs'),xlim=[-0.1,5.2],
# ax2b.plot(t[1:ii],arm_length*k1*-U_MAX[1]*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\phi}$',linewidth=lwidth,color='k')
# ax2b.plot(t[1:ii],arm_length*k1* U_MAX[1]*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2b.plot(t[:ii],arm_length*k1*(-sols[:ii,ecc,1]+sols[:ii,ecc,3]),label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2b.set(xlabel='Time (sec)',ylabel=r'$u_{y}$',ylim=[-1.1*arm_length*k1*U_MAX[1],1.1*arm_length*k1* U_MAX[1]])#xlim=[-0.1,5.2],
# ax2c.plot(t[1:ii],arm_length*k1*-U_MAX[0]*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\theta}$',linewidth=lwidth,color='k')
# ax2c.plot(t[1:ii],arm_length*k1* U_MAX[0]*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2c.plot(t[:ii],arm_length*k1*(-sols[:ii,ecc,2]+sols[:ii,ecc,0]),label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2c.set(xlabel='Time (sec)',ylabel=r'$u_{y}$',ylim=[1.1*arm_length*k1*-U_MAX[0],1.1*arm_length*k1* U_MAX[0]])#xlim=[-0.1,5.2],
# ax2d.plot(t[1:ii],2*k2*-U_MAX[0]*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\psi}$',linewidth=lwidth,color='k')
# ax2d.plot(t[1:ii],2*k2* U_MAX[0]*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2d.plot(t[:ii],k2*(sols[:ii,ecc,1]+sols[:ii,ecc,3]-sols[:ii,ecc,0]-sols[:ii,ecc,2]),label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2d.set(xlabel='Time (sec)',ylabel=r'$u_{y}$',ylim=[1.1*2*k2*-U_MAX[0],1.1*2*k2*U_MAX[0]])#xlim=[-0.1,5.2],

# Control via Force
ax2a.plot(t[1:ii],f_max*np.ones(t[1:ii].shape),label=r'$F_{max}$',linewidth=lwidth+1,color='k')
ax2a.plot(t[1:ii],0.0*np.ones(t[1:ii].shape),label=r'$F_{min}$',linewidth=lwidth+1,color='k')
ax2a.plot(t[:ii],sols_nom[:ii,0],label=r'$F_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2a.plot(t[:ii],sols[:ii,0],label='F',linewidth=lwidth,color=colors[ecc])
# ax2a.plot(t[:ii],G*M*np.ones(len(t[:ii]),),label='Gravity',linewidth=lwidth,color=colors[3],dashes=dash)
ax2a.set(ylabel=r'$F$',ylim=[-0.5,f_max*1.1],title='Control Inputs')#,xlim=[-0.1,5.2],
ax2b.plot(t[1:ii],-tx_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\phi}$',linewidth=lwidth+1,color='k')
ax2b.plot(t[1:ii], tx_max*np.ones(t[1:ii].shape),linewidth=lwidth+1,color='k')
# ax2b.plot(t[:ii],arm_length*k1*(x[:ii,ecc,15] - x[:ii,ecc,13]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2b.plot(t[:ii],sols_nom[:ii,1],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2b.plot(t[:ii],sols[:ii,1],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2b.set(ylabel=r'$\mu$')#xlim=[-0.1,5.2],)
ax2b.set(ylabel=r'$\tau_{\phi}$',ylim=[-1.1*tx_max,1.1*tx_max])#xlim=[-0.1,5.2],
ax2c.plot(t[1:ii],-ty_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\theta}$',linewidth=lwidth+1,color='k')
ax2c.plot(t[1:ii], ty_max*np.ones(t[1:ii].shape),linewidth=lwidth+1,color='k')
# ax2c.plot(t[:ii],arm_length*k1*(x[:ii,ecc,12] - x[:ii,ecc,14]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2c.plot(t[:ii],sols_nom[:ii,2],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2c.plot(t[:ii],sols[:ii,2],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2c.set(ylabel=r'$\tau_{\theta}$',ylim=[-1.1*ty_max,1.1*ty_max])#xlim=[-0.1,5.2],
ax2d.plot(t[1:ii],-tz_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\psi}$',linewidth=lwidth+1,color='k')
ax2d.plot(t[1:ii], tz_max*np.ones(t[1:ii].shape),linewidth=lwidth+1,color='k')
# ax2d.plot(t[:ii],k2*(-x[:ii,ecc,12]+x[:ii,ecc,13]-x[:ii,ecc,14]+x[:ii,ecc,15]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2d.plot(t[:ii],sols_nom[:ii,3],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2d.plot(t[:ii],sols[:ii,3],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2d.set(xlabel='Time (sec)',ylabel=r'$\tau_{\psi}$',ylim=[-1.1*tz_max,1.1*tz_max])#xlim=[-0.1,5.2],


for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
             ax2a.get_xticklabels() + ax2a.get_yticklabels()):
    item.set_fontsize(25)
ax2a.legend(fancybox=True)
ax2a.grid(True,linestyle='dotted',color='white')

for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
             ax2b.get_xticklabels() + ax2b.get_yticklabels()):
    item.set_fontsize(25)
ax2b.legend(fancybox=True)
ax2b.grid(True,linestyle='dotted',color='white')

for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
             ax2c.get_xticklabels() + ax2c.get_yticklabels()):
    item.set_fontsize(25)
ax2c.legend(fancybox=True)
ax2c.grid(True,linestyle='dotted',color='white')

for item in ([ax2d.title, ax2d.xaxis.label, ax2d.yaxis.label] +
             ax2d.get_xticklabels() + ax2d.get_yticklabels()):
    item.set_fontsize(25)
ax2d.legend(fancybox=True)
ax2d.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)




############################################
### CBF Trajectories ###
nCBFs = len(cbf_val[1])
cbf_labels = ['Altitude','Attitude','xVel','yVel','zVel']
fig5 = plt.figure(figsize=(8,8))
ax3  = fig5.add_subplot(111)
set_edges_black(ax3)
ax3.plot(t[1:ii],np.zeros(t[1:ii].shape),label=r'Boundary',linewidth=lwidth,color='k')
for nn in range(nCBFs):
    ax3.plot(t[1:ii],cbf_val[1:ii,nn],label=cbf_labels[nn],linewidth=lwidth,color=colors[ecc+nn])


#ax3.plot(t[1:ii],cbf_val[1:ii,0],label='Altitude',linewidth=lwidth,color=colors[ecc])
#ax3.plot(t[1:ii],cbf_val[1:ii,1],label='Attitude',linewidth=lwidth,color=colors[ecc+5])


ax3.set(xlabel='Time (sec)',ylabel='h(x)',title='CBF Trajectory')
ax3.set(xlim=[-0.5,6.5],ylim=[-1,20])
for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
             ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(25)
ax3.legend(fancybox=True,loc=1)
ax3.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)



############################################
### Min Singular Value Trajectories ###
fig999 = plt.figure(figsize=(8,8))
ax999   = fig999.add_subplot(111)
set_edges_black(ax999)
ax999.plot(t[:ii],minSigma[:ii],label=r'M',linewidth=lwidth,color=colors[ecc])
ax999.set(xlabel='Time (sec)',ylabel=r'$\sigma$',title='Minimum Singular Value')
#ax999.set(xlim=[-0.5,6.5],ylim=[-1,20])
for item in ([ax999.title, ax999.xaxis.label, ax999.yaxis.label] +
             ax999.get_xticklabels() + ax999.get_yticklabels()):
    item.set_fontsize(25)
ax999.legend(fancybox=True,loc=1)
ax999.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)



############################################
### Error Trajectories ###
fig0999 = plt.figure(figsize=(8,8))
ax0999   = fig0999.add_subplot(111)
set_edges_black(ax0999)
ax0999.plot(t[:ii],error[:ii,3],label=r'e4',linewidth=lwidth,color=colors[ecc])
ax0999.plot(t[:ii],error[:ii,4],label=r'e5',linewidth=lwidth,color=colors[ecc+1])
ax0999.plot(t[:ii],error[:ii,5],label=r'e6',linewidth=lwidth,color=colors[ecc+2])
ax0999.set(xlabel='Time (sec)',ylabel=r'$e$',title='Observer Error')
ylimlo = np.max([np.min(error),-100])
ylimhi = np.min([np.max(error),100]) 
ax0999.set(ylim=[ylimlo,ylimhi])
for item in ([ax0999.title, ax0999.xaxis.label, ax0999.yaxis.label] +
             ax0999.get_xticklabels() + ax0999.get_yticklabels()):
    item.set_fontsize(25)
ax0999.legend(fancybox=True,loc=1)
ax0999.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)


############################################
### Error Trajectories ###
fig012 = plt.figure(figsize=(8,8))
ax012   = fig012.add_subplot(111)
set_edges_black(ax012)
ax012.plot(t[:ii],x[:ii,0],label=r'x',linewidth=lwidth,color=colors[ecc])
ax012.plot(t[:ii],x[:ii,1],label=r'y',linewidth=lwidth,color=colors[ecc+1])
ax012.plot(t[:ii],x[:ii,2],label=r'z',linewidth=lwidth,color=colors[ecc+2])
ax012.set(xlabel='Time (sec)',ylabel=r'$Position (m)$',title='XYZ Trajectories')
ylimlo = np.max([np.min(x[:,:3]),-100])
ylimhi = np.min([np.max(x[:,:3]), 100]) 
ax012.set(ylim=[ylimlo,ylimhi])
for item in ([ax012.title, ax012.xaxis.label, ax012.yaxis.label] +
             ax012.get_xticklabels() + ax012.get_yticklabels()):
    item.set_fontsize(25)
ax012.legend(fancybox=True,loc=1)
ax012.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)



############################################
### PE, PN ###
fig10 = plt.figure(figsize=(8,8))
ax2  = fig10.add_subplot(111)
set_edges_black(ax2)

trig_freq     = 2 * np.pi * F_GERONO
xx1           =  A_GERONO * np.sin(trig_freq * t)
yy1           =  A_GERONO * np.sin(trig_freq * t) * np.cos(trig_freq * t)

plot_tf = 0#4.25
if ii*dt > plot_tf:
    new_ii = ii - int(plot_tf/dt)
else:
    new_ii = ii

ax2.plot(x[1:new_ii,0],x[1:new_ii,1],label='Actual',linewidth=lwidth+3,color=colors[ecc])
ax2.plot(xx1[1:new_ii],yy1[1:new_ii],label='Track',linewidth=lwidth,color=colors[1])
ax2.set(ylabel='Y',xlabel='X',title='Position Trajectories',ylim=[-5,5],xlim=[-5,5])#,title='Control Inputs')
for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(25)
ax2.legend(fancybox=True)
ax2.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)


################################
### Parameter Estimates Plot ###
# plt.close('all')
# T_fixed = 1.0

if False:

    fig6, ax4 = plt.subplots(3,1,figsize=(8,8))
    ax4[0].spines['bottom'].set_color('#000000')
    ax4[0].spines['top'].set_color('#000000')
    ax4[0].spines['right'].set_color('#000000')
    ax4[0].spines['left'].set_color('#000000')
    ax4[0].plot(T_fixed,theta[0],'gd',label='Fixed-Time',markersize=1)
    ax4[0].plot(t[:ii],-thetaMax[0]*np.ones((ii,)),label=r'$\theta _{1,bounds}$',linewidth=lwidth+4,color='k')
    ax4[0].plot(t[:ii],thetaMax[0]*np.ones((ii,)),linewidth=lwidth+4,color='k')
    ax4[0].plot(t[:ii],theta[0]*np.ones((ii,)),label=r'$\theta _{1,true}$',color='c',linewidth=lwidth,dashes=dash)
    # ax4[0].plot(t[:ii],np.clip(theta_hat[:ii,lsm,0],-10,10),label=r'$\hat\theta _{1,LSM}$',color=colors[lsm],linewidth=lwidth)
    # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,0],-10,10),':',label=r'$\hat\theta _{1,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
    # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,1],-10,10),'-.',label=r'$\hat\theta _{1,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
    ax4[0].plot(t[:ii],np.clip(theta_hat[:ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,PRO}$',linewidth=lwidth)
    ax4[0].legend(fancybox=True,markerscale=15)
    ax4[0].set(ylabel=r'$\theta _1$',xlim=[-0.1,ii*dt+0.75],ylim=[-thetaMax[0]-0.5,thetaMax[0]+0.5])
    ax4[0].set_xticklabels([])
    ax4[0].grid(True,linestyle='dotted',color='white')

    if ii*dt > 1.0:
        ax4a_inset = inset_axes(ax4[0],width="100%",height="100%",
                              bbox_to_anchor=(.1, .2, .4, .2),bbox_transform=ax4[0].transAxes, loc=3)
        ax4a_inset.spines['bottom'].set_color('#000000')
        ax4a_inset.spines['top'].set_color('#000000')
        ax4a_inset.spines['right'].set_color('#000000')
        ax4a_inset.spines['left'].set_color('#000000')
        ax4a_inset.plot(t[:ii],theta[0]*np.ones((ii,)),label=r'$\theta _{1,true}$',color='c',linewidth=lwidth,dashes=dash)
        # ax4a_inset.plot(t[:ii],np.clip(theta_hat[:ii,lsm,0],-10,10),label=r'$\theta _{1,LSM}$',color=colors[lsm],linewidth=lwidth)
        # ax4a_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,0,0],-10,10),':',label=r'$\hat\theta _{1,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
        # ax4a_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,0,1],-10,10),'-.',label=r'$\hat\theta _{1,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
        ax4a_inset.plot(t[:ii],np.clip(theta_hat[:ii,0],-thetaMax[0],thetaMax[0]),label=r'$\theta _{1,PRO}$',linewidth=lwidth)
        ax4a_inset.plot(T_fixed,theta[0],'gd',label='Fixed-Time',markersize=10)
        ax4a_inset.xaxis.set_major_locator(MaxNLocator(5))
        ax4a_inset.yaxis.set_major_locator(MaxNLocator(2))
        ax4a_inset.set_xlim(0.75,1.25)
        ax4a_inset.set_ylim(theta[0] - 0.1,theta[0] + 0.1)
        # ax4a_inset.xaxis.tick_top()
        for item in ([ax4a_inset.title, ax4a_inset.xaxis.label, ax4a_inset.yaxis.label] +
                     ax4a_inset.get_xticklabels() + ax4a_inset.get_yticklabels()):
            item.set_fontsize(12)
        ax4a_inset.set_xticks(ax4a_inset.get_xticks().tolist())
        # ax4a_inset.set_xticklabels(["    0.75",None,1.0,None,1.25,None])
        ax4a_inset.set_yticks(ax4a_inset.get_yticks().tolist())
        # ax4a_inset.set_yticklabels([theta[0]-0.1,None,theta[0],None,theta[0]+0.1])

        ax4a_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4a_inset.set_yticklabels([None,-1.00,-0.95])
        # ax4a_inset.set_yticklabels([None,-1.00,-0.9])
        # ax4a_inset.spines.set_edgecolor('black')
        # ax4a_inset.spines.set_linewidth(1)
        ax4a_inset.grid(True,linestyle='dotted',color='white')

        mark_inset(ax4[0],ax4a_inset,loc1=2,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")
        plt.draw()

    ax4[1].spines['bottom'].set_color('#000000')
    ax4[1].spines['top'].set_color('#000000')
    ax4[1].spines['right'].set_color('#000000')
    ax4[1].spines['left'].set_color('#000000')
    ax4[1].plot(T_fixed,theta[1],'gd',label='Fixed-Time',markersize=1)
    ax4[1].plot(t[:ii],-thetaMax[1]*np.ones((ii,)),label=r'$\theta _{2,bounds}$',linewidth=lwidth+4,color='k')
    ax4[1].plot(t[:ii],thetaMax[1]*np.ones((ii,)),linewidth=lwidth+4,color='k')
    ax4[1].plot(t[:ii],theta[1]*np.ones((ii,)),label=r'$\theta _{2,true}$',linewidth=lwidth,dashes=dash,color='c')
    # ax4[1].plot(t[:ii],np.clip(theta_hat[:ii,lsm,1],-10,10),label=r'$\hat\theta _{2,LSM}$',linewidth=lwidth,color=colors[lsm])
    # ax4[1].plot(t[:ii],np.clip(psi_hat[:ii,tay,1,0],-10,10),':',label=r'$\hat\theta _{2,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
    # ax4[1].plot(t[:ii],np.clip(psi_hat[:ii,tay,1,1],-10,10),'-.',label=r'$\hat\theta _{2,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
    ax4[1].plot(t[:ii],np.clip(theta_hat[:ii,1],-thetaMax[1],thetaMax[1]),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth,color=colors[ecc])
    ax4[1].legend(fancybox=True,markerscale=15)
    ax4[1].set(ylabel=r'$\theta _2$',xlim=[-0.1,ii*dt+0.75],ylim=[-thetaMax[1]-0.5,thetaMax[1]+0.5])
    ax4[1].set_xticklabels([])
    ax4[1].grid(True,linestyle='dotted',color='white')

    if ii*dt > 1.0:
        ax4b_inset = inset_axes(ax4[1],width="100%",height="100%",
                              bbox_to_anchor=(.1, .5, .4, .2),bbox_transform=ax4[1].transAxes, loc=3)
        ax4b_inset.spines['bottom'].set_color('#000000')
        ax4b_inset.spines['top'].set_color('#000000')
        ax4b_inset.spines['right'].set_color('#000000')
        ax4b_inset.spines['left'].set_color('#000000')
        ax4b_inset.plot(t[:ii],theta[1]*np.ones((ii,)),label=r'$\theta _{2,true}$',color='c',linewidth=lwidth,dashes=dash)
        ax4b_inset.plot(t[:ii],np.clip(theta_hat[:ii,1],-thetaMax[1],thetaMax[1]),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth)
        ax4b_inset.plot(T_fixed,theta[1],'gd',label='Fixed-Time',markersize=10)
        ax4b_inset.set_xlim(0.75,1.25)
        ax4b_inset.set_ylim(theta[1] - 0.1,theta[1] + 0.1)
        for item in ([ax4b_inset.title, ax4b_inset.xaxis.label, ax4b_inset.yaxis.label] +
                     ax4b_inset.get_xticklabels() + ax4b_inset.get_yticklabels()):
            item.set_fontsize(12)
        ax4b_inset.set_xticks(ax4b_inset.get_xticks().tolist())
        ax4b_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        ax4b_inset.xaxis.tick_top()
        ax4b_inset.set_yticks(ax4b_inset.get_yticks().tolist())
        ax4b_inset.set_yticklabels([theta[1]-0.1,None,theta[1],None,theta[1]+0.1])

        # ax4b_inset.set_yticklabels([None,None,1.00,None,1.10])
        # ax4b_inset.spines.set_edgecolor('black')
        # ax4b_inset.spines.set_linewidth(1)
        ax4b_inset.grid(True,linestyle='dotted',color='white')

        mark_inset(ax4[1],ax4b_inset,loc1=3,loc2=4,fc="none",ec="0.2",lw=1.5)#,ls="--")
        plt.draw()

    ax4[2].spines['bottom'].set_color('#000000')
    ax4[2].spines['top'].set_color('#000000')
    ax4[2].spines['right'].set_color('#000000')
    ax4[2].spines['left'].set_color('#000000')
    ax4[2].plot(T_fixed,theta[2],'gd',label='Fixed-Time',markersize=1)
    ax4[2].plot(t[:ii],-thetaMax[2]*np.ones((ii,)),label=r'$\theta _{3,bounds}$',linewidth=lwidth+4,color='k')
    ax4[2].plot(t[:ii],thetaMax[2]*np.ones((ii,)),linewidth=lwidth+4,color='k')
    ax4[2].plot(t[:ii],theta[2]*np.ones((ii,)),label=r'$\theta _{3,true}$',linewidth=lwidth,color='c',dashes=dash)
    ax4[2].plot(t[:ii],np.clip(theta_hat[:ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,PRO}$',linewidth=lwidth,color=colors[ecc])
    ax4[2].legend(fancybox=True,markerscale=15)
    ax4[2].set(xlabel='Time (sec)',ylabel=r'$\theta _3$',xlim=[-0.1,ii*dt+0.75],ylim=[-thetaMax[2]-0.5,thetaMax[2]+0.5])
    ax4[2].grid(True,linestyle='dotted',color='white')

    if ii*dt > 1.0:
        ax4c_inset = inset_axes(ax4[2],width="100%",height="100%",
                              bbox_to_anchor=(.2, .2, .4, .2),bbox_transform=ax4[2].transAxes, loc=3)
        ax4c_inset.spines['bottom'].set_color('#000000')
        ax4c_inset.spines['top'].set_color('#000000')
        ax4c_inset.spines['right'].set_color('#000000')
        ax4c_inset.spines['left'].set_color('#000000')
        ax4c_inset.plot(t[:ii],theta[2]*np.ones((ii,)),label=r'$\theta _{2,true}$',color='c',linewidth=lwidth,dashes=dash)
        ax4c_inset.plot(t[:ii],np.clip(theta_hat[:ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth)
        ax4c_inset.plot(T_fixed,theta[2],'gd',label='Fixed-Time',markersize=10)
        ax4c_inset.set_xlim(0.75,1.25)
        ax4c_inset.set_ylim(theta[2] - 0.1,theta[2] + 0.1)
        for item in ([ax4c_inset.title, ax4c_inset.xaxis.label, ax4c_inset.yaxis.label] +
                     ax4c_inset.get_xticklabels() + ax4c_inset.get_yticklabels()):
            item.set_fontsize(12)
        ax4c_inset.set_xticks(ax4c_inset.get_xticks().tolist())
        ax4c_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4c_inset.xaxis.tick_top()
        ax4c_inset.set_yticks(ax4c_inset.get_yticks().tolist())
        ax4c_inset.set_yticklabels([0.90,1.00,1.10])
        # ax4b_inset.spines.set_edgecolor('black')
        # ax4b_inset.spines.set_linewidth(1)
        ax4c_inset.grid(True,linestyle='dotted',color='white')


        mark_inset(ax4[2],ax4c_inset,loc1=2,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")
        plt.draw()
else:

    ################################
    ### Parameter Estimates Plot ###
    # plt.close('all')
    # T_fixed = 1.0
    c1 = 'r'; c2 = 'c'; c3 = 'm'
    # Teal, Coral, Khaki
    c1 = '#029386'; c2 = '#FC5A50'; c3 = '#AAA662'
    # Azure, Cyan, Indigo
    c4 = '#069AF3'; c5 = '#00FFFF'; c6 = '#4B0082'

    fig6, ax4 = plt.subplots(1,1,figsize=(8,8))
    ax4.spines['bottom'].set_color('#000000')
    ax4.spines['top'].set_color('#000000')
    ax4.spines['right'].set_color('#000000')
    ax4.spines['left'].set_color('#000000')
    ax4.plot(T_fixed,theta[0],'gd',label='Fixed-Time',markersize=1)
    ax4.plot(T_fixed,theta[0],'gd',markersize=10)
    ax4.plot(T_fixed,theta[1],'gd',markersize=10)
    ax4.plot(T_fixed,theta[2],'gd',markersize=10)
    #ax4.plot(t[:ii],-thetaMax[0]*np.ones((ii,)),label=r'$\theta _{bounds}$',linewidth=lwidth+4,color='k')
    #ax4.plot(t[:ii],thetaMax[0]*np.ones((ii,)),linewidth=lwidth+4,color='k')
    #ax4.plot(t[:ii],theta[0]*np.ones((ii,)),label=r'$\theta _{1,true}$',color=c1,linewidth=lwidth+3,dashes=dash)
    #ax4.plot(t[:ii],theta[1]*np.ones((ii,)),label=r'$\theta _{2,true}$',color=c2,linewidth=lwidth+3,dashes=dash)
    #ax4.plot(t[:ii],theta[2]*np.ones((ii,)),label=r'$\theta _{3,true}$',color=c3,linewidth=lwidth+3,dashes=dash)
    ax4.plot(t[:ii],disturb[:ii,3],label=r'$True (x)$',color=c1,linewidth=lwidth+3,dashes=dash)
    ax4.plot(t[:ii],disturb[:ii,4],label=r'$True (y)$',color=c2,linewidth=lwidth+3,dashes=dash)
    ax4.plot(t[:ii],disturb[:ii,5],label=r'$True (z)$',color=c3,linewidth=lwidth+3,dashes=dash)
    #ax4.plot(t[:ii],force_est[:ii,0],label=r'$Estimated (x)$',linewidth=lwidth+1,color=c4)
    #ax4.plot(t[:ii],force_est[:ii,1],label=r'$Estimated (y)$',linewidth=lwidth+1,color=c5)
    #ax4.plot(t[:ii],force_est[:ii,2],label=r'$Estimated (z)$',linewidth=lwidth+1,color=c6)
    ax4.plot(t[:ii],np.clip(theta_hat[:ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,PRO}$',linewidth=lwidth+1,color=c4)
    ax4.plot(t[:ii],np.clip(theta_hat[:ii,1],-thetaMax[1],thetaMax[1]),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth+1,color=c5)
    ax4.plot(t[:ii],np.clip(theta_hat[:ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,PRO}$',linewidth=lwidth+1,color=c6)
    # ax4.plot(t[:ii],np.clip(theta_inv[:ii,0],-thetaMax[0],thetaMax[0]),':',label=r'$\hat\theta _{1,PRO}$',linewidth=lwidth+1)#,color=c1)
    # ax4.plot(t[:ii],np.clip(theta_inv[:ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth+1)#,color=c2)
    # ax4.plot(t[:ii],np.clip(theta_inv[:ii,2],-thetaMax[2],thetaMax[2]),':',label=r'$\hat\theta _{3,PRO}$',linewidth=lwidth+1)#,color=c3)


    # ax4[0].plot(t[:ii],np.clip(theta_hat[:ii,lsm,0],-10,10),label=r'$\hat\theta _{1,LSM}$',color=colors[lsm],linewidth=lwidth)
    # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,0],-10,10),':',label=r'$\hat\theta _{1,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
    # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,1],-10,10),'-.',label=r'$\hat\theta _{1,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
    ax4.legend(fancybox=True,markerscale=15,fontsize=25)
    ax4.set(xlabel='Time (sec)',ylabel=r'$Disturbance$',xlim=[-0.1,ii*dt+4])#,ylim=[-thetaMax[0]-0.5,thetaMax[0]+0.5])
    for item in ([ax4.title, ax4.xaxis.label, ax4.yaxis.label] +
                     ax4.get_xticklabels() + ax4.get_yticklabels()):
            item.set_fontsize(25)
    # ax4.set_xticklabels([])
    ax4.grid(True,linestyle='dotted',color='white')

    if False:#ii*dt > 1.0:
        ax4a_inset = inset_axes(ax4,width="100%",height="100%",
                              bbox_to_anchor=(.45, .6, .3, .1),bbox_transform=ax4.transAxes, loc=3)
                              # bbox_to_anchor=(.35, .44, .3, .1),bbox_transform=ax4.transAxes, loc=3)
        ax4a_inset.spines['bottom'].set_color('#000000')
        ax4a_inset.spines['top'].set_color('#000000')
        ax4a_inset.spines['right'].set_color('#000000')
        ax4a_inset.spines['left'].set_color('#000000')
        ax4a_inset.plot(t[:ii],theta[0]*np.ones((ii,)),color=c1,linewidth=lwidth+3,dashes=dash)
        ax4a_inset.plot(t[:ii],np.clip(theta_hat[:ii,0],-thetaMax[0],thetaMax[0]),linewidth=lwidth+1,color=c4)
        # ax4a_inset.plot(t[:ii],np.clip(theta_inv[:ii,0],-thetaMax[0],thetaMax[0]),':',linewidth=lwidth-1)#,color=c1)
        ax4a_inset.plot(T_fixed,theta[0],'gd',label='Fixed-Time',markersize=10)
        ax4a_inset.xaxis.set_major_locator(MaxNLocator(5))
        ax4a_inset.yaxis.set_major_locator(MaxNLocator(2))
        ax4a_inset.set_xlim(T_fixed-0.1,T_fixed+0.1)
        ax4a_inset.set_ylim(theta[0] - 0.05,theta[0] + 0.05)
        ax4a_inset.xaxis.tick_top()
        ax4a_inset.yaxis.tick_right()
        for item in ([ax4a_inset.title, ax4a_inset.xaxis.label, ax4a_inset.yaxis.label] +
                     ax4a_inset.get_xticklabels() + ax4a_inset.get_yticklabels()):
            item.set_fontsize(18)
        ax4a_inset.set_xticks(ax4a_inset.get_xticks().tolist())
        # ax4a_inset.set_yticks(ax4a_inset.get_yticks().tolist())
        # ax4a_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4a_inset.set_yticklabels([None,np.round(theta[0]-0.05,2),None,None,theta[0],None,None,theta[0]+0.05,None])
        # ax4a_inset.get_yaxis().set_visible(False)
        ax4a_inset.grid(True,linestyle='dotted',color='white')
        mark_inset(ax4,ax4a_inset,loc1=4,loc2=2,fc="none",ec="0.2",lw=1.5)#,ls="--")

        ax4b_inset = inset_axes(ax4,width="100%",height="100%",
                              bbox_to_anchor=(.2, .2, .3, .1),bbox_transform=ax4.transAxes, loc=3)
        ax4b_inset.spines['bottom'].set_color('#000000')
        ax4b_inset.spines['top'].set_color('#000000')
        ax4b_inset.spines['right'].set_color('#000000')
        ax4b_inset.spines['left'].set_color('#000000')
        ax4b_inset.plot(t[:ii],theta[1]*np.ones((ii,)),color=c2,linewidth=lwidth+3,dashes=dash)
        ax4b_inset.plot(t[:ii],np.clip(theta_hat[:ii,1],-thetaMax[1],thetaMax[1]),linewidth=lwidth+1,color=c5)
        # ax4b_inset.plot(t[:ii],np.clip(theta_inv[:ii,1],-thetaMax[1],thetaMax[1]),':',linewidth=lwidth)#,color=c2)
        ax4b_inset.plot(T_fixed,theta[1],'gd',label='Fixed-Time',markersize=10)
        ax4b_inset.xaxis.set_major_locator(MaxNLocator(5))
        ax4b_inset.yaxis.set_major_locator(MaxNLocator(2))
        ax4b_inset.set_xlim(T_fixed-0.1,T_fixed+0.1)
        ax4b_inset.set_ylim(theta[1] - 0.05,theta[1] + 0.05)
        ax4b_inset.yaxis.tick_right()
        for item in ([ax4b_inset.title, ax4b_inset.xaxis.label, ax4b_inset.yaxis.label] +
                     ax4b_inset.get_xticklabels() + ax4b_inset.get_yticklabels()):
            item.set_fontsize(18)
        ax4b_inset.set_xticks(ax4b_inset.get_xticks().tolist())
        # ax4b_inset.set_yticks(ax4b_inset.get_yticks().tolist())
        # ax4b_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4b_inset.set_yticklabels([None,np.round(theta[1]-0.05,2),None,None,theta[1],None,None,theta[1]+0.05,None])
        # ax4b_inset.get_yaxis().set_visible(False)
        ax4b_inset.grid(True,linestyle='dotted',color='white')
        mark_inset(ax4,ax4b_inset,loc1=3,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")

        ax4c_inset = inset_axes(ax4,width="100%",height="100%",
                              bbox_to_anchor=(.1, .75, .3, .1),bbox_transform=ax4.transAxes, loc=3)
        ax4c_inset.spines['bottom'].set_color('#000000')
        ax4c_inset.spines['top'].set_color('#000000')
        ax4c_inset.spines['right'].set_color('#000000')
        ax4c_inset.spines['left'].set_color('#000000')
        ax4c_inset.plot(t[:ii],theta[2]*np.ones((ii,)),color=c3,linewidth=lwidth+3,dashes=dash)
        ax4c_inset.plot(t[:ii],np.clip(theta_hat[:ii,2],-thetaMax[2],thetaMax[2]),linewidth=lwidth+1,color=c6)
        # ax4c_inset.plot(t[:ii],np.clip(theta_inv[:ii,2],-thetaMax[2],thetaMax[2]),':',linewidth=lwidth)#,color=c3)
        ax4c_inset.plot(T_fixed,theta[2],'gd',label='Fixed-Time',markersize=10)
        ax4c_inset.xaxis.set_major_locator(MaxNLocator(5))
        ax4c_inset.yaxis.set_major_locator(MaxNLocator(2))
        ax4c_inset.set_xlim(T_fixed-0.1,T_fixed+0.1)
        ax4c_inset.set_ylim(theta[2] - 0.05,theta[2] + 0.05)
        ax4c_inset.xaxis.tick_top()
        ax4c_inset.yaxis.tick_right()
        for item in ([ax4c_inset.title, ax4c_inset.xaxis.label, ax4c_inset.yaxis.label] +
                     ax4c_inset.get_xticklabels() + ax4c_inset.get_yticklabels()):
            item.set_fontsize(18)
        ax4c_inset.set_xticks(ax4c_inset.get_xticks().tolist())
        # ax4c_inset.set_yticks(ax4c_inset.get_yticks().tolist())
        # ax4c_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4c_inset.set_yticklabels([None,np.round(theta[2]-0.05,2),None,None,theta[2],None,None,theta[2]+0.05,None])
        # ax4c_inset.get_yaxis().set_visible(False)
        ax4c_inset.grid(True,linestyle='dotted',color='white')
        mark_inset(ax4,ax4c_inset,loc1=2,loc2=4,fc="none",ec="0.2",lw=1.5)#,ls="--")

        plt.draw()

plt.show()

# plt.tight_layout(pad=1.0)

# fig6.savefig(filepath+"ShootTheGap_ThetaHats_RegX.eps",bbox_inches='tight',dpi=300)
# fig6.savefig(filepath+"ShootTheGap_ThetaHats_RegX.png",bbox_inches='tight',dpi=300)


# %matplotlib notebook
import project_path
import pickle
import traceback
import numpy as np
import gurobipy as gp
# import latex
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
import pandas as pd

from quadrotor_settings import U_MAX, XS, k1, k2, arm_length, f_max, d_max, tx_max, ty_max, tz_max
from ecc_controller_test import T_e

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

filepath = '/home/dasc/MB/Code/FxT_AdaptationLaw_ParametricUncertainty/simdata/quadrotor/'

tf       = 5.0
dt       = 1e-4 # Timestep (sec)
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
psi_hat   = np.zeros(((int(tf/dt)+1),nControl,2,2))
cbf_val   = np.zeros(((int(tf/dt)+1),nControl,2))
acbf_val  = np.zeros(((int(tf/dt)+1),nControl,2))
clf_val   = np.zeros(((int(tf/dt)+1),nControl))
aclf_val  = np.zeros(((int(tf/dt)+1),nControl))
dclf_val  = np.zeros(((int(tf/dt)+1),nControl,2))
d0_val    = np.zeros(((int(tf/dt)+1),nControl))
set1_vals = np.zeros(((int(tf/dt)+1),nControl,2))
set2_vals = np.zeros(((int(tf/dt)+1),nControl,2))
dhat_vals = np.zeros(((int(tf/dt)+1),nControl,2))
xdot_vals = np.zeros(((int(tf/dt)+1),nControl,2))
xfdot     = np.zeros(((int(tf/dt)+1),nControl,2))
safety    = np.zeros(((int(tf/dt)+1),nControl,2))
Vmax      = np.zeros(((int(tf/dt)+1),nControl))

filename = filepath + 'CODE_DEBUGGING.pkl'


with open(filename,'rb') as f:
    try:
        data      = pickle.load(f)
        x         = data['x']
        p         = data['p']
        theta     = data['theta']
        sols      = data['sols']
        theta_hat = data['theta_hat']
        cbf_val   = data['cbf']
        clf_val   = data['clf']
        Vmax      = data['Vmax']
        psi_hat   = data['psi_hat']
        ii        = data['ii']
    except:
        traceback.print_exc()
        
lwidth = 5
dash = [3,2]

    
ii = np.min([int(tf/dt),ii])

def set_edges_black(ax):
    ax.spines['bottom'].set_color('#000000')
    ax.spines['top'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_color('#000000')

plt.close('all')

#####################
### CLF Evolution ###
# fig1 = plt.figure(figsize=(7,7))

# ax1  = fig1.add_subplot(111)
# ax1.spines['bottom'].set_color('#000000')
# ax1.spines['top'].set_color('#000000')
# ax1.spines['right'].set_color('#000000')
# ax1.spines['left'].set_color('#000000')

# ax1.plot(t[1:ii],clf_val[1:ii,ecc],dashes=dash,label='PRO',linewidth=lwidth,color=colors[ecc])
# ax1.legend(fancybox=True)
# ax1.set(xlabel='Time (sec)',ylabel='V(x)')
# ax1.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=0.5)
# # plt.show()

#######################
### Delta Evolution ###
# fig2 = plt.figure(figsize=(7,7))

# ax1  = fig2.add_subplot(111)
# ax1.spines['bottom'].set_color('#000000')
# ax1.spines['top'].set_color('#000000')
# ax1.spines['right'].set_color('#000000')
# ax1.spines['left'].set_color('#000000')

# ax1.plot(t[1:ii],sols[1:ii,ecc,5],label='d1',linewidth=lwidth,color=colors[ecc+1])
# ax1.plot(t[1:ii],sols[1:ii,ecc,6],label='d2',linewidth=lwidth,color=colors[ecc+2])
# ax1.plot(t[1:ii],sols[1:ii,ecc,4],label='d0',linewidth=lwidth,color=colors[ecc])
# ax1.legend(fancybox=True)
# ax1.set(xlabel='Time (sec)',ylabel='Deltas')
# ax1.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=0.5)
# # plt.show()

############################################
### State, Control, and CBF Trajectories ###
# plt.close('all')
# fig3 = plt.figure(figsize=(8,8))
# # grid = plt.GridSpec(2,3,hspace=0.2,wspace=0.2)

# ax1  = fig3.add_subplot(111)
# ax1.spines['bottom'].set_color('#000000')
# ax1.spines['top'].set_color('#000000')
# ax1.spines['right'].set_color('#000000')
# ax1.spines['left'].set_color('#000000')


# xx1   = np.linspace(OBS_X1-OBS_R,OBS_X1+OBS_R,1000)
# yy1a  = OBS_Y1 + np.sqrt(OBS_R**2 * (1 - ((xx1 - OBS_X1)/OBS_R)**2))
# yy1b  = OBS_Y1 - np.sqrt(OBS_R**2 * (1 - ((xx1 - OBS_X1)/OBS_R)**2))

# xx2   = np.linspace(OBS_X2-OBS_R,OBS_X2+OBS_R,1000)
# yy2a  = OBS_Y2 + np.sqrt(OBS_R**2 * (1 - ((xx2 - OBS_X2)/OBS_R)**2))
# yy2b  = OBS_Y2 - np.sqrt(OBS_R**2 * (1 - ((xx2 - OBS_X2)/OBS_R)**2))

# ax1.plot(xx1,yy1a,color='k',linewidth=lwidth+2)
# ax1.plot(xx1,yy1b,color='k',linewidth=lwidth+2)
# ax1.plot(xx1,yy2a,color='k',linewidth=lwidth+2)
# ax1.plot(xx1,yy2b,color='k',linewidth=lwidth+2,label='Barrier')
# ax1.plot(0,0,'o',markersize=20,color='r')
# ax1.plot(0,0,'o',markersize=10,color='w')
# ax1.plot(0,0,'o',markersize=5,color='r',label='Goal')
# ax1.plot(5,0,'*',markersize=20,color='b',label=r'$z_0$')
# # ax1.plot(x[:ii,tay,0],x[:ii,tay,1],label='TAY',linewidth=lwidth,color=colors[tay])
# # ax1.plot(x[:ii,bla,0],x[:ii,bla,1],label='BLA',linewidth=lwidth,color=colors[bla])
# # ax1.plot(x[:ii,lop,0],x[:ii,lop,1],label='LOP',linewidth=lwidth,color=colors[lop])
# # ax1.plot(x[:ii,lsm,0],x[:ii,lsm,1],label='LSM',linewidth=lwidth,color=colors[lsm])
# # ax1.plot(x[:ii,zha,0],x[:ii,zha,1],label='ZHA',linewidth=lwidth,color=colors[zha])
# ax1.plot(x[:ii,ecc,0],x[:ii,ecc,1],label='PRO',linewidth=3,color=colors[ecc])
# for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
#              ax1.get_xticklabels() + ax1.get_yticklabels()):
#     item.set_fontsize(25)
# ax1.legend(fancybox=True)
# ax1.set(xlabel=r'$x$')#,title='State Trajectories')
# ax1.set_ylabel(r'$y$',rotation=90)
# ax1.set_ylim([-2.5,2.5])
# ax1.grid(True,linestyle='dotted',color='white')

# ax1inset = inset_axes(ax1,width="100%",height="100%",
#                       bbox_to_anchor=(.45, .05, .5, .35),bbox_transform=ax1.transAxes, loc=3)
# ax1inset.spines['bottom'].set_color('#000000')
# ax1inset.spines['top'].set_color('#000000')
# ax1inset.spines['right'].set_color('#000000')
# ax1inset.spines['left'].set_color('#000000')
# ax1inset.plot(xx1,yy1a,color='k',linewidth=5)
# ax1inset.plot(xx1,yy1b,color='k',linewidth=5)
# ax1inset.plot(xx1,yy2a,color='k',linewidth=5)
# ax1inset.plot(xx1,yy2b,color='k',linewidth=5)
# # ax1inset.plot(x[:ii,tay,0],x[:ii,tay,1],label='TAY',linewidth=lwidth,color=colors[tay])
# # ax1inset.plot(x[:ii,bla,0],x[:ii,bla,1],label='BLA',linewidth=lwidth,color=colors[bla])
# # ax1inset.plot(x[:ii,lop,0],x[:ii,lop,1],label='LOP',linewidth=lwidth,color=colors[lop])
# # ax1inset.plot(x[:ii,lsm,0],x[:ii,lsm,1],label='LSM',linewidth=lwidth,color=colors[lsm])
# # ax1inset.plot(x[:ii,zha,0],x[:ii,zha,1],label='ZHA',linewidth=lwidth,color=colors[zha])
# ax1inset.plot(x[:ii,ecc,0],x[:ii,ecc,1],dashes=dash,label='PRO',linewidth=3,color=colors[ecc])
# ax1inset.set_xlim(0.75,1.25)
# ax1inset.set_ylim(-1.05,-0.95)
# ax1inset.set_xticklabels([])
# ax1inset.set_yticklabels([])
# # ax3inset.spines.set_edgecolor('black')
# # ax3inset.spines.set_linewidth(1)
# ax1inset.grid(True,linestyle='dotted',color='white')
# mark_inset(ax1,ax1inset,loc1=2,loc2=3,fc="none",ec="0.2",lw=1.5)#,ls="--")

# plt.tight_layout(pad=2.0)




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
ax2a.plot(t[1:ii],f_max*np.ones(t[1:ii].shape),label=r'$F_{max}$',linewidth=lwidth,color='k')
ax2a.plot(t[1:ii],0.0*np.ones(t[1:ii].shape),linewidth=lwidth,color='k',label=r'$F_{min}$')
ax2a.plot(t[:ii],x[:ii,ecc,12],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2a.set(ylabel=r'$F$',ylim=[-0.5,f_max*1.1],title='Control Inputs')#,xlim=[-0.1,5.2],
ax2b.plot(t[1:ii],-tx_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\phi}$',linewidth=lwidth,color='k')
ax2b.plot(t[1:ii], tx_max*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2b.plot(t[:ii],arm_length*k1*(x[:ii,ecc,15] - x[:ii,ecc,13]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2b.plot(t[:ii],sols[:ii,ecc,1],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2b.set(ylabel=r'$\mu$')#xlim=[-0.1,5.2],)
ax2b.set(ylabel=r'$\tau_{\phi}$',ylim=[-1.1*tx_max,1.1*tx_max])#xlim=[-0.1,5.2],
ax2c.plot(t[1:ii],-ty_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\theta}$',linewidth=lwidth,color='k')
ax2c.plot(t[1:ii], ty_max*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2c.plot(t[:ii],arm_length*k1*(x[:ii,ecc,12] - x[:ii,ecc,14]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2c.plot(t[:ii],sols[:ii,ecc,2],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2c.set(ylabel=r'$\tau_{\theta}$',ylim=[-1.1*ty_max,1.1*ty_max])#xlim=[-0.1,5.2],
ax2d.plot(t[1:ii],-tz_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\psi}$',linewidth=lwidth,color='k')
ax2d.plot(t[1:ii], tz_max*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2d.plot(t[:ii],k2*(-x[:ii,ecc,12]+x[:ii,ecc,13]-x[:ii,ecc,14]+x[:ii,ecc,15]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2d.plot(t[:ii],sols[:ii,ecc,2],label='PRO',linewidth=lwidth,color=colors[ecc])
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
fig5 = plt.figure(figsize=(8,8))
ax3  = fig5.add_subplot(111)
set_edges_black(ax3)
ax3.plot(t[1:ii],np.zeros(t[1:ii].shape),label=r'Boundary',linewidth=lwidth,color='k')
ax3.plot(t[1:ii],cbf_val[1:ii,ecc,0],label=r'$PRO1$',linewidth=lwidth,color=colors[ecc])
ax3.plot(t[1:ii],cbf_val[1:ii,ecc,1],label=r'$PRO2$',linewidth=lwidth,color=colors[ecc+1])
# ax3.plot(t[1:ii],cbf_val[1:ii,ecc,1],label=r'$PRO2$',linewidth=lwidth,color=colors[ecc+1])
# ax3.plot(t[1:ii],cbf_val[1:ii,ecc,2],label=r'$PRO3$',linewidth=lwidth,color=colors[ecc+2])
ax3.set(xlabel='Time (sec)',ylabel='h(x)',title='CBF Trajectory')
ax3.set(xlim=[-0.5,6.5],ylim=[-1,20])
for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
             ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(25)
ax3.legend(fancybox=True,loc=1)
ax3.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)




############################################
### PE, PN, H Trajectories ###
fig10 = plt.figure(figsize=(8,8))
ax2a  = fig10.add_subplot(311)
set_edges_black(ax2a)
ax2a.plot(t[:ii],x[:ii,ecc,0],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2a.plot(t[:ii],XS(t[:ii])[0],label='SPHERE',linewidth=lwidth,color=colors[ecc+1])
ax2a.set(ylabel='PN',title='Position Trajectories')#,title='Control Inputs')
for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
             ax2a.get_xticklabels() + ax2a.get_yticklabels()):
    item.set_fontsize(25)
ax2a.legend(fancybox=True)
ax2a.grid(True,linestyle='dotted',color='white')

ax2b  = fig10.add_subplot(312)
set_edges_black(ax2b)
ax2b.plot(t[:ii],x[:ii,ecc,1],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2b.plot(t[:ii],XS(t[:ii])[1],label='SPHERE',linewidth=lwidth,color=colors[ecc+1])
ax2b.set(ylabel='PE')
for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
             ax2b.get_xticklabels() + ax2b.get_yticklabels()):
    item.set_fontsize(25)
ax2b.legend(fancybox=True)
ax2b.grid(True,linestyle='dotted',color='white')

ax2c  = fig10.add_subplot(313)
set_edges_black(ax2c)
ax2c.plot(t[:ii],x[:ii,ecc,2],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2c.plot(t[:ii],XS(t[:ii])[2],label='SPHERE',linewidth=lwidth,color=colors[ecc+1])
ax2c.set(xlabel='Time (sec)',ylabel='H')
for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
             ax2c.get_xticklabels() + ax2c.get_yticklabels()):
    item.set_fontsize(25)
ax2c.legend(fancybox=True)
ax2c.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)




# ############################################
# ### phi, theta, psi Trajectories ###
# fig100 = plt.figure(figsize=(8,8))
# ax2a  = fig100.add_subplot(311)
# set_edges_black(ax2a)
# ax2a.plot(t[:ii],x[:ii,ecc,6],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2a.set(ylabel=r'$\phi$')#,title='Control Inputs')
# for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
#              ax2a.get_xticklabels() + ax2a.get_yticklabels()):
#     item.set_fontsize(25)
# ax2a.legend(fancybox=True)
# ax2a.grid(True,linestyle='dotted',color='white')

# ax2b  = fig100.add_subplot(312)
# set_edges_black(ax2b)
# ax2b.plot(t[:ii],x[:ii,ecc,7],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2b.set(xlabel='Time (sec)',ylabel=r'$\theta$')
# for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
#              ax2b.get_xticklabels() + ax2b.get_yticklabels()):
#     item.set_fontsize(25)
# ax2b.legend(fancybox=True)
# ax2b.grid(True,linestyle='dotted',color='white')

# ax2c  = fig100.add_subplot(313)
# set_edges_black(ax2c)
# ax2c.plot(t[:ii],x[:ii,ecc,8],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2c.set(xlabel='Time (sec)',ylabel=r'$\psi$')
# for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
#              ax2c.get_xticklabels() + ax2c.get_yticklabels()):
#     item.set_fontsize(25)
# ax2c.legend(fancybox=True)
# ax2c.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=2.0)



#################################################
### XYZ in Sphere Plots ###
xx1   = np.linspace(-1,1,1000)
yy1a  = np.sqrt(1 - xx1**2)
yy1b  = -np.sqrt(1 - xx1**2)

fig101 = plt.figure(figsize=(8,8))
ax2a  = fig101.add_subplot(311)
set_edges_black(ax2a)
ax2a.plot(x[:ii,ecc,0] - XS(t[:ii])[0],x[:ii,ecc,1] - XS(t[:ii])[1],label='XY',linewidth=lwidth,color=colors[ecc])
ax2a.plot(xx1,yy1a,linewidth=lwidth,color='k')
ax2a.plot(xx1,yy1b,linewidth=lwidth,color='k')
ax2a.set(xlabel='X',ylabel='Y')#,title='Control Inputs')
for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
             ax2a.get_xticklabels() + ax2a.get_yticklabels()):
    item.set_fontsize(25)
ax2a.legend(fancybox=True)
ax2a.grid(True,linestyle='dotted',color='white')

ax2b  = fig101.add_subplot(312)
set_edges_black(ax2b)
ax2b.plot(x[:ii,ecc,2] - XS(t[:ii])[2],x[:ii,ecc,0] - XS(t[:ii])[0],label='ZX',linewidth=lwidth,color=colors[ecc])
ax2b.plot(xx1,yy1a,linewidth=lwidth,color='k')
ax2b.plot(xx1,yy1b,linewidth=lwidth,color='k')
ax2b.set(xlabel='Z',ylabel='X')
for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
             ax2b.get_xticklabels() + ax2b.get_yticklabels()):
    item.set_fontsize(25)
ax2b.legend(fancybox=True)
ax2b.grid(True,linestyle='dotted',color='white')

ax2c  = fig101.add_subplot(313)
set_edges_black(ax2c)
ax2c.plot(x[:ii,ecc,1] - XS(t[:ii])[1],x[:ii,ecc,2] - XS(t[:ii])[2],label='YZ',linewidth=lwidth,color=colors[ecc])
ax2c.plot(xx1,yy1a,linewidth=lwidth,color='k')
ax2c.plot(xx1,yy1b,linewidth=lwidth,color='k')
ax2c.set(xlabel='Y',ylabel='Z')
for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
             ax2c.get_xticklabels() + ax2c.get_yticklabels()):
    item.set_fontsize(25)
ax2c.legend(fancybox=True)
ax2c.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)



#################################################
### U,V,W Plots ###
fig11 = plt.figure(figsize=(8,8))
ax2a  = fig11.add_subplot(311)
set_edges_black(ax2a)
ax2a.plot(t[:ii],x[:ii,ecc,3],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2a.set(ylabel='u',title='Body-Fixed Velocities')
for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
             ax2a.get_xticklabels() + ax2a.get_yticklabels()):
    item.set_fontsize(25)
ax2a.legend(fancybox=True)
ax2a.grid(True,linestyle='dotted',color='white')

ax2b  = fig11.add_subplot(312)
set_edges_black(ax2b)
ax2b.plot(t[:ii],x[:ii,ecc,4],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2b.set(ylabel='v')
for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
             ax2b.get_xticklabels() + ax2b.get_yticklabels()):
    item.set_fontsize(25)
ax2b.legend(fancybox=True)
ax2b.grid(True,linestyle='dotted',color='white')

ax2c  = fig11.add_subplot(313)
set_edges_black(ax2c)
ax2c.plot(t[:ii],x[:ii,ecc,5],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2c.set(xlabel='Time (sec)',ylabel='w')
for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
             ax2c.get_xticklabels() + ax2c.get_yticklabels()):
    item.set_fontsize(25)
ax2c.legend(fancybox=True)
ax2c.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)




# #######################
# ### Delta Evolution ###
# fig2 = plt.figure(figsize=(7,7))

# ax1  = fig2.add_subplot(111)
# set_edges_black(ax1)


# ax1.plot(t[1:ii],sols[1:ii,ecc,5],label='p1',linewidth=lwidth,color=colors[ecc])
# # ax1.plot(t[1:ii],sols[1:ii,ecc,6],label='p2',linewidth=lwidth,color=colors[ecc+1])
# # ax1.plot(t[1:ii],sols[1:ii,ecc,7],label='p3',linewidth=lwidth,color=colors[ecc+2])
# ax1.legend(fancybox=True)
# ax1.set(xlabel='Time (sec)',ylabel='Deltas')
# ax1.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=0.5)
# plt.show()




# fig5 = plt.figure(figsize=(8,8))
# ax3  = fig5.add_subplot(111)
# set_edges_black(ax3)
# ax3.plot(t[1:ii],np.min(cbf_val[1:ii,ecc],axis=-1),dashes=dash,label=r'$PRO$',linewidth=lwidth,color=colors[ecc])
# ax3.set(xlabel='Time (sec)',ylabel='h(x)')#,title='CBFs')
# ax3.set(xlim=[-0.5,6.5],ylim=[-1,20])
# for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
#              ax3.get_xticklabels() + ax3.get_yticklabels()):
#     item.set_fontsize(25)
# ax3.legend(fancybox=True,loc=1)
# ax3.grid(True,linestyle='dotted',color='white')

# ax3inset = inset_axes(ax3,width="100%",height="100%",
#                       bbox_to_anchor=(.45, .1, .5, .35),bbox_transform=ax3.transAxes, loc=3)
# ax3inset.spines['bottom'].set_color('#000000')
# ax3inset.spines['top'].set_color('#000000')
# ax3inset.spines['right'].set_color('#000000')
# ax3inset.spines['left'].set_color('#000000')
# ax3inset.plot(t[1:ii],np.zeros(t[1:ii].shape),label=r'Boundary',linewidth=lwidth,color='k')
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,tay],axis=-1),label=r'$TAY$',linewidth=3,color=colors[tay])
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,bla],axis=-1),label=r'$BLA$',linewidth=3,color=colors[bla])
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,lop],axis=-1),label=r'$LOP$',linewidth=3,color=colors[lop])
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,lsm],axis=-1),label=r'$LSM$',linewidth=3,color=colors[lsm])
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,zha],axis=-1),label=r'$ZHA$',linewidth=3,color=colors[zha])
# ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,ecc],axis=-1),dashes=dash,label=r'$PRO$',linewidth=3,color=colors[ecc])
# ax3inset.set_xlim(0.75,4.1)
# ax3inset.set_ylim(-0.01,0.18)
# ax3inset.set_xticklabels([])
# ax3inset.set_yticklabels([0.00,0.00,0.05,0.10])
# # ax3inset.spines.set_edgecolor('black')
# # ax3inset.spines.set_linewidth(1)
# ax3inset.grid(True,linestyle='dotted',color='white')
# mark_inset(ax3,ax3inset,loc1=2,loc2=4,fc="none",ec="0.2",lw=1.5)#,ls="--")
# plt.draw()

# plt.tight_layout(pad=2.0)

# fig3.savefig(filepath+"ShootTheGap_allFxTS_Trajectories_RegX.eps",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig1.savefig(filepath+"ShootTheGap_allFxTS_Trajectories_RegX.png",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig4.savefig(filepath+"ShootTheGap_allFxTS_Controls_RegX.eps",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig2.savefig(filepath+"ShootTheGap_allFxTS_Controls_RegX.png",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig5.savefig(filepath+"ShootTheGap_allFxTS_CBFs_RegX.eps",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig3.savefig(filepath+"ShootTheGap_allFxTS_CBFs_RegX.png",bbox_inches='tight',dpi=300,pad_inches=0.5)


# ################################
# ### Parameter Estimates Plot ###
# # plt.close('all')
# T_fixed = 0.149

# fig6, ax4 = plt.subplots(2,1,figsize=(8,8))
# ax4[0].spines['bottom'].set_color('#000000')
# ax4[0].spines['top'].set_color('#000000')
# ax4[0].spines['right'].set_color('#000000')
# ax4[0].spines['left'].set_color('#000000')
# ax4[0].plot(0.2,-1,'gd',label='Fixed-Time',markersize=1)
# ax4[0].plot(t[:ii],-10*np.ones((ii,)),label=r'$\theta _{1,bounds}$',linewidth=lwidth+4,color='k')
# ax4[0].plot(t[:ii],10*np.ones((ii,)),linewidth=lwidth+4,color='k')
# ax4[0].plot(t[:ii],theta[0]*np.ones((ii,)),label=r'$\theta _{1,true}$',color='c',linewidth=lwidth+4)
# # ax4[0].plot(t[:ii],np.clip(theta_hat[:ii,lsm,0],-10,10),label=r'$\hat\theta _{1,LSM}$',color=colors[lsm],linewidth=lwidth)
# # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,0],-10,10),':',label=r'$\hat\theta _{1,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
# # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,1],-10,10),'-.',label=r'$\hat\theta _{1,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
# ax4[0].plot(t[:ii],np.clip(theta_hat[:ii,ecc,0],-10,10),label=r'$\hat\theta _{1,PRO}$',linewidth=lwidth)
# ax4[0].legend(fancybox=True,markerscale=15)
# ax4[0].set(ylabel=r'$\theta _1$',xlim=[-0.1,5.75],ylim=[-11,11])
# ax4[0].set_xticklabels([])
# ax4[0].grid(True,linestyle='dotted',color='white')

# ax4a_inset = inset_axes(ax4[0],width="100%",height="100%",
#                       bbox_to_anchor=(.1, .6, .4, .2),bbox_transform=ax4[0].transAxes, loc=3)
# ax4a_inset.spines['bottom'].set_color('#000000')
# ax4a_inset.spines['top'].set_color('#000000')
# ax4a_inset.spines['right'].set_color('#000000')
# ax4a_inset.spines['left'].set_color('#000000')
# ax4a_inset.plot(t[:ii],theta[0]*np.ones((ii,)),label=r'$\theta _{1,true}$',color='c',linewidth=lwidth+4)
# # ax4a_inset.plot(t[:ii],np.clip(theta_hat[:ii,lsm,0],-10,10),label=r'$\theta _{1,LSM}$',color=colors[lsm],linewidth=lwidth)
# # ax4a_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,0,0],-10,10),':',label=r'$\hat\theta _{1,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
# # ax4a_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,0,1],-10,10),'-.',label=r'$\hat\theta _{1,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
# ax4a_inset.plot(t[:ii],np.clip(theta_hat[:ii,ecc,0],-10,10),label=r'$\theta _{1,PRO}$',linewidth=lwidth)
# ax4a_inset.plot(T_fixed,-1,'gd',label='Fixed-Time',markersize=15)
# ax4a_inset.set_xlim(0.0,0.25)
# ax4a_inset.set_ylim(-1.05,-0.95)
# ax4a_inset.set_ylim(-1.1,-0.9)
# ax4a_inset.xaxis.tick_top()
# for item in ([ax4a_inset.title, ax4a_inset.xaxis.label, ax4a_inset.yaxis.label] +
#              ax4a_inset.get_xticklabels() + ax4a_inset.get_yticklabels()):
#     item.set_fontsize(12)
# ax4a_inset.set_xticks(ax4a_inset.get_xticks().tolist())
# ax4a_inset.set_xticklabels(["     0.0",None,0.1,None,0.2,None])
# ax4a_inset.set_yticks(ax4a_inset.get_yticks().tolist())
# ax4a_inset.set_yticklabels([None,None,-1.00,None,-0.90])

# # ax4a_inset.set_xticklabels(["     0.0",0.1,0.2,0.3])
# # ax4a_inset.set_yticklabels([None,-1.00,-0.95])
# # ax4a_inset.set_yticklabels([None,-1.00,-0.9])
# # ax4a_inset.spines.set_edgecolor('black')
# # ax4a_inset.spines.set_linewidth(1)
# ax4a_inset.grid(True,linestyle='dotted',color='white')
# mark_inset(ax4[0],ax4a_inset,loc1=3,loc2=4,fc="none",ec="0.2",lw=1.5)#,ls="--")
# plt.draw()

# ax4[1].spines['bottom'].set_color('#000000')
# ax4[1].spines['top'].set_color('#000000')
# ax4[1].spines['right'].set_color('#000000')
# ax4[1].spines['left'].set_color('#000000')
# ax4[1].plot(T_fixed,1,'gd',label='Fixed-Time',markersize=1)
# ax4[1].plot(t[:ii],-10*np.ones((ii,)),label=r'$\theta _{2,bounds}$',linewidth=lwidth+4,color='k')
# ax4[1].plot(t[:ii],10*np.ones((ii,)),linewidth=lwidth+4,color='k')
# ax4[1].plot(t[:ii],theta[1]*np.ones((ii,)),label=r'$\theta _{2,true}$',linewidth=lwidth+4,color='c')
# # ax4[1].plot(t[:ii],np.clip(theta_hat[:ii,lsm,1],-10,10),label=r'$\hat\theta _{2,LSM}$',linewidth=lwidth,color=colors[lsm])
# # ax4[1].plot(t[:ii],np.clip(psi_hat[:ii,tay,1,0],-10,10),':',label=r'$\hat\theta _{2,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
# # ax4[1].plot(t[:ii],np.clip(psi_hat[:ii,tay,1,1],-10,10),'-.',label=r'$\hat\theta _{2,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
# ax4[1].plot(t[:ii],np.clip(theta_hat[:ii,ecc,1],-10,10),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth,color=colors[ecc])
# ax4[1].legend(fancybox=True,markerscale=15)
# ax4[1].set(xlabel='Time (sec)',ylabel=r'$\theta _2$',xlim=[-0.1,5.75],ylim=[-11,11])
# ax4[1].grid(True,linestyle='dotted',color='white')

# ax4b_inset = inset_axes(ax4[1],width="100%",height="100%",
#                       bbox_to_anchor=(.1, .13, .4, .2),bbox_transform=ax4[1].transAxes, loc=3)
# ax4b_inset.spines['bottom'].set_color('#000000')
# ax4b_inset.spines['top'].set_color('#000000')
# ax4b_inset.spines['right'].set_color('#000000')
# ax4b_inset.spines['left'].set_color('#000000')
# ax4b_inset.plot(t[:ii],theta[1]*np.ones((ii,)),label=r'$\theta _{2,true}$',color='c',linewidth=lwidth+4)
# # ax4b_inset.plot(t[:ii],np.clip(theta_hat[:ii,lsm,1],-10,10),label=r'$\hat\theta _{2,LSM}$',color=colors[lsm],linewidth=lwidth)
# # ax4b_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,1,0],-10,10),':',label=r'$\hat\theta _{2,TAY,h_1}$',color=colors[tay],linewidth=lwidth)
# # ax4b_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,1,1],-10,10),'-.',label=r'$\hat\theta _{2,TAY,h_2}$',color=colors[tay],linewidth=lwidth)
# ax4b_inset.plot(t[:ii],np.clip(theta_hat[:ii,ecc,1],-10,10),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth)
# ax4b_inset.plot(T_fixed,1,'gd',label='Fixed-Time',markersize=15)
# ax4b_inset.set_xlim(0.0,0.25)
# ax4b_inset.set_ylim(0.95,1.05)
# ax4b_inset.set_ylim(0.9,1.1)
# for item in ([ax4b_inset.title, ax4b_inset.xaxis.label, ax4b_inset.yaxis.label] +
#              ax4b_inset.get_xticklabels() + ax4b_inset.get_yticklabels()):
#     item.set_fontsize(12)
# ax4b_inset.set_xticks(ax4b_inset.get_xticks().tolist())
# ax4b_inset.set_xticklabels(["     0.0",None,0.1,None,0.2,None])
# ax4b_inset.set_yticks(ax4b_inset.get_yticks().tolist())
# ax4b_inset.set_yticklabels([None,None,1.00,None,1.10])
# # ax4b_inset.spines.set_edgecolor('black')
# # ax4b_inset.spines.set_linewidth(1)
# ax4b_inset.grid(True,linestyle='dotted',color='white')
# mark_inset(ax4[1],ax4b_inset,loc1=2,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")

# plt.draw()
# plt.show()
# # plt.tight_layout(pad=1.0)

# fig6.savefig(filepath+"ShootTheGap_ThetaHats_RegX.eps",bbox_inches='tight',dpi=300)
# fig6.savefig(filepath+"ShootTheGap_ThetaHats_RegX.png",bbox_inches='tight',dpi=300)

plt.show()
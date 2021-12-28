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

filepath = '/home/dasc/MB/datastore/fxt_adaptation/quadrotor/money_sets/DragCoefficient_Simple/'
filepath2 = '/home/dasc/MB/datastore/fxt_adaptation/quadrotor/money_sets/WindEstimation/'

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
    #filename = filepath + 'money_set/MONEY_DATASET.pkl'

    with open(filepath + 'NONE_CD.pkl','rb') as f:
        try:
            data        = pickle.load(f)
            xNon        = data['x']
            solsNon     = data['sols']
            solsNomNon  = data['sols_nom']
        except:
            traceback.print_exc()

    with open(filepath + 'ESTIMATOR_CD.pkl','rb') as f:
        try:
            data        = pickle.load(f)
            theta       = data['theta']
            xEst        = data['x']
            solsEst     = data['sols']
            solsNomEst  = data['sols_nom']
            thetaHatEst = data['thetahat']
            minSigmaEst = data['minSigma']
            GammaEst    = data['Gamma']
            iiEst       = data['ii']
        except:
            traceback.print_exc()

    # with open(filepath + 'FILTER_CD.pkl','rb') as f:
    #     try:
    #         data        = pickle.load(f)
    #         xFil        = data['x']
    #         solsFil     = data['sols']
    #         solsNomFil  = data['sols_nom']
    #         thetaHatFil = data['thetahat']
    #         iiFil       = data['ii']
    #     except:
    #         traceback.print_exc()

    with open(filepath + 'RATE_CD.pkl','rb') as f:
        try:
            data        = pickle.load(f)
            xRat        = data['x']
            solsRat     = data['sols']
            solsNomRat  = data['sols_nom']
            thetaHatRat = data['thetahat']
            minSigmaRat = data['minSigma']
            GammaRat    = data['Gamma']
            iiRat       = data['ii']
        except:
            traceback.print_exc()

    with open(filepath2 + 'RATE_WIND.pkl','rb') as f:
        try:
            data        = pickle.load(f)
            xRatw        = data['x']
            solsRatw     = data['sols']
            solsNomRatw  = data['sols_nom']
            thetaHatRatw = data['thetahat']
            minSigmaRatw = data['minSigma']
            GammaRatw    = data['Gamma']
            iiRatw       = data['ii']
        except:
            traceback.print_exc()

# Compute FxT Bound
offset = 0.1
sig = np.min(minSigmaEst[int(offset/dt):iiEst])
lam = 1 / GammaEst[0,0]
c1 = sig * np.sqrt(2 / lam)
c2 = sig**3 * (2 / lam)**(3/2)
T_fixedEst = 2 * (c1 + c2)/(c1*c2) + offset
ic(T_fixedEst)

sig = np.min(minSigmaRat[int(0.01/dt):iiRat])
lam = 1 / GammaRat[0,0]
c1 = sig * np.sqrt(2 / lam)
c2 = sig**3 * (2 / lam)**(3/2)
T_fixedRat = 2 * (c1 + c2)/(c1*c2)
ic(T_fixedRat)


lwidth = 2
dash = [3,1]


def set_edges_black(ax):
    ax.spines['bottom'].set_color('#000000')
    ax.spines['top'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_color('#000000')

plt.close('all')

plot_tf = 11.0#4.25
# iis = [iiEst,iiFil,iiRat]
iis = [iiEst,iiRat]
if dt*np.min(iis) > plot_tf:
    new_ii = int(plot_tf/dt)
else:
    new_ii = np.min(iis)

new_ii = new_ii - int(0.15 * 10000)
ic(new_ii)
ic(tz_max)

############################################
### Controlla, Controlla ###
fig4 = plt.figure(figsize=(8,8))
ax2a  = fig4.add_subplot(411)
ax2b  = fig4.add_subplot(412)
ax2c  = fig4.add_subplot(413)
ax2d  = fig4.add_subplot(414)
set_edges_black(ax2a)
set_edges_black(ax2b)
set_edges_black(ax2c)
set_edges_black(ax2d)

x_min = -0.1
x_max = 13.0

# ax2a.plot(t[1:new_ii],0.0*np.ones(t[1:new_ii].shape),label=r'$u_{min}$',linewidth=lwidth+1,color='k')
# ax2a.plot(t[:new_ii],solsNomEst[:new_ii,0],label=r'$F_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2a.plot(t[:new_ii],solsNon[:new_ii,0],label='No Estimation',linewidth=lwidth,color=colors[ecc+5])
ax2a.plot(t[:new_ii],solsEst[:new_ii,0],label='Estimator: Const Wind',linewidth=lwidth,color=colors[ecc])
# ax2a.plot(t[:new_ii],solsNomRat[:new_ii,0],label=r'$F_{nom}$',linewidth=lwidth,color=colors[ecc+3],dashes=dash)
ax2a.plot(t[:new_ii],solsRat[:new_ii,0],label='Rates: Const Wind',linewidth=lwidth,color=colors[ecc+2])
ax2a.plot(t[:new_ii],solsRatw[:new_ii,0],label='Rates: Wind Gusts',dashes=dash,linewidth=lwidth,color=colors[ecc+2])
ax2a.plot(t[1:new_ii],f_max*np.ones(t[1:new_ii].shape),label=r'$\pm u_{max}$',linewidth=lwidth+1,color='k')
ax2a.set(ylabel=r'$F$',ylim=[-0.5,f_max*1.1],title='Control Inputs',xlim=[x_min,x_max])
ax2b.plot(t[1:new_ii],-tx_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2b.plot(t[1:new_ii], tx_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
# ax2b.plot(t[:new_ii],solsNomEst[:new_ii,1],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2b.plot(t[:new_ii],solsNon[:new_ii,1],linewidth=lwidth,color=colors[ecc+5])
ax2b.plot(t[:new_ii],solsEst[:new_ii,1],linewidth=lwidth,color=colors[ecc])
# ax2b.plot(t[:new_ii],solsNomRat[:new_ii,1],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+3],dashes=dash)
ax2b.plot(t[:new_ii],solsRat[:new_ii,1],linewidth=lwidth,color=colors[ecc+2])
ax2b.plot(t[:new_ii],solsRatw[:new_ii,1],dashes=dash,linewidth=lwidth,color=colors[ecc+2])
ax2b.set(ylabel=r'$\mu$')#xlim=[-0.1,5.2],)
ax2b.set(ylabel=r'$\tau_{\phi}$',ylim=[-1.1*tx_max,1.1*tx_max],xlim=[x_min,x_max])
ax2c.plot(t[1:new_ii],-ty_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2c.plot(t[1:new_ii], ty_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
# ax2c.plot(t[:new_ii],solsNomEst[:new_ii,2],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2c.plot(t[:new_ii],solsNon[:new_ii,2],linewidth=lwidth,color=colors[ecc+5])
ax2c.plot(t[:new_ii],solsEst[:new_ii,2],linewidth=lwidth,color=colors[ecc])
# ax2c.plot(t[:new_ii],solsNomRat[:new_ii,2],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+3],dashes=dash)
ax2c.plot(t[:new_ii],solsRat[:new_ii,2],linewidth=lwidth,color=colors[ecc+2])
ax2c.plot(t[:new_ii],solsRatw[:new_ii,2],dashes=dash,linewidth=lwidth,color=colors[ecc+2])
ax2c.set(ylabel=r'$\tau_{\theta}$',ylim=[-1.1*ty_max,1.1*ty_max],xlim=[x_min,x_max])
ax2d.plot(t[1:new_ii],-tz_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2d.plot(t[1:new_ii], tz_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
# ax2d.plot(t[:new_ii],solsNomEst[:new_ii,3],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2d.plot(t[:new_ii],solsNon[:new_ii,3],linewidth=lwidth,color=colors[ecc+5])
ax2d.plot(t[:new_ii],solsEst[:new_ii,3],linewidth=lwidth,color=colors[ecc])
# ax2d.plot(t[:new_ii],solsNomRat[:new_ii,3],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+3],dashes=dash)
ax2d.plot(t[:new_ii],solsRat[:new_ii,3],linewidth=lwidth,color=colors[ecc+2])
ax2d.plot(t[:new_ii],solsRatw[:new_ii,3],dashes=dash,linewidth=lwidth,color=colors[ecc+2])
ax2d.set(xlabel='Time (sec)',ylabel=r'$\tau_{\psi}$',ylim=[-1.1*tz_max,1.1*tz_max],xlim=[x_min,x_max])

for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
             ax2a.get_xticklabels() + ax2a.get_yticklabels()):
    item.set_fontsize(25)
ax2a.legend(fancybox=True,loc='right')
ax2a.grid(True,linestyle='dotted',color='white')

for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
             ax2b.get_xticklabels() + ax2b.get_yticklabels()):
    item.set_fontsize(25)
ax2b.legend(fancybox=True,loc='right')
ax2b.grid(True,linestyle='dotted',color='white')

for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
             ax2c.get_xticklabels() + ax2c.get_yticklabels()):
    item.set_fontsize(25)
ax2c.legend(fancybox=True,loc='right')
ax2c.grid(True,linestyle='dotted',color='white')

for item in ([ax2d.title, ax2d.xaxis.label, ax2d.yaxis.label] +
             ax2d.get_xticklabels() + ax2d.get_yticklabels()):
    item.set_fontsize(25)
ax2d.legend(fancybox=True,loc='right')
ax2d.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)






############################################
### PE, PN ###
fig1 = plt.figure(figsize=(8,8))
ax2  = fig1.add_subplot(111)
set_edges_black(ax2)

trig_freq     = 2 * np.pi * F_GERONO
xx1           =  A_GERONO * np.sin(trig_freq * t)
yy1           =  A_GERONO * np.sin(trig_freq * t) * np.cos(trig_freq * t)

ax2.plot(xNon[1:new_ii,0],xNon[1:new_ii,1],label='No Estimation',linewidth=lwidth+3,color=colors[ecc+5])
ax2.plot(xEst[1:new_ii,0],xEst[1:new_ii,1],label='Estimator: Const Wind',linewidth=lwidth+3,color=colors[ecc])
# ax2.plot(xFil[1:new_ii,0],xFil[1:new_ii,1],label='Filtering',linewidth=lwidth+3,color=colors[ecc+1])
ax2.plot(xRat[1:new_ii,0],xRat[1:new_ii,1],label='Rates: Const Wind',linewidth=lwidth+3,color=colors[ecc+2])
ax2.plot(xRatw[1:new_ii,0],xRatw[1:new_ii,1],dashes=dash,label='Rates: Wind Gusts',linewidth=lwidth+3,color=colors[ecc+2])
ax2.plot(xx1[1:new_ii-8600],yy1[1:new_ii-8600],'--',label='Track',linewidth=lwidth,color=colors[ecc+3])
ax2.set(ylabel='Y (m)',xlabel='X (m)',title='XY Trajectories',ylim=[-2.0,3.5],xlim=[-3.15,3.75])#,title='Control Inputs')
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

ax4.plot(T_fixedEst,theta[0],'bd',label=r'$T_{b,Est}$',markersize=20)
ax4.plot(T_fixedEst,theta[0],'bd',markersize=20)
ax4.plot(T_fixedEst,theta[1],'bd',markersize=20)
ax4.plot(T_fixedEst,theta[2],'bd',markersize=20)
ax4.plot(T_fixedRat,theta[0],'rd',label=r'$T_{b,Rat}$',markersize=20)
ax4.plot(T_fixedRat,theta[0],'rd',markersize=20)
ax4.plot(T_fixedRat,theta[1],'rd',markersize=20)
ax4.plot(T_fixedRat,theta[2],'rd',markersize=20)

ax4.plot(t[:new_ii],theta[0]*np.ones((new_ii,)),label=r'$\theta _{1,true}$',color=c1,linewidth=lwidth+3)
ax4.plot(t[:new_ii],theta[1]*np.ones((new_ii,)),':',label=r'$\theta _{2,true}$',color=c2,linewidth=lwidth+3)
ax4.plot(t[:new_ii],theta[2]*np.ones((new_ii,)),label=r'$\theta _{3,true}$',color=c3,linewidth=lwidth+3,dashes=dash)

ax4.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,EST}$',linewidth=lwidth+1,color=c4)
# ax4.plot(t[:new_ii],np.clip(thetaHatFil[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,FIL}$',linewidth=lwidth+1,color=c5)
ax4.plot(t[:new_ii],np.clip(thetaHatRat[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,RAT}$',linewidth=lwidth+1,color=c6)

ax4.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,EST}$',linewidth=lwidth+1,color=c4)
# ax4.plot(t[:new_ii],np.clip(thetaHatFil[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,FIL}$',linewidth=lwidth+1,color=c5)
ax4.plot(t[:new_ii],np.clip(thetaHatRat[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,RAT}$',linewidth=lwidth+1,color=c6)

ax4.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,EST}$',linewidth=lwidth+1,color=c4,dashes=dash)
# ax4.plot(t[:new_ii],np.clip(thetaHatFil[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,FIL}$',linewidth=lwidth+1,color=c5,dashes=dash)
ax4.plot(t[:new_ii],np.clip(thetaHatRat[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,RAT}$',linewidth=lwidth+1,color=c6,dashes=dash)

ax4.legend(fancybox=True,markerscale=1,fontsize=25)
ax4.set(xlabel='Time (sec)',ylabel=r'$\theta$',xlim=[-0.5,plot_tf+2],ylim=[-thetaMax[0]-0.25,thetaMax[0]+0.25])
ax4.set(xlabel='Time (sec)',ylabel=r'$\theta$',xlim=[-0.5,plot_tf+2],ylim=[-2.0,4.0])
for item in ([ax4.title, ax4.xaxis.label, ax4.yaxis.label] +
                 ax4.get_xticklabels() + ax4.get_yticklabels()):
        item.set_fontsize(25)
# ax4.set_xticklabels([])
ax4.grid(True,linestyle='dotted',color='white')

if True:#ii*dt > 1.0:
    ax4a_inset = inset_axes(ax4,width="100%",height="100%",
                          bbox_to_anchor=(.2, .08, .5, .3),bbox_transform=ax4.transAxes, loc=3)
                          # bbox_to_anchor=(.35, .44, .3, .1),bbox_transform=ax4.transAxes, loc=3)
    ax4a_inset.spines['bottom'].set_color('#000000')
    ax4a_inset.spines['top'].set_color('#000000')
    ax4a_inset.spines['right'].set_color('#000000')
    ax4a_inset.spines['left'].set_color('#000000')
    
    ax4a_inset.plot(T_fixedEst,theta[0],'bd',markersize=20)
    ax4a_inset.plot(T_fixedEst,theta[1],'bd',markersize=20)
    ax4a_inset.plot(T_fixedEst,theta[2],'bd',markersize=20)
    ax4a_inset.plot(T_fixedRat,theta[0],'rd',markersize=20)
    ax4a_inset.plot(T_fixedRat,theta[1],'rd',markersize=20)
    ax4a_inset.plot(T_fixedRat,theta[2],'rd',markersize=20)
    ax4a_inset.plot(t[:new_ii],theta[0]*np.ones((new_ii,)),label=r'$\theta _{1,true}$',color=c1,linewidth=lwidth+3)
    ax4a_inset.plot(t[:new_ii],theta[1]*np.ones((new_ii,)),':',label=r'$\theta _{2,true}$',color=c2,linewidth=lwidth+3)
    ax4a_inset.plot(t[:new_ii],theta[2]*np.ones((new_ii,)),label=r'$\theta _{3,true}$',color=c3,linewidth=lwidth+3,dashes=dash)

    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatRat[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,RAT}$',linewidth=lwidth+1,color=c6)
    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,EST}$',linewidth=lwidth+1,color=c4)
    # ax4a_inset.plot(t[:new_ii],np.clip(thetaHatFil[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,FIL}$',linewidth=lwidth+1,color=c5)

    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,EST}$',linewidth=lwidth+1,color=c4)
    # ax4a_inset.plot(t[:new_ii],np.clip(thetaHatFil[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,FIL}$',linewidth=lwidth+1,color=c5)
    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatRat[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,RAT}$',linewidth=lwidth+1,color=c6)

    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,EST}$',linewidth=lwidth+1,color=c4,dashes=dash)
    # ax4a_inset.plot(t[:new_ii],np.clip(thetaHatFil[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,FIL}$',linewidth=lwidth+1,color=c5,dashes=dash)
    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatRat[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,RAT}$',linewidth=lwidth+1,color=c6,dashes=dash)

    ax4a_inset.xaxis.set_major_locator(MaxNLocator(5))
    ax4a_inset.yaxis.set_major_locator(MaxNLocator(2))
    #ax4a_inset.set_xlim(T_fixed-0.1,T_fixed+0.1)
    ax4a_inset.set_xlim(-0.1,0.9)
    ax4a_inset.set_ylim(0.75,3.25)
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
    mark_inset(ax4,ax4a_inset,loc1=3,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")

    plt.draw()

plt.show()

# plt.tight_layout(pad=1.0)

filepath = '/home/dasc/MB/datastore/fxt_adaptation/quadrotor/money_sets/'

fig1.savefig(filepath+"QuadrotorTrajectories_AllStudies.eps",bbox_inches='tight',dpi=300)
fig1.savefig(filepath+"QuadrotorTrajectories_AllStudies.png",bbox_inches='tight',dpi=300)
fig4.savefig(filepath+"QuadrotorControls_AllStudies.eps",bbox_inches='tight',dpi=300)
fig4.savefig(filepath+"QuadrotorControls_AllStudies.png",bbox_inches='tight',dpi=300)
fig6.savefig(filepath+"QuadrotorParameters_RateEstimator.eps",bbox_inches='tight',dpi=300)
fig6.savefig(filepath+"QuadrotorParameters_RateEstimator.png",bbox_inches='tight',dpi=300)



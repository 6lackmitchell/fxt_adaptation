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


from quadrotor_settings import tf, dt, k1, k2, arm_length, f_max, tx_max, ty_max, tz_max, G, M, F_GERONO, A_GERONO, thetaMax, thetaMin, regressor
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

# filepath = '/Users/mblack/Documents/git/fxt_adaptation/datastore/money_sets/DragCoefficient_Simple/'
# filepath2 = '/Users/mblack/Documents/git/fxt_adaptation/datastore/money_sets/WindEstimation/'
filepath2 = '/Users/mblack/Documents/git/fxt_adaptation/datastore/quadrotor/'

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

    with open(filepath2 + 'NO_ESTIMATOR_CONSTANTWIND.pkl','rb') as f:
        try:
            data        = pickle.load(f)
            theta       = data['theta']
            xNon        = data['x']
            solsNon     = data['sols']
            solsNomNon  = data['sols_nom']
            cbfNon      = data['cbf']
        except:
            traceback.print_exc()

    with open(filepath2 + 'ESTIMATOR_CONSTANTWIND.pkl','rb') as f:
        try:
            data        = pickle.load(f)
            xEst        = data['x']
            solsEst     = data['sols']
            solsNomEst  = data['sols_nom']
            thetaHatEst = data['thetahat']
            minSigmaEst = data['minSigma']
            GammaEst    = data['Gamma']
            cbfEst      = data['cbf']
            iiEst       = data['ii']
        except:
            traceback.print_exc()

    with open(filepath2 + 'ESTIMATOR_CONSTANTWIND_PLUSGUST.pkl','rb') as f:
        try:
            data        = pickle.load(f)
            xEstD        = data['x']
            solsEstD     = data['sols']
            solsNomEstD  = data['sols_nom']
            thetaHatEstD = data['thetahat']
            minSigmaEstD = data['minSigma']
            GammaEstD    = data['Gamma']
            cbfEstD      = data['cbf']
            iiEstD       = data['ii']
        except:
            traceback.print_exc()

# Compute FxT Bound
offset = 5

sig = np.min(minSigmaEst[int(offset/dt):iiEst])
lam = 1 / GammaEst[0,0]
c1 = sig * np.sqrt(2 / lam)
c2 = sig**3 * (2 / lam)**(3/2)
T_fixedEst = 2 * (c1 + c2)/(c1*c2) + offset
ic(T_fixedEst)



sig = np.min(minSigmaEstD[int(offset/dt):iiEstD])
lam = 1 / GammaEstD[0,0]
c1 = sig * np.sqrt(2 / lam)
c2 = sig**3 * (2 / lam)**(3/2)
T_fixedEstD = 2 * (c1 + c2)/(c1*c2) + offset
ic(T_fixedEstD)
ic(sig)
ic(GammaEstD)
ic(iiEstD)
ic(iiEst)

# sys.exit()


# lwidth = 2
# dash = [3,1]
lwidth   = 3
dash     = [3,2]
legend_fontsize = 2*12.5
fontsize = 1.5*32
lstyle = ['solid','dashed','dotted','dashdot']
c1 = 'r'; c2 = 'c'; c3 = 'm'
# Teal, Coral, Khaki
c1 = '#029386'; c2 = '#FC5A50'; c3 = '#AAA662'
# Azure, Cyan, Indigo
c4 = '#069AF3'; c5 = '#00FFFF'; c6 = '#4B0082'


def set_edges_black(ax):
    ax.spines['bottom'].set_color('#000000')
    ax.spines['top'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_color('#000000')

plt.close('all')


plot_tf = 11.0#4.25
# iis = [iiEst,iiFil,iiRat]
iis = [iiEst,iiEstD]
if dt*np.min(iis) > plot_tf:
    new_ii = int(plot_tf/dt)
else:
    new_ii = np.min(iis)

# new_ii = new_ii - int(0.15 * 10000)
ic(new_ii)
ic(tz_max)


# ############################################
# ### PE, PN ###
# fig1 = plt.figure(figsize=(8,8))
# ax2  = fig1.add_subplot(111)
# # set_edges_black(ax2)

# ax2.plot(t[int(offset/dt):iiEst],minSigmaEst[int(offset/dt):iiEst],label='Est',linewidth=lwidth,color=c2)
# ax2.plot(t[:new_ii],minSigmaEstD[:new_ii],label='D',linewidth=lwidth,color=c5)
# ax2.legend(fancybox=True,fontsize=legend_fontsize/1.05)
# plt.tight_layout(pad=2.0)
# plt.show()


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
x_max = 13.5

ax2a.plot(t[:new_ii],solsNon[:new_ii,0],label='0-CW',linewidth=lwidth,color=c2,linestyle=lstyle[2])
ax2a.plot(t[:new_ii],solsEst[:new_ii,0],linewidth=lwidth,color=c6,linestyle=lstyle[1])
ax2a.plot(t[:new_ii],solsEstD[:new_ii,0],linewidth=lwidth,color=c4,linestyle=lstyle[3])
ax2a.plot(t[1:new_ii],f_max*np.ones(t[1:new_ii].shape),label=r'$\pm u_{max}$',linewidth=lwidth+1,color='k')
ax2a.set(ylabel=r'$F$',ylim=[-2,f_max*1.1],xlim=[x_min,x_max])#title='Control Inputs',
ax2a.set(xticklabels=[])

ax2b.plot(t[1:new_ii],-tx_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2b.plot(t[1:new_ii], tx_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2b.plot(t[:new_ii],solsNon[:new_ii,1],linewidth=lwidth,color=c2,linestyle=lstyle[2])
ax2b.plot(t[:new_ii],solsEst[:new_ii,1],label='SP-CW',linewidth=lwidth,color=c6,linestyle=lstyle[1])
ax2b.plot(t[:new_ii],solsEstD[:new_ii,1],label='SP-WG',linewidth=lwidth,color=c4,linestyle=lstyle[3])
ax2b.set(ylabel=r'$\mu$')#xlim=[-0.1,5.2],)
ax2b.set(ylabel=r'$\tau_{\phi}$',ylim=[-1.1*tx_max,1.1*tx_max],xlim=[x_min,x_max])
ax2b.set(xticklabels=[])

ax2c.plot(t[1:new_ii],-ty_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2c.plot(t[1:new_ii], ty_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2c.plot(t[:new_ii],solsNon[:new_ii,2],linewidth=lwidth,color=c2,linestyle=lstyle[2])
ax2c.plot(t[:new_ii],solsEst[:new_ii,2],linewidth=lwidth,color=c6,linestyle=lstyle[1])
ax2c.plot(t[:new_ii],solsEstD[:new_ii,2],linewidth=lwidth,color=c4,linestyle=lstyle[3])
ax2c.set(ylabel=r'$\tau_{\theta}$',ylim=[-1.1*ty_max,1.1*ty_max],xlim=[x_min,x_max])
ax2c.set(xticklabels=[])

ax2d.plot(t[1:new_ii],-tz_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2d.plot(t[1:new_ii], tz_max*np.ones(t[1:new_ii].shape),linewidth=lwidth+1,color='k')
ax2d.plot(t[:new_ii],solsNon[:new_ii,3],linewidth=lwidth,color=c2,linestyle=lstyle[2])
ax2d.plot(t[:new_ii],solsEst[:new_ii,3],linewidth=lwidth,color=c6,linestyle=lstyle[1])
ax2d.plot(t[:new_ii],solsEstD[:new_ii,3],linewidth=lwidth,color=c4,linestyle=lstyle[3])
ax2d.set(xlabel=r'$t$',ylabel=r'$\tau_{\psi}$',ylim=[-1.1*tz_max,1.1*tz_max],xlim=[x_min,x_max])

for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
             ax2a.get_xticklabels() + ax2a.get_yticklabels()):
    item.set_fontsize(fontsize)
ax2a.legend(fancybox=True,loc='right',fontsize=legend_fontsize)
ax2a.grid(True,linestyle='dotted',color='white')

for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
             ax2b.get_xticklabels() + ax2b.get_yticklabels()):
    item.set_fontsize(fontsize)
ax2b.legend(fancybox=True,loc='right',fontsize=legend_fontsize)
ax2b.grid(True,linestyle='dotted',color='white')

for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
             ax2c.get_xticklabels() + ax2c.get_yticklabels()):
    item.set_fontsize(fontsize)
# ax2c.legend(fancybox=True,loc='right',fontsize=legend_fontsize)
ax2c.grid(True,linestyle='dotted',color='white')

for item in ([ax2d.title, ax2d.xaxis.label, ax2d.yaxis.label] +
             ax2d.get_xticklabels() + ax2d.get_yticklabels()):
    item.set_fontsize(fontsize)
# ax2d.legend(fancybox=True,loc='right',fontsize=legend_fontsize)
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

ax2.plot(xx1[1:new_ii-8600],yy1[1:new_ii-8600],label='RT',linewidth=lwidth,color=colors[ecc+3],linestyle=lstyle[0])
ax2.plot(xNon[1:new_ii,0],xNon[1:new_ii,1],label='0-CW',linewidth=lwidth,color=c2,linestyle=lstyle[2])
ax2.plot(xEst[1:new_ii,0],xEst[1:new_ii,1],label='SP-CW',linewidth=lwidth,color=c6,linestyle=lstyle[1])
ax2.plot(xEstD[1:new_ii,0],xEstD[1:new_ii,1],label='SP-WG',linewidth=lwidth,color=c4,linestyle=lstyle[3])
ax2.set(ylabel=r'$y$',xlabel=r'$x$',xlim=[-3.15,4.25])#title='XY Trajectories',,title='Control Inputs')
for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(fontsize)
ax2.legend(fancybox=True,fontsize=legend_fontsize/1.05)
ax2.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)


############################################
### CBF Trajectories ###

fig10 = plt.figure(figsize=(8,8))

ax10a  = fig10.add_subplot(211)
ax10b  = fig10.add_subplot(212)
set_edges_black(ax10a)
set_edges_black(ax10b)

ax10a.plot(t[1:new_ii],np.zeros(t[1:new_ii].shape),label='Barrier',linewidth=lwidth+1,color='k')
ax10a.plot(t[1:new_ii],cbfNon[1:new_ii,0],label='0-CW',linewidth=lwidth,color=c2,linestyle=lstyle[2])
ax10a.plot(t[1:new_ii],cbfEst[1:new_ii,0],label='SP-CW',linewidth=lwidth,color=c6,linestyle=lstyle[1])
ax10a.plot(t[1:new_ii],cbfEstD[1:new_ii,0],label='SP-WG',linewidth=lwidth,color=c4,linestyle=lstyle[3])

ax10b.plot(t[1:new_ii],np.zeros(t[1:new_ii].shape),label='Barrier',linewidth=lwidth+1,color='k')
ax10b.plot(t[1:new_ii],cbfNon[1:new_ii,1],label='0-CW',linewidth=lwidth,color=c2,linestyle=lstyle[2])
ax10b.plot(t[1:new_ii],cbfEst[1:new_ii,1],label='SP-CW',linewidth=lwidth,color=c6,linestyle=lstyle[1])
ax10b.plot(t[1:new_ii],cbfEstD[1:new_ii,1],label='SP-WG',linewidth=lwidth,color=c4)


for item in ([ax10a.title, ax10a.xaxis.label, ax10a.yaxis.label] +
             ax10a.get_xticklabels() + ax10a.get_yticklabels()):
    item.set_fontsize(fontsize)
ax10a.set(ylabel=r'$h_1$',ylim=[-0.1,1.4],xlim=[-0.5,plot_tf+2],xticklabels=[])#title='XY Trajectories',,title='Control Inputs')
ax10a.yaxis.set_major_locator(plt.MaxNLocator(3))
ax10a.legend(fancybox=True,loc='right',fontsize=legend_fontsize)
ax10a.grid(True,linestyle='dotted',color='white')

for item in ([ax10b.title, ax10b.xaxis.label, ax10b.yaxis.label] +
             ax10b.get_xticklabels() + ax10b.get_yticklabels()):
    item.set_fontsize(fontsize)
ax10b.set(ylabel=r'$h_2$',xlabel=r'$t$',ylim=[-0.1,1.4],xlim=[-0.5,plot_tf+2])#title='XY Trajectories',,title='Control Inputs')
ax10b.yaxis.set_major_locator(plt.MaxNLocator(3))
ax10b.legend(fancybox=True,loc='right',fontsize=legend_fontsize)
ax10b.grid(True,linestyle='dotted',color='white')

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

fig6 = plt.figure(figsize=(8,8))
ax4  = fig6.add_subplot(111)
ax4.spines['bottom'].set_color('#000000')
ax4.spines['top'].set_color('#000000')
ax4.spines['right'].set_color('#000000')
ax4.spines['left'].set_color('#000000')


ax4.plot(T_fixedEst,theta[0],'rd',label=r'$T_{b}$',markersize=20,color=c1)
ax4.plot(T_fixedEst,theta[0],'rd',markersize=20,color=c1)
ax4.plot(T_fixedEst,theta[1],'rd',markersize=20,color=c1)
ax4.plot(T_fixedEst,theta[2],'rd',markersize=20,color=c1)
# ax4.plot(T_fixedEstD,theta[0],'bd',label=r'$T_{b,SPD}$',markersize=20,color=c4))
# ax4.plot(T_fixedEstD,theta[0],'bd',markersize=20,color=c4)
# ax4.plot(T_fixedEstD,theta[1],'bd',markersize=20,color=c4)
# ax4.plot(T_fixedEstD,theta[2],'bd',markersize=20,color=c4)
ax4.plot(t[:new_ii],thetaMax[0]*np.ones((new_ii,)),'k',label=r"$C _{Min}$",linewidth=lwidth+3)
ax4.plot(t[:new_ii],thetaMin[0]*np.ones((new_ii,)),'k',linewidth=lwidth+3)

ax4.plot(t[:new_ii],theta[0]*np.ones((new_ii,)),label=r"$C_x$",color=c3,linewidth=lwidth)
ax4.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r"$\hat C_{x,SP-CW}$",linewidth=lwidth+1,color=c6)
ax4.plot(t[:new_ii],np.clip(thetaHatEstD[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r"$\hat C_{x,SP-WG}$",linewidth=lwidth+1,color=c4)

ax4.plot(t[:new_ii],theta[1]*np.ones((new_ii,)),':',label=r"$C_y$",color=c3,linewidth=lwidth)
ax4.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r"$\hat C_{y,SP-CW}$",linewidth=lwidth+1,color=c6)
ax4.plot(t[:new_ii],np.clip(thetaHatEstD[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r"$\hat C_{y,SP-WG}$",linewidth=lwidth+1,color=c4)

ax4.plot(t[:new_ii],theta[2]*np.ones((new_ii,)),label=r"$C_z$",color=c3,linewidth=lwidth,dashes=dash)
ax4.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r"$\hat C_{z,SP-CW}$",linewidth=lwidth+1,color=c6,dashes=dash)
ax4.plot(t[:new_ii],np.clip(thetaHatEstD[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r"$\hat C_{z,SP-WG}$",linewidth=lwidth+1,color=c4,dashes=dash)

ax4.legend(fancybox=True,markerscale=1,fontsize=legend_fontsize*1.23)
# for line in ax4.legend().get_lines():
#     line.set_linewidth(lwidth)
# for text in ax4.legend().get_texts():
#     text.set_fontsize(legend_fontsize)
ax4.set(xlabel=r"$t$",ylabel=r"$C_d$",xlim=[-0.5,plot_tf+3.2],ylim=[thetaMin[0]-0.1,1])
# ax4.set(xlabel=r"$t$",ylabel=r"$\theta$",xlim=[-0.5,plot_tf+3.1],ylim=[-0.1,2.1])
for item in ([ax4.title, ax4.xaxis.label, ax4.yaxis.label] +
                 ax4.get_xticklabels() + ax4.get_yticklabels()):
        item.set_fontsize(fontsize)
# ax4.set_xticklabels([])
ax4.grid(True,linestyle='dotted',color='white')

if False:#ii*dt > 1.0:
    ax4a_inset = inset_axes(ax4,width="100%",height="100%",
                          bbox_to_anchor=(.05, .5, .5, .3),bbox_transform=ax4.transAxes, loc=3)
    ax4a_inset.spines['bottom'].set_color('#000000')
    ax4a_inset.spines['top'].set_color('#000000')
    ax4a_inset.spines['right'].set_color('#000000')
    ax4a_inset.spines['left'].set_color('#000000')
    
    ax4a_inset.plot(T_fixedEst,theta[0],'rd',markersize=20,color=c6)
    ax4a_inset.plot(T_fixedEst,theta[1],'rd',markersize=20,color=c6)
    ax4a_inset.plot(T_fixedEst,theta[2],'rd',markersize=20,color=c6)
    ax4a_inset.plot(T_fixedEstD,theta[0],'bd',markersize=20,color=c4)
    ax4a_inset.plot(T_fixedEstD,theta[1],'bd',markersize=20,color=c4)
    ax4a_inset.plot(T_fixedEstD,theta[2],'bd',markersize=20,color=c4)
    ax4a_inset.plot(t[:new_ii],theta[0]*np.ones((new_ii,)),label=r'$\theta _1$',color=c3,linewidth=lwidth+5)
    ax4a_inset.plot(t[:new_ii],theta[1]*np.ones((new_ii,)),':',label=r'$\theta _2$',color=c3,linewidth=lwidth+5)
    ax4a_inset.plot(t[:new_ii],theta[2]*np.ones((new_ii,)),label=r'$\theta _3$',color=c3,linewidth=lwidth+5,dashes=dash)

    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,SP-CW}$',linewidth=lwidth+1,color=c6)
    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEstD[:new_ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,SP-WG}$',linewidth=lwidth+1,color=c4)

    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,SP-CW}$',linewidth=lwidth+1,color=c6)
    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEstD[:new_ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,SP-WG}$',linewidth=lwidth+1,color=c4)

    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEst[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,SP-CW}$',linewidth=lwidth+1,color=c6,dashes=dash)
    ax4a_inset.plot(t[:new_ii],np.clip(thetaHatEstD[:new_ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,SP-WG}$',linewidth=lwidth+1,color=c4,dashes=dash)

    ax4a_inset.xaxis.set_major_locator(MaxNLocator(5))
    ax4a_inset.yaxis.set_major_locator(MaxNLocator(2))
    #ax4a_inset.set_xlim(T_fixed-0.1,T_fixed+0.1)
    ax4a_inset.set_xlim(5.25,5.75)
    ax4a_inset.set_ylim(0.0,1)
    ax4a_inset.xaxis.tick_top()
    ax4a_inset.yaxis.tick_right()
    for item in ([ax4a_inset.title, ax4a_inset.xaxis.label, ax4a_inset.yaxis.label] +
                 ax4a_inset.get_xticklabels() + ax4a_inset.get_yticklabels()):
        item.set_fontsize(fontsize/2)
    ax4a_inset.set_xticks(ax4a_inset.get_xticks().tolist())
    # ax4a_inset.set_yticks(ax4a_inset.get_yticks().tolist())
    ax4a_inset.set_xticklabels([None,None,None,None,None,None,None])
    ax4a_inset.set_yticklabels([None,None,None,None,None,None])
    # ax4a_inset.get_yaxis().set_visible(False)
    ax4a_inset.grid(True,linestyle='dotted',color='white')
    mark_inset(ax4,ax4a_inset,loc1=3,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")

    plt.draw()

plt.tight_layout(pad=1.0)

plt.show()


filepath = '/Users/mblack/Documents/git/fxt_adaptation/datastore/quadrotor/'

# fig1.savefig(filepath+"QuadrotorTrajectories_AllStudies.eps",bbox_inches='tight',dpi=300)
# fig1.savefig(filepath+"QuadrotorTrajectories_AllStudies.png",bbox_inches='tight',dpi=300)
# fig4.savefig(filepath+"QuadrotorControls_AllStudies.eps",bbox_inches='tight',dpi=300)
# fig4.savefig(filepath+"QuadrotorControls_AllStudies.png",bbox_inches='tight',dpi=300)
# fig10.savefig(filepath+"QuadrotorCBFs_AllStudies.eps",bbox_inches='tight',dpi=300)
# fig10.savefig(filepath+"QuadrotorCBFs_AllStudies.png",bbox_inches='tight',dpi=300)
# fig6.savefig(filepath+"QuadrotorParameters_RateEstimator.eps",bbox_inches='tight',dpi=300)
# fig6.savefig(filepath+"QuadrotorParameters_RateEstimator.png",bbox_inches='tight',dpi=300)

fig1.savefig(filepath+"QuadrotorTrajectories_AllStudies.eps",bbox_inches='tight',format='eps')
fig4.savefig(filepath+"QuadrotorControls_AllStudies.eps",bbox_inches='tight',format='eps')
fig10.savefig(filepath+"QuadrotorCBFs_AllStudies.eps",bbox_inches='tight',format='eps')
fig6.savefig(filepath+"QuadrotorParameters_RateEstimator.svg",bbox_inches='tight')

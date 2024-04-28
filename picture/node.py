import numpy as np
import math
import random
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import optimize
import time


path = "./t"

def totalNode(sim):
    eps = []
    node = []
    for i in range(0, 99900):
        eps.append(i)
        node.append(sim[i][2])

    return eps,node

def findStepsIndex(steps):
    b = base = 5000
    index = []
    for i in range(len(steps)):
        if steps[i] > b:
            index.append(i)
            b += base

    return index


ddpgBpath = path + "/ddpg/b/sim.npy"
ddpgBSim = np.load(ddpgBpath)

td3Bpath = path + "/td3/b/sim.npy"
td3BSim = np.load(td3Bpath)

dbeps, dbnode = totalNode(ddpgBSim)

tbeps, tbnode = totalNode(td3BSim)


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#plt.rcParams['font.sans-serif']= ['Times New Roman']
plt.rcParams['font.size']= 18


font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 13}
plt.rc('font', **font)
fig, ax = plt.subplots(1, 1)

plt.ticklabel_format(style='sci', scilimits=(0,0))


marke = findStepsIndex(tbeps)
ax.plot(tbeps, tbnode, color="#219EBC", linestyle='-', linewidth=1.5)#,marker="o",markevery=marke,markersize = 6)

marke = findStepsIndex(dbeps)
ax.plot(dbeps, dbnode, color="#FFB703", linestyle='-', linewidth=1.5)#,marker="x",markevery=marke,markersize = 6)

ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.legend(labels=["Node numbers of TD3 LCBT attack","Node numbers of DDPG LCBT attack"], fontsize=14,loc='best')#, ncol=3)
ax.set_xlabel("Episodes", fontsize=18)#,fontsize=18)
ax.set_ylabel("Node numbers", fontsize=18)#,fontsize=18)
plt.title("Environment 3", fontsize=18)
parts = path.split('/')
#ax.set_title(parts[-1].upper(), fontsize=18)
plt.savefig(path+"/node.pdf")
plt.show()

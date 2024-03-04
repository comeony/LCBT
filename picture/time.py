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

def totalTime(time):
    percent = []
    step = []
    for i in range(2500,100010):
        step.append(time[i][0])
        if(time[i][0] == 500000):
            print(time[i][1],time[i][2],time[i][1] / time[i][2])
        if(time[i][0] == 1000000):
            print(time[i][1],time[i][2],time[i][1] / time[i][2])
        percent.append(time[i][1] / time[i][2])

    return step, percent

def findStepsIndex(steps):
    b = base = 5000
    index = []
    for i in range(len(steps)):
        if steps[i] > b:
            index.append(i)
            b += base

    return index


ddpgBpath = path + "/ddpg/b5_1/times.npy"
ddpgBTime = np.load(ddpgBpath)
ddpgWpath = path + "/ddpg/b5_1/times.npy"
ddpgWTime = np.load(ddpgWpath)


ppoBpath = path + "/TD3/b5_2/times.npy"
ppoBTime = np.load(ppoBpath)
ppoWpath = path + "/TD3/b5_2/times.npy"
ppoWTime = np.load(ppoWpath)


dstep, dpercent = totalTime(ddpgBTime)

pstep, ppercent = totalTime(ppoBTime)


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

ax.plot(pstep, ppercent, color="#8e6fad", linestyle='-',linewidth=1.5)#,marker="o",markevery=marke,markersize = 6)

ax.plot(dstep, dpercent, color="#c72f2f", linestyle='-',linewidth=1.5)#,marker="x",markevery=marke,markersize = 6)

ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.legend(labels=["Attack Time Percentage in LCBT on TD3","Attack Time Percentage in LCBT on DDPG"], fontsize=14,loc='upper right')#, ncol=3)
ax.set_xlabel("Steps",fontsize = 14)#,fontsize=18)
ax.set_ylabel("Time Percentage",fontsize = 14)#,fontsize=18)
plt.title("Environment 3")
parts = path.split('/')
#ax.set_title(parts[-1].upper(), fontsize=18)
plt.savefig(path+"/time.pdf")
plt.show()

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


ddpgBpath = path + "/ddpg/b/times.npy"
ddpgBTime = np.load(ddpgBpath)


td3Bpath = path + "/TD3/b/times.npy"
td3BTime = np.load(td3Bpath)



dstep, dpercent = totalTime(ddpgBTime)

tstep, tpercent = totalTime(td3BTime)


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

ax.plot(tstep, tpercent, color="#8e6fad", linestyle='-',linewidth=1.5)#,marker="o",markevery=marke,markersize = 6)

ax.plot(dstep, dpercent, color="#c72f2f", linestyle='-',linewidth=1.5)#,marker="x",markevery=marke,markersize = 6)

ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style='plain')
ax.legend(labels=["Attack Time Percentage in LCBT on TD3","Attack Time Percentage in LCBT on DDPG"], fontsize=14,loc='lower right')#, ncol=3)
ax.set_xlabel("Steps",fontsize = 14)#,fontsize=18)
ax.set_ylabel("Time Percentage",fontsize = 14)#,fontsize=18)
plt.title("Environment 3")
parts = path.split('/')
#ax.set_title(parts[-1].upper(), fontsize=18)
plt.savefig(path+"/time.pdf")
plt.show()

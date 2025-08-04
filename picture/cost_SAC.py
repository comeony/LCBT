import numpy as np
import matplotlib.pyplot as plt

# SAC in env3

path = "./t"
SACBSim = np.load(path + "/SAC/b/sim.npy")

# colors = ["#2779ac",
#           "#8e6fad",
#           "#349d35",
#           "#c72f2f"]
colors = ["#219EBC", "#023047", "#FFB703", "#FB8402"]
print(len(SACBSim))
l = min(len(SACBSim),105000)
print(l)
alltimestep = []

SACB = []

alltimestep.append(0)
SACB.append(10 - SACBSim[0][0])

for i in range(1,l):
    alltimestep.append(i*10)

    SACB.append(SACB[i - 1] +10 -  SACBSim[i][0])


window_size = 5
alltimestep = alltimestep[:-window_size]
window = np.ones(int(window_size))/float(window_size)

SACB = np.convolve(SACB, window, 'same')
SACB = SACB[:-window_size]


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
marke = [i for i in range(0,len(alltimestep),10000)]

ax.plot(alltimestep, SACB, color=colors[3], linestyle='-',linewidth=1.5,marker="x",markevery=marke,markersize = 6)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.legend(labels=["Cost of SAC LCBT attack"], fontsize=12,loc='best')#, ncol=3)

ax.set_xlabel("Time steps",fontsize = 18)#,fontsize=18)
ax.set_ylabel("",fontsize = 14)#,fontsize=18)

parts = path.split('/')
#ax.set_title(parts[-1].upper(), fontsize=18)
plt.savefig(path+"/cost.pdf")
plt.show()
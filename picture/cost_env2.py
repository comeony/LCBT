import numpy as np
import matplotlib.pyplot as plt



path = "./s"
ddpgBSim = np.load(path + "/ddpg/b/sim.npy")
ddpgWSim = np.load(path + "/ddpg/w/sim.npy")
td3BSim = np.load(path + "/td3/b/sim.npy")
td3WSim = np.load(path + "/td3/w/sim.npy")

colors = ["#219EBC", "#023047", "#FFB703", "#FB8402"]

l = min(len(td3WSim),len(td3BSim),len(ddpgWSim),len(ddpgBSim),105000)
print(l)
alltimestep = []
ddpgW = []
ddpgB = []
td3W = []
td3B = []
alltimestep.append(0)
ddpgW.append(10 - ddpgWSim[0][0])
ddpgB.append(10 - ddpgBSim[0][0])
td3W.append(10 - td3WSim[0][0])
td3B.append(10 - td3BSim[0][0])
for i in range(1,l):
    alltimestep.append(i*10)
    ddpgW.append(ddpgW[i-1]+10 - ddpgWSim[i][0])
    ddpgB.append(ddpgB[i - 1] +10 -  ddpgBSim[i][0])
    td3W.append(td3W[i - 1] +10 -  td3WSim[i][0])
    td3B.append(td3B[i - 1] +10 -  td3BSim[i][0])

window_size = 5000
alltimestep = alltimestep[:-window_size]
window = np.ones(int(window_size))/float(window_size)
ddpgW = np.convolve(ddpgW, window, 'same')
ddpgW = ddpgW[:-window_size]
ddpgB = np.convolve(ddpgB, window, 'same')
ddpgB = ddpgB[:-window_size]
td3W = np.convolve(td3W, window, 'same')
td3W = td3W[:-window_size]
td3B = np.convolve(td3B, window, 'same')
td3B = td3B[:-window_size]

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
ax.plot(alltimestep, td3W, color=colors[0], linestyle='--',linewidth=1.5,marker="D",markevery=marke,markersize = 6)
ax.plot(alltimestep, td3B, color=colors[1], linestyle='-.',linewidth=1.5,marker="o",markevery=marke,markersize = 6)
ax.plot(alltimestep, ddpgW, color =colors[2],linestyle=':',linewidth=1.5,marker="v",markevery=marke,markersize = 6)
ax.plot(alltimestep, ddpgB, color=colors[3], linestyle='-',linewidth=1.5,marker="x",markevery=marke,markersize = 6)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.legend(labels=["Cost of TD3 Oracle attack","Cost of TD3 LCBT attack", "Cost of DDPG Oracle attack","Cost of DDPG LCBT attack"], fontsize=13.5,loc='best')#, ncol=3)
ax.set_xlabel("Time steps",fontsize = 18)#,fontsize=18)
ax.set_ylabel("Cumulative cost",fontsize = 18)#,fontsize=18)

parts = path.split('/')
#ax.set_title(parts[-1].upper(), fontsize=18)
plt.savefig(path+"/cost.pdf")
plt.show()

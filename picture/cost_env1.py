import numpy as np
import matplotlib.pyplot as plt

path = "./f"

colors = ["#219EBC", "#023047", "#FFB703", "#FB8402"]
def totalCost(sim):
    steps = [0,]
    cost = [0,]
    i = 0
    while steps[-1] < 3*1e5+40000:
        steps.append(steps[-1] + sim[i][1])
        cost.append(cost[-1] + sim[i][1] - sim[i][0])
        i += 1

    return steps, cost


def findStepsIndex(steps):
    b = base = 60000
    index = []
    for i in range(len(steps)):
        if steps[i] > b:
            index.append(i)
            b += base

    return index


ddpgBpath = path + "/ddpg/b/sim.npy"
ddpgBSim = np.load(ddpgBpath)

ddpgWpath = path + "/ddpg/w/sim.npy"
ddpgWSim = np.load(ddpgWpath)


ppoBpath = path + "/ppo/b/sim.npy"
#ppoBpath =  "05160131/f/ppo/b/sim.npy"
ppoBSim = np.load(ppoBpath)

ppoWpath = path + "/ppo/w/sim.npy"
#ppoWpath = "05160131/f/ppo/w/sim.npy"
ppoWSim = np.load(ppoWpath)


dbsteps, dbcost = totalCost(ddpgBSim)
dwsteps, dwcost = totalCost(ddpgWSim)

pbsteps, pbcost = totalCost(ppoBSim)
pwsteps, pwcost = totalCost(ppoWSim)
print(len(pwsteps))


window_size = 5000
dbsteps = dbsteps[:-window_size]
dwsteps = dwsteps[:-window_size]
pbsteps = pbsteps[:-window_size]
pwsteps = pwsteps[:-window_size]
window = np.ones(int(window_size))/float(window_size)
dbcost = np.convolve(dbcost, window, 'same')
dbcost = dbcost[:-window_size]
dwcost = np.convolve(dwcost, window, 'same')
dwcost = dwcost[:-window_size]
pbcost = np.convolve(pbcost, window, 'same')
pbcost = pbcost[:-window_size]
pwcost = np.convolve(pwcost, window, 'same')
pwcost = pwcost[:-window_size]


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

marke = findStepsIndex(pwsteps)
ax.plot(pwsteps, pwcost, color=colors[0], linestyle='--',linewidth=1.5,marker="D",markevery=marke,markersize = 6)

marke = findStepsIndex(pbsteps)
ax.plot(pbsteps, pbcost, color=colors[1], linestyle='-.',linewidth=1.5,marker="o",markevery=marke,markersize = 6)

marke = findStepsIndex(dwsteps)
ax.plot(dwsteps, dwcost, color=colors[2], linestyle=':',linewidth=1.5,marker="v",markevery=marke,markersize = 6)

marke = findStepsIndex(dbsteps)
ax.plot(dbsteps, dbcost, color=colors[3], linestyle='-',linewidth=1.5,marker="x",markevery=marke,markersize = 6)

ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.legend(labels=["Cost of PPO Oracle attack","Cost of PPO LCBT attack", "Cost of DDPG Oracle attack","Cost of DDPG LCBT attack"], fontsize=14,loc='lower right')#, ncol=3)
ax.set_xlabel("Time steps",fontsize = 18)#,fontsize=18)
ax.set_ylabel("Cumulative cost",fontsize = 18)#,fontsize=18)

parts = path.split('/')
#ax.set_title(parts[-1].upper(), fontsize=18)
plt.savefig(path+"/cost.pdf")
plt.show()


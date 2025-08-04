
import numpy as np

import matplotlib.pyplot as plt




ALLtargetReward = {
    "./s/ddpg" : 6.603,
    "./s/td3" : 7.399,
    "./f/ddpg" : 6.179,
    "./f/ppo" : 6.05,
    "./f/td3" : 6.226,
    "./t/ddpg": 5.727,
    "./t/td3": 6.932,
    "./t/SAC": 6.921
}
# colors = ["#2779ac",
# "#8e6fad",#"#f2811d"
# "#349d35",
# "#c72f2f"]
colors = ["#219EBC", "#023047", "#FFB703", "#FB8402"]
p = [ "./s/ddpg", "./s/td3", "./f/ddpg", "./f/ppo", "./t/SAC"]

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#plt.rcParams['font.sans-serif']= ['Times New Roman']
plt.rcParams['font.size'] = 18

n = 3000000
K = 0.9


def totalCost(sim, reward, MT):
    steps = [0,]
    cost = [0,]
    r = [reward[0],]
    i = 0
    t = 0
    while steps[-1] < MT:
        steps.append(steps[-1] + sim[i][1])
        cost.append(cost[-1] + sim[i][1] - sim[i][0])
        r.append(reward[i+1])
        i += 1

    sp = 100
    rsteps = [steps[i] for i in range(0, len(steps), sp)]
    sub_arrays = np.array_split(r, len(r) // sp)
    r = [np.mean(sub_array) for sub_array in sub_arrays]

    l = min(len(rsteps), len(r))
    rsteps = rsteps[0:l]
    r = r[0:l]

    return steps,cost,rsteps,r

def stepsReward(step,reward,MT):
    steps = [0,]
    cost = [0,]
    r = [reward[0],]
    i = 0
    t = 0
    while steps[-1] < MT :
        steps.append(steps[-1] + step[i])
        r.append(reward[i+1])
        i += 1

    sp = 100
    rsteps = [steps[i] for i in range(0, len(steps), sp)]
    sub_arrays = np.array_split(r, len(r) // sp)
    r = [np.mean(sub_array) for sub_array in sub_arrays]

    l = min(len(rsteps), len(r))
    rsteps = rsteps[0:l]
    r = r[0:l]

    return steps,cost,rsteps,r

for path in p:
    cost = []
    realreward = []
    blackreward = []
    whitereward = []
    targetReward = []
    allsteps = []
    trainsteps = []
    nsteps = []
    bsteps = []
    wsteps = []
    bpath = path + "/b"
    wpath = path + "/w"
    npath = path + "/n"
    if path == "./s/td3":
        sp = 10
        tempb = np.load(bpath + "/reward.npy")
        tempw = np.load(wpath + "/reward.npy")
        tempr = np.load(npath + "/reward.npy")
        l = min(len(tempw)//sp, len(tempb)//sp, 100000 // sp)
        lr = min(100000 // sp, len(tempr)//sp)

        sub_arrays = np.array_split(tempb, len(tempb) // sp)
        tempb = [np.mean(sub_array) for sub_array in sub_arrays]

        sub_arrays = np.array_split(tempw, len(tempw) // sp)
        tempw = [np.mean(sub_array) for sub_array in sub_arrays]

        sub_arrays = np.array_split(tempr, len(tempr) // sp)
        tempr = [np.mean(sub_array) for sub_array in sub_arrays]


        for i in range(0,int(l),40):
            allsteps.append(i * sp * 10)
            blackreward.append(tempb[i])
            whitereward.append(tempw[i])
            targetReward.append(ALLtargetReward[path])

            if i < lr:
                trainsteps.append(i * sp * 10)
                realreward.append(tempr[i])

        bsteps = allsteps
        wsteps = allsteps
        nsteps = allsteps
    elif path == "./s/ddpg":
        sp = 10
        tempb = np.load(bpath + "/reward.npy")
        tempw = np.load(wpath + "/reward.npy")
        tempr = np.load(npath + "/reward.npy")
        l = min(len(tempw) // sp, len(tempb) // sp, 100000 // sp)
        lr = min(100000 // sp, len(tempr) // sp)

        sub_arrays = np.array_split(tempb, len(tempb) // sp)
        tempb = [np.mean(sub_array) for sub_array in sub_arrays]

        sub_arrays = np.array_split(tempw, len(tempw) // sp)
        tempw = [np.mean(sub_array) for sub_array in sub_arrays]

        sub_arrays = np.array_split(tempr, len(tempr) // sp)
        tempr = [np.mean(sub_array) for sub_array in sub_arrays]

        for i in range(0, int(l), 40):
            allsteps.append(i * sp * 10)
            blackreward.append(tempb[i])
            whitereward.append(tempw[i])
            targetReward.append(ALLtargetReward[path])

            if i < lr:
                trainsteps.append(i * sp * 10)
                realreward.append(tempr[i])
        bsteps = allsteps
        wsteps = allsteps
        nsteps = allsteps
    elif path == "./f/ddpg":
        sp = 100
        MT = 3e5
        tempb = np.load(bpath + "/reward.npy")
        tempw = np.load(wpath + "/reward.npy")
        tempr = np.load(npath + "/reward.npy")
        print(tempr)
        db = np.load(bpath + "/sim.npy")
        dw = np.load(wpath + "/sim.npy")
        dr = np.load(npath + "/steps.npy")
        _, _, bsteps, blackreward = totalCost(db, tempb, MT)
        _, _, wsteps, whitereward = totalCost(dw, tempw, MT)
        _, _, nsteps, realreward = stepsReward(dr, tempr, MT)
        for i in range(0,int(len(bsteps))):
            targetReward.append(ALLtargetReward[path])
        allsteps = bsteps


    elif path == "./f/ppo":
        sp = 100
        MT = 3e5
        tempb = np.load(bpath + "/reward.npy")
        tempw = np.load(wpath + "/reward.npy")
        tempr = np.load(npath + "/reward.npy")
        db = np.load(bpath + "/sim.npy")
        dw = np.load(wpath + "/sim.npy")
        dr = np.load(npath + "/steps.npy")
        _, _, bsteps, blackreward = totalCost(db, tempb,MT)
        _, _, wsteps, whitereward = totalCost(dw, tempw,MT)
        _, _, nsteps, realreward = stepsReward(dr,tempr,MT)
        for i in range(0,int(len(bsteps))):
            targetReward.append(ALLtargetReward[path])
        allsteps = bsteps
    elif path == "./t/ddpg":
        sp = 10
        tempb = np.load(bpath + "/reward.npy")
        tempw = np.load(wpath + "/reward.npy")
        tempr = np.load(npath + "/reward.npy")
        l = min(len(tempw) // sp, len(tempb) // sp, 100000 // sp)
        lr = min(100000 // sp, len(tempr) // sp)

        sub_arrays = np.array_split(tempb, len(tempb) // sp)
        tempb = [np.mean(sub_array) for sub_array in sub_arrays]

        sub_arrays = np.array_split(tempw, len(tempw) // sp)
        tempw = [np.mean(sub_array) for sub_array in sub_arrays]

        sub_arrays = np.array_split(tempr, len(tempr) // sp)
        tempr = [np.mean(sub_array) for sub_array in sub_arrays]

        for i in range(0, int(l), 40):
            allsteps.append(i * sp * 10)
            blackreward.append(tempb[i])
            whitereward.append(tempw[i])
            targetReward.append(ALLtargetReward[path])

            if i < lr:
                trainsteps.append(i * sp * 10)
                realreward.append(tempr[i])
        bsteps = allsteps
        wsteps = allsteps
        nsteps = allsteps

    elif path == "./t/td3" or path == "./t/SAC":
        sp = 10
        tempb = np.load(bpath + "/reward.npy")
        tempw = np.load(wpath + "/reward.npy")
        tempr = np.load(npath + "/reward.npy")
        print(len(tempw), len(tempb), len(tempr))

        l = min(len(tempw)//sp, len(tempb)//sp, 100000 // sp)
        lr = min(100000 // sp, len(tempr)//sp)

        sub_arrays = np.array_split(tempb, len(tempb) // sp)
        tempb = [np.mean(sub_array) for sub_array in sub_arrays]

        sub_arrays = np.array_split(tempw, len(tempw) // sp)
        tempw = [np.mean(sub_array) for sub_array in sub_arrays]

        sub_arrays = np.array_split(tempr, len(tempr) // sp)
        tempr = [np.mean(sub_array) for sub_array in sub_arrays]


        # for i in range(0,len(tempw),40):
        for i in range(0, int(l), 40):
            allsteps.append(i * sp * 10)
            if i < len(tempb):
                bsteps.append(i * sp * 10)
                blackreward.append(tempb[i])
            whitereward.append(tempw[i])
            targetReward.append(ALLtargetReward[path])

            if i < lr:
                nsteps.append(i * sp * 10)
                trainsteps.append(i * sp * 10)
                realreward.append(tempr[i])
        # bsteps = allsteps
        wsteps = allsteps


    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 13}
    plt.rc('font', **font)
    fig, ax = plt.subplots(1, 1)
    #plt.ticklabel_format(style='sci', scilimits=(0,0))
    print(len(nsteps), len(realreward))
    ax.plot(nsteps, realreward, color=colors[0], linestyle='-',linewidth=1)
    ax.plot(bsteps, blackreward, color=colors[1], linestyle='-',linewidth=1)
    if path != "./t/SAC":
        ax.plot(wsteps, whitereward, color=colors[2],linestyle='-',linewidth=1)
    ax.plot(allsteps, targetReward, color=colors[3], linestyle=':',linewidth=3)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.legend(labels=["Attack-free reward","LCBT attack reward", "Oracle attack reward","Target policy reward"], fontsize=15,loc='lower right')#, ncol=3)
    ax.set_xlabel("Time steps",fontsize = 18)#,fontsize=18)
    ax.set_ylabel("Average Episodic Reward",fontsize = 18)#,fontsize=18)

    parts = path.split('/')
    ax.set_title(parts[-1].upper(), fontsize=18)
    plt.savefig(path+"/reward.pdf")
    plt.show()

import math
import numpy as np
import random
from copy import deepcopy
class TreeNode:
    def __init__(self, s_class, min_a, max_a,deep):
        self.B = [0. for s in range(s_class)]
        self.Q = [0. for s in range(s_class)]
        self.L = [float('-inf') for s in range(s_class)]
        self.a_range = [min_a, max_a]
        self.a_mean = (min_a + max_a) / 2.0
        self.vp = self.getDistance(min_a, max_a) * 1.1
        self.T = 0
        self.deep = deep
        self.left = None
        self.right = None

    @staticmethod
    def getDistance(a, b):
        dist = math.sqrt(sum([(xi - yi) ** 2 for xi, yi in zip(a, b)]))
        return dist

class ActionTree:
    def __init__(self, s_classes,a_dim,min_a,max_a, H):
        self.s_classes = s_classes
        self.a_dim = a_dim
        self.min_a = min_a
        self.max_a = max_a
        self.H = H
        self.treeHead = [TreeNode(self.s_classes, self.min_a, self.max_a, 0) for i in range(self.H)]
        self.nodeNums = H

    def WorTraverse(self,h, state_id):
        htreeHead = self.treeHead[h]
        P = []
        while(htreeHead.left != None and htreeHead.right != None):
            if(htreeHead.left.B[state_id] <= htreeHead.right.B[state_id]):
                P.append(0)
                htreeHead = htreeHead.left
            else:
                P.append(1)
                htreeHead = htreeHead.right
        return htreeHead,P


    def getActionNode(self,orAciton,h):
        htreeHead = self.treeHead[h]
        P = []
        tree_h = 0
        while(htreeHead.left != None and htreeHead.right != None):
            if(orAciton[tree_h % self.a_dim] < htreeHead.left.a_range[1][tree_h % self.a_dim]):
                P.append(0)
                htreeHead = htreeHead.left
            else:
                P.append(1)
                htreeHead = htreeHead.right
            tree_h += 1
        return htreeHead,P


    #tarj[i]:
    #def update(self,tarj):
    #    return 0


class Attacker:
    def __init__(self, attacker_policy, s_dim, a_dim, min_a, max_a, min_s, max_s, args):
        self.env = args.env_name
        self.s_dim = s_dim
        self.min_s = min_s
        self.max_s = max_s
        self.a_dim = a_dim
        self.min_a = min_a
        self.max_a = max_a
        self.H = args.max_episode_length
        self.p = args.p
        self.ls = args.ls
        self.rs_piece = args.rs_piece
        self.attack_method = args.attack_method
        self.attacker_policy = attacker_policy
        # The states are classified to obtain the center point and the total number of types for each subcategory
        self.s_classes, self.rs = self.getStateClassNum(self.rs_piece, self.s_dim, self.min_s, self.max_s)
        # self.ra = self.getra(self.s_classes,args.ra_base_rs, self.a_dim, self.min_a, self.max_a)
        self.ra = self.getra_(args.ra_piece, self.a_dim, self.min_a, self.max_a)
        self.v = args.multiples_of_v * self.getDistance(self.min_a, self.max_a) / self.ls
        # get actionTree
        self.actionTree = ActionTree(self.s_classes, self.a_dim, self.min_a, self.max_a, self.H)

        self.delta = args.delta
        self.K = 1
        self.lrs = args.lrs
        self.no_attack_num = 0
        self.similarity = []

        self.isWeak = args.isWeak
        self.beginAttackK = args.beginAttackK
        random.seed(args.attack_random_seed)
        np.random.seed(args.attack_random_seed)
        print(args.multiples_of_v)
        print(self.getDistance(self.min_a, self.max_a))
        print("s_classes:",self.s_classes)
        print("ds:", self.rs)
        print("ra:", self.ra)
        print("v:", self.v)
    def antiAction(self, orAciton, h, state):
        tarAction = self.attacker_policy(state)

        # The target action is similar to the original action
        if self.K < self.beginAttackK or self.isSimAction(tarAction,orAciton):
            self.no_attack_num += 1
            return orAciton, 1

        # The target action is not similar to the original action
        else:
            if self.isWeak == True and random.random() > 1.0 / self.H:
                return tarAction, self.H / (self.H - 1)

            if self.attack_method == 'black':
                state_id = self.similarStateId(state)
                warActionNode,_ = self.actionTree.WorTraverse(h, state_id)
                action = []
                for i in range(len(warActionNode.a_range[0])):
                    action.append(random.uniform(warActionNode.a_range[0][i], warActionNode.a_range[1][i]))
                #print("attack:", action)
            elif self.attack_method == 'white' and self.env == 'ControlSlideEnv':
                if state[0] < 0:
                    action = [-1.0]
                else:
                    action = [1.0]
            elif self.attack_method == 'white' and \
                    (self.env == 'CarFindFlagMEnv' or self.env == 'CarFindFlag3MEnv' or self.env == 'CarFindFlag5MEnv'):
                action = [0.0] * self.a_dim
                if self.K > self.beginAttackK:
                    for i in range(self.a_dim):
                        if state[i] <= 4.0:
                            action[i] = -1.0
                        else:
                            action[i] = 1.0
                # if self.K > self.beginAttackK:
                #     for i in range(self.a_dim):
                #         if state[i] <= 4.0:
                #             action[i] = -1.0
                #         else:
                #             action[i] = 1.0
                # else:
                #     action = orAciton
            if self.isWeak == True:
                return action, 0.0
            else:
                return action, 0.0

    def getStateClassNum(self, rs_piece, s_dim, min_s, max_s):
        s_classes = rs_piece ** s_dim
        n = []
        f = []
        for i in range(len(min_s)):
            f.append((max_s[i]-min_s[i])/rs_piece)
            n.append([0.0])
        return s_classes, self.getDistance(n, f)

    def getra(self,s_classes,ra_base_rs, a_dim, min_a, max_a):
        single = int(pow(s_classes * ra_base_rs, 1.0 / a_dim))
        ra = self.getDistance(min_a,max_a) / single
        return ra

    def getra_(self,ra_piece, a_dim, min_a, max_a):
        a_classes = ra_piece ** a_dim
        n = []
        f = []
        for i in range(len(min_a)):
            f.append((max_a[i] - min_a[i]) / ra_piece)
            n.append([0.0])
        return self.getDistance(n,f)

    # Returns the distance between two vectors
    def getDistance(self,a,b):
        dist = math.sqrt(sum([(xi - yi) ** 2 for xi, yi in zip(a, b)]))
        return dist

    def isSimAction(self,taraction,oraction):
        b = self.getDistance(taraction,oraction) < self.ra * self.lrs
        return b

    def similarStateId(self, state):
        simi = 0
        for i in range(self.s_dim):
            d1 = self.max_s[i] - self.min_s[i]
            d2 = state[i] - self.min_s[i]
            b = d2 // (d1 / self.rs_piece)
            if b == self.rs_piece and self.rs_piece > 0:
                b -= 1
            simi = int(simi * self.rs_piece + b)
        return simi


    # [tarAction,reward,state,next_state,wh]
    def update(self, tarj):
        self.K += 1
        if self.K <= self.beginAttackK:
            return
        #self.actionTree.update(tarj)
        if self.K % 10 == 0:
            print(self.no_attack_num, len(tarj), self.actionTree.nodeNums)
        self.similarity.append([self.no_attack_num,len(tarj), self.actionTree.nodeNums])
        self.no_attack_num = 0
        Rho = 1.0
        G = 0.0
        tH = len(tarj) - 1
        while(self.attack_method == 'black' and tH >= 0):
            action = tarj[tH][0]
            reward = tarj[tH][1]
            '''
                reward 
                todo
            '''
            state = tarj[tH][2]
            next_state = tarj[tH][3]
            wh = tarj[tH][4]

            actionNode, p = self.actionTree.getActionNode(action,tH)
            actionNode.T += 1
            t = actionNode.T
            state_id = self.similarStateId(state)

            actionNode.Q[state_id] = (1 - 1/t) * actionNode.Q[state_id] + (1 / t) * (reward + G * Rho)
            Rho = Rho * wh
            G = G + reward
            if(actionNode.left == None and actionNode.right == None and self.canSegment(tH,t,actionNode.deep)):
            # if (actionNode.left == None and actionNode.right == None and self.canSegmentByself(tH, t, actionNode)):

                left = deepcopy(actionNode.a_range[0])
                right = deepcopy(actionNode.a_range[1])
                axis = actionNode.deep % self.a_dim
                mean = actionNode.a_range[0][axis] + actionNode.a_range[1][axis]
                mean /= 2.0
                left[axis] = mean
                right[axis] = mean
                actionNode.left = TreeNode(self.s_classes,actionNode.a_range[0],right,actionNode.deep + 1)
                actionNode.right = TreeNode(self.s_classes,left,actionNode.a_range[1],actionNode.deep + 1)
                self.actionTree.nodeNums += 2

            self.updateLB(self.actionTree.treeHead[tH],state_id,tH)

            tH -= 1

        if self.K % 2000 == 0 and self.lrs > 1:
            self.lrs -= 1

    def updateLB(self,treeHead,state_id,h):
        if(treeHead == None):
            return None
        self.updateLB(treeHead.left,state_id,h)
        self.updateLB(treeHead.right,state_id,h)

        tHeadL = treeHead.left
        tHeadR = treeHead.right
        if treeHead.T != 0:
            treeHead.L[state_id] = treeHead.Q[state_id] - self.bound_Hh(h,treeHead.T) - self.ls * self.rs - self.v * pow(self.p,treeHead.deep)
            # treeHead.L[state_id] = treeHead.Q[state_id] - self.bound_Hh(h, treeHead.T) - \
            #                        self.ls * self.rs - \
            #                        self.v * pow(self.p, treeHead.deep) * (self.H - h + 1)
            # treeHead.L[state_id] = treeHead.Q[state_id] - self.bound(h, treeHead.T) - self.ls * self.rs - treeHead.vp

        if tHeadL == None:
            treeHead.B[state_id] = treeHead.L[state_id]
        else:
            treeHead.B[state_id] = max(treeHead.L[state_id],min(tHeadL.B[state_id],tHeadR.B[state_id]))


    def bound(self,h,t):
        # left = (self.H - h + 1) / math.sqrt(2 * t)
        left = 1 / math.sqrt(2 * t)
        right_in = (2 * self.s_classes * self.K * self.actionTree.nodeNums) / self.delta
        right = math.sqrt(math.log(right_in))
        return left * right

    def bound_Hh(self,h,t):
        left = (self.H - h + 1) / math.sqrt(2 * t)
        right_in = (2 * self.s_classes * self.K * self.actionTree.nodeNums) / self.delta
        right = math.sqrt(math.log(right_in))
        return left * right

    def canSegment(self,h,t,d):
        r = self.bound_Hh(h,t)
        l = self.v * pow(self.p, d)
        return l >= r

    def canSegmentByself(self,h, t, node):
        r = self.bound_Hh(h, t)
        l = node.vp
        return l >= r
import copy
import matplotlib.pyplot as plt
import numpy as np
from Actor_Critic_for_JSP.Instance_Generator import Generate

def Gantt(Machines):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    M = ['red', 'blue', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta',
         'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
         'navajowhite', 'navy', 'sandybrown', 'moccasin']
    Job_text = ['J' + str(i + 1) for i in range(100)]
    Machine_text = ['M' + str(i + 1) for i in range(50)]

    for i in range(len(Machines)):
        for j in range(len(Machines[i].start)):
            if Machines[i].finish[j] - Machines[i].start[j]!= 0:
                plt.barh(i, width=Machines[i].finish[j] - Machines[i].start[j],
                         height=0.8, left=Machines[i].start[j],
                         color=M[Machines[i]._on[j]],
                         edgecolor='black')
                plt.text(x=Machines[i].start[j]+(Machines[i].finish[j] - Machines[i].start[j])/2 - 0.1,
                         y=i,
                         s=Job_text[Machines[i]._on[j]],
                         fontsize=12)
    plt.show()

class Machine:
    def __init__(self,idx):
        self.idx=idx
        self.start=[]
        self.finish=[]
        self._on=[]
        self.end=0

    def handling(self,Ji,pt):
        s=self.insert(Ji,pt)
        # s=max(Ji.end,self.end)
        e=s+pt
        self.start.append(s)
        self.finish.append(e)
        self.start.sort()
        self.finish.sort()
        self._on.insert(self.start.index(s),Ji.idx)
        # self._on.append(Ji.idx)
        if self.end<e:
            self.end=e
        Ji.update(s,e)

    def Gap(self):
        Gap=0
        if self.start==[]:
            return 0
        else:
            Gap+=self.start[0]
            if len(self.start)>1:
                G=[self.start[i+1]-self.finish[i] for i in range(0,len(self.start)-1)]
                return Gap+sum(G)
            return Gap

    def judge_gap(self,t):
        Gap = []
        if self.start == []:
            return Gap
        else:
            if self.start[0]>0 and self.start[0]>t:
                Gap.append([0,self.start[0]])
            if len(self.start) > 1:
                Gap.extend([[self.finish[i], self.start[i + 1]] for i in range(0, len(self.start) - 1) if
                            self.start[i + 1] - self.finish[i] > 0 and self.start[i + 1] > t])
                return Gap
            return Gap

    def insert(self,Ji,pt):
        start=max(Ji.end,self.end)
        Gap=self.judge_gap(Ji.end)
        if Gap!=[]:
            for Gi in Gap:
                if Gi[0]>=Ji.end and Gi[1]-Gi[0]>=pt:
                    return Gi[0]
                elif Gi[0]<Ji.end and Gi[1]-Ji.end>=pt:
                    return Ji.end
        return start

class Job:
    def __init__(self,idx,max_ol):
        self.idx=idx
        self.start=0
        self.end=0
        self.op=0
        self.max_ol=max_ol
        self.Gap=0
        self.l=0
        self.T=[]

    def wether_end(self):
        if self.op<self.max_ol:
            return False
        else:
            return True

    def update(self,s,e):
        self.op+=1
        self.end=e
        self.start=s
        self.l=self.l+e-s
        self.T.append(e-s)

class JSP_Env:
    def __init__(self,n,m,PT,MT):
        self.PT=PT
        self.M=MT
        self.n,self.m=n,m
        self.g=0

    def Create_Item(self):
        self.Jobs=[]
        for i in range(self.n):
            Ji=Job(i,len(self.PT[i]))
            self.Jobs.append(Ji)
        self.Machines=[]
        for i in range(self.n):
            Mi=Machine(i)
            self.Machines.append(Mi)

    def C_max(self):
        m=0
        for Mi in self.Machines:
            if Mi.end>m:
                m=Mi.end
        return m

    def s3(self):
        C_max=float(self.C_max())
        for Ji in self.Jobs:
            for i in range(len(Ji.T)):
                self.s[2][Ji.idx][i]=float(self.M_t[self.M[Ji.idx][i]])/C_max


    def reset(self,):
        # self.PT,self.M = Generate(self.n,self.m)
        self.O_max_len = len(self.PT[0])
        self.u=0
        self.P = 0  # total working time
        self.finished=[]
        self.Num_finished=0
        self.M_t = [0 for _ in range(self.m)]
        done=False
        self.Create_Item()
        self.S1_Matrix = np.array(copy.copy(self.PT),dtype=float)
        self.S2_Matrix = np.zeros_like(self.S1_Matrix)
        self.S3_Matrix = np.zeros_like(self.S1_Matrix)
        self.s=np.stack((self.S1_Matrix,self.S2_Matrix,self.S3_Matrix),0)
        # self.s = np.stack((self.S1_Matrix, self.S2_Matrix), 0)
        # s=self.s.flatten()
        return copy.copy(self.s),done

    def Gap(self):
        G=0
        for Mi in self.Machines:
            G+=float(Mi.Gap())
        return float(G/self.C_max())

    def U(self):
        C_max = self.C_max()
        return self.P/(self.m*C_max)

    def step(self,action):

        # print(action)
        done=False
        if action in self.finished:
            return self.s,-999,done
        Ji=self.Jobs[action]
        op=Ji.op
        # print('a',action,op)
        pt=self.PT[action][op]
        self.P+=pt
        self.s[0][action][op] = 0
        Mi=self.Machines[self.M[action][op]]
        self.M_t[Mi.idx]+=pt
        Mi.handling(Ji,pt)
        self.s[1][action][op]=Ji.end
        if Ji.wether_end():
            self.finished.append(action)
            self.Num_finished+=1
        if self.Num_finished==self.n:
            done=True
        Gap=self.Gap()
        self.s3()
        # print(self.s[2])
        u=self.U()
        r=u-self.u
        self.u=u
        # s=self.s.flatten()
        return self.s,r,done


if __name__=="__main__":

    from Dataset.data_extract import change
    from Actor_Critic_for_JSP.action_space import Dispatch_rule
    from Actor_Critic_for_JSP.Instance_Generator import Generate
    import random

    jsp=JSP_Env(6,6)


    s,done=jsp.reset()
    PT=jsp.PT
    os1 = []
    for i in range(len(PT)):
        for j in range(len(PT[i])):
            os1.append(i)
    while not done:
        a=random.randint(0,16)
        a=Dispatch_rule(a,jsp)
        s, r, done=jsp.step(a)
        # print(s)
        os1.remove(a)
    Gantt(jsp.Machines)
    print(jsp.C_max())









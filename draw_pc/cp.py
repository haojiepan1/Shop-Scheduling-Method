import os

import seaborn as sns; sns.set()
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set() # 因为sns.set()一般不用改，可以在导入模块时顺便设置好


def Algo_Compare(path):
    Data=[]
    files=os.listdir(path)
    for file in files:
        # F1=os.listdir(path+'/'+file)
        # for f in F1:
        k=path+'/'+file
        with open(k, "rb") as fb:
            d = np.array(pickle.load(fb))
            print(1)
            d=d.T
            df = pd.DataFrame(d).melt(var_name='episode', value_name='reward')

        Data.append(df)
    # print(Data)

    # sns.lineplot(x='episode',y='reward',data=Data[0][0])
    sns.lineplot(x='episode', y='reward', data=Data[0])
    # plt.xticks([0,20,40,60,80,100], [0,1000,2000,3000,4000,5000])
    plt.show()
    # sns.lineplot(x='episode', y='reward', data=Data[0][1])
    sns.lineplot(x='episode', y='reward', data=Data[1])
    # plt.xticks([0, 20, 40, 60, 80, 100], [0, 1000, 2000, 3000, 4000, 5000])
    plt.show()

    # for i in range(len(C)):


file=r'C:\Users\Administrator\PycharmProjects\MADRL_for_-two_AGVs\Actor_Critic_for_JSP\result\la\la16'
Algo_Compare(file)
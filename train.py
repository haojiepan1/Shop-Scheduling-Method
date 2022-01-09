import copy

from Actor_Critic_for_JSP.JSP_env import JSP_Env,Gantt
import matplotlib.pyplot as plt
from Actor_Critic_for_JSP.Dataset.data_extract import change
from Actor_Critic_for_JSP.action_space import Dispatch_rule
from Actor_Critic_for_JSP.Agent.Agent import Agent
import pickle

def main(Agent,env,batch_size,file):

    Reward_total = []
    C_total = []
    rewards_list = []

    C = []
    L=[]
    episodes = 10000
    print("Collecting Experience....")
    for i in range(episodes):
        print(i)
        state,done = env.reset()
        ep_reward = 0
        while True:

            action = Agent.choose_action(state)
            # print(action)
            a=Dispatch_rule(action,env)
            next_state, reward, done = env.step(a)
            Agent.store_transition(state, action, reward, next_state)
            ep_reward += reward
            if Agent.memory_counter >= batch_size:
                loss=Agent.learn()
                L.append(loss)

                if done and i%10==0:
                    ret, f, C1, R1 = evaluate(i,Agent,env,loss)
                    Reward_total.append(R1)
                    C_total.append(C1)
                    rewards_list.append( ep_reward)
                    C.append(env.C_max())
            if done:
                break
            state = copy.copy(next_state)
    Agent.save_model(file)
    x = [_ for _ in range(len(C))]
    plt.subplot(221)
    plt.plot(x, rewards_list)
    plt.subplot(222)
    # plt.plot(x, C)
    # plt.subplot(333)
    y=[_ for _ in range(len(L))]
    plt.plot(y,L)
    result={'reward':rewards_list,'C':C,'Loss':L}
    f1=file+'.pkl'
    with open(f1,'wb') as fa:
        pickle.dump(result, fa, pickle.HIGHEST_PROTOCOL)
    f2=file+'.png'
    with open(f2, 'wb') as fb:
        plt.savefig(fb, dpi=600, bbox_inches='tight')
    plt.close()
    return Reward_total,C_total

def evaluate(i,Agent,env,loss):
    returns = []
    C=[]
    for  total_step in range(4):
        state, done = env.reset()
        ep_reward = 0
        while True:
            action = Agent.choose_action(state)
            a = Dispatch_rule(action, env)
            next_state, reward, done = env.step(a)
            ep_reward += reward
            if done == True:
                fitness = env.C_max()
                C.append(fitness)
                break
        returns.append(ep_reward)
    print('time step:',i,'','Reward ：',sum(returns)/4 ,'','C_max:',sum(C) /4,' ','loss:',loss)
    return sum(returns) / 4,sum(C) /4,C,returns


if __name__ == '__main__':
    from Actor_Critic_for_JSP.Dataset.data_extract import change
    n, m, PT, MT = change('ft', 6)
    import pickle
    import os
    env = JSP_Env( n, m, PT, MT)
    f=r"./result/ft06"
    if not os.path.exists(f):
        os.mkdir(f)
    f2 = r"./Model/ft06"
    if not os.path.exists(f2):
        os.mkdir(f2)
    net_update=[100,200,300,400,500]
    learning_rate=[0.00001,0.000001,0.0000001]
    Memory_sieze=[10000,50000,100000,500000,1000000]
    batch_size=[128,256,512]
    Gamma=[0.8,0.82,0.85,0.87,0.90,0.92,0.95,0.98,1]                              #gamma越大Agent向后考虑的步数越多，反之越注重眼前利益。1/(1-gamma)来估计决策依据的有效范围。
    dueling,DOUBLE,PER=1,1,1

    for Gi in Gamma:
        for ni in net_update:
            for li in learning_rate:
                for Mi in Memory_sieze:
                    for bi in  batch_size:
                        #(1,1,1)分别表示dueling_network,double,PER
                        agent = Agent(env.n, 6, Gi, ni,li,Mi,bi,dueling,DOUBLE,PER)
                        fb=os.path.join(f,str(dueling)+str(DOUBLE)+str(PER)+str(Gi)+'_'+str(ni)+'_'+str(li)+'_'+str(Mi)+'_'+str(bi))
                        print(fb)
                        Reward_total,C_total=main(agent,env,100,fb)

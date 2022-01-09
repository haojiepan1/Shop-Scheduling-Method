import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Actor_Critic_for_JSP.Agent.RL_network import CNN_FNN,CNN_dueling
from Actor_Critic_for_JSP.Memory.Memory import Memory
from Actor_Critic_for_JSP.Memory.PreMemory import preMemory

class Agent():
    """docstring for DQN"""
    def __init__(self,n,O_max_len,Gamma,net_update,learning_rate,M_size,B_size,dueling=False,double=False,PER=False,model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.es=8000*n*O_max_len
        self.record=0
        self.double=double
        self.PER=PER
        self.GAMMA=Gamma
        self.n=n
        self.O_max_len=O_max_len
        super(Agent, self).__init__()
        if dueling:
            self.eval_net, self.target_net = CNN_dueling(self.n,self.O_max_len).to(self.device), CNN_dueling(self.n,self.O_max_len).to(self.device)
        else:
            self.eval_net, self.target_net = CNN_FNN(self.n, self.O_max_len).to(self.device), CNN_FNN(self.n, self.O_max_len).to(self.device)
        self.Q_NETWORK_ITERATION=net_update
        self.BATCH_SIZE=B_size
        self.learn_step_counter = 0
        self.memory_counter = 0
        if PER:
            self.memory = preMemory(M_size)
        else:
            self.memory = Memory(M_size)
        self.Min_EPISILO=0.005
        self.EPISILO=1
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        if model_path!=None:
            if os.path.exists(model_path + '/eval_net.pkl'):
                self.eval_net.load_state_dict(torch.load(model_path + '/eval_net.pkl'))
                self.target_net.load_state_dict(torch.load(model_path + '/target_net.pkl'))

    def choose_action(self, state):
        self.record+=1
        state=np.reshape(state,(-1,3,self.n,self.O_max_len))
        state=torch.FloatTensor(state).to(self.device)
        # print(state.size())
        # state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() > self.EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,17)
            # action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        self.EPISILO=1-(1-self.Min_EPISILO)*min(1,(self.record/self.es))
        # print((1-self.Min_EPISILO)*min(1,(self.record/self.es)),self.record/self.es,self.EPISILO)
        return action

    def PER_error(self,state, action, reward, next_state):

        state = torch.FloatTensor(np.reshape(state, (-1, 3, self.n, self.O_max_len))).to(self.device)
        next_state= torch.FloatTensor(np.reshape(next_state, (-1, 3, self.n, self.O_max_len))).to(self.device)
        p=self.eval_net.forward(state)
        p_=self.eval_net.forward(next_state)
        p_target=self.target_net(state)

        if self.double:
            q_a=p_.argmax(dim=1)
            q_a=torch.reshape(q_a,(-1,len(q_a)))
            qt=reward+self.GAMMA*p_target.gather(1,q_a)
        else:
            qt=reward+self.GAMMA*p_target.max()
        qt=qt.detach().cpu().numpy()
        p=p.detach().cpu().numpy()
        errors=np.abs(p[0][action]-qt)
        return errors

    def store_transition(self, state, action, reward, next_state):
        # print(reward)
        if self.PER:
            errors=self.PER_error(state, action, reward, next_state)
            self.memory.remember((state, action, reward, next_state), errors)
            self.memory_counter += 1
        else:
            self.memory.remember((state, action, reward, next_state))
            self.memory_counter+=1

    def learn(self):

        # update the parameters
        if self.learn_step_counter % self.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.PER:
            [batch, batch_indices, batch_priorities] = self.memory.sample(self.BATCH_SIZE)
            batch_priorities=torch.Tensor(batch_priorities).to(self.device)
        else:
            batch = self.memory.sample(self.BATCH_SIZE)

        #sample batch from memory
        batch_state=np.array([o[0] for o in batch])
        batch_next_state= np.array([o[3] for o in batch])
        batch_action=np.array([o[1] for o in batch])
        batch_reward=np.array([o[2] for o in batch])
        # print(batch_reward)
        # print( )
        batch_action = torch.LongTensor(np.reshape(batch_action, (-1, len(batch_action)))).detach().to(self.device)
        batch_reward =  torch.FloatTensor(np.reshape(batch_reward, (-1, len(batch_reward)))).to(self.device)
        # print(batch_action)

        batch_state=torch.FloatTensor(np.reshape(batch_state, (-1, 3, self.n, self.O_max_len))).to(self.device)
        batch_next_state =torch.FloatTensor(np.reshape(batch_next_state, (-1, 3, self.n, self.O_max_len))).to(self.device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action).to(self.device)
        q_next = self.target_net(batch_next_state).detach().to(self.device)
        if self.double:
            q_next_eval=self.eval_net( batch_next_state).detach()
            q_a=q_next_eval.argmax(dim=1)
            q_a=torch.reshape(q_a,(-1,len(q_a))).to(self.device)
            q_target = batch_reward + self.GAMMA * q_next.gather(1, q_a)
        else:
            q_target = batch_reward + self.GAMMA * q_next.max(1)[0]

        if self.PER:
            errors=torch.abs(q_eval-q_target).to(self.device)
            loss=(batch_priorities*errors**2).sum()
            for Bi in range(self.BATCH_SIZE):
                index=batch_indices[Bi]
                self.memory.update(index,errors[0][Bi].detach().cpu().numpy())
        else:
            loss = self.loss_func(q_eval, q_target).to(self.device)

        l=loss.detach().cpu().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return l

    def save_model(self,file):
        if not os.path.exists(file):
            os.makedirs(file)
        torch.save(self.eval_net.state_dict(), file +'/' +'eval_net.pkl')
        torch.save(self.target_net.state_dict(), file +'/'+'target_net.pkl')
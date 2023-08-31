import torch 
import numpy as np
from rl_utils import onehot_from_logits, gumbel_softmax
from model import FCwithGRU

# single agent actor网络 for simulation
class Agent: # 训练使用MADDPG，测试时只需要使用actor而不需要critic
    def __init__(self, state_dim, action_dim, hidden_dim_a, device, memory_len, id) -> None:
        # self.alg = DDPG(state_dim, action_dim, critic_input_dim, hidden_dim_a,hidden_dim_c,
        #          actor_lr, critic_lr, device).actor # 不需要critic网络
        self.alg = FCwithGRU(state_dim, action_dim, hidden_dim_a).to(device) # actor网络
        self.memory_len = memory_len # 5
        self.state_space = 6 # 4个状态 + D2LT
        self.STATES = np.eye(4) # 自己成功传 1000；别人成功传0100； 碰撞0010；空闲0001
        self.device = device
        self.D2LT = [0, 0] # 0: 多久没有传， 1：其他agent多久没有传
        self.id = id
        self.reward_log = []
        self.states_mem = np.zeros(self.state_space * self.memory_len)
        self.last_action = [1, 0]
        assert id > 0, "agent id must > 0"
        
    def reset(self):
        self.reward_log.clear()
        self.last_action = [1, 0] 
        self.D2LT = [0, 0]
        self.states_mem = np.zeros(self.state_space * self.memory_len)
        
    @property
    def normed_d2lt(self):
        ret = [0.0, 0.0]
        sum = self.D2LT[0] + self.D2LT[1]
        ret[0] = self.D2LT[0] / (sum + 1e-12)
        ret[1] = self.D2LT[1] / (sum + 1e-12)
        return ret
    
    
    def update_states_mem(self, link_ret):
        self.updateD2LT(link_ret['occupy_id'])
        # print(self.last_action)
        if self.last_action[0] == 1: # 时隙开始时，没发送消息
            if link_ret['occupy_id'] == 0: # 空闲 
                link_obs = self.STATES[3]
            elif link_ret['occupy_id'] < 0: # 碰撞
                link_obs = self.STATES[2]
            else:
                link_obs = self.STATES[1]

        elif self.last_action[1] == 1: # 时隙开始时，发送了消息
            if link_ret['occupy_id'] == self.id: # 自己发送成功
                link_obs = self.STATES[0]
            elif link_ret['occupy_id'] > 0:
                link_obs = self.STATES[1] # 别人发送成功
            else:
                link_obs = self.STATES[2] # 碰撞
        self.states_mem = np.concatenate([self.states_mem[6:], link_obs, self.normed_d2lt])
            
            
    def take_action(self, explore=False):
        ''' eps表示随机动作生成的概率，默认为0.0%'''
        # 读取各个agent的state,并转变格式
        obs = torch.tensor(np.array([self.states_mem]), dtype=torch.float, device=self.device)
        action = self.alg(obs)  # actor网络将传入进来的state转换为动作
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action, eps=0.001)
        # detach(): 返回一个新的Tensor，但返回的结果是没有梯度的;numpy()将tensor转变为数组；[0]相当于去掉一个[]
        self.last_action = action.detach().cpu().numpy()[0]
        return self.last_action
    
    def updateD2LT(self, reward):
        '''
        reward: 0 表示发生碰撞或者空闲，即所有agent都没有发送成功
        :非0 表示agent[i]发送消息成功
        '''
        if reward == self.id:
            self.D2LT[0] = 0
            self.D2LT[1] += 1
            self.reward_log.append(1)
        elif reward > 0:
            self.D2LT[0] += 1
            self.D2LT[1] = 0
            self.reward_log.append(0)
        else: # == 0 or < 0 空闲 or 碰撞
            self.D2LT[0] += 1
            self.D2LT[1] += 1
            self.reward_log.append(0)
        


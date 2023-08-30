import torch 
import numpy as np
from rl_utils import onehot_from_logits, gumbel_softmax
from model import FCwithGRU

# single agent actor网络 for simulation
class Agent: # 训练使用MADDPG，测试时只需要使用actor而不需要critic
    def __init__(self, state_dim, action_dim, hidden_dim_a, device, id) -> None:
        # self.alg = DDPG(state_dim, action_dim, critic_input_dim, hidden_dim_a,hidden_dim_c,
        #          actor_lr, critic_lr, device).actor # 不需要critic网络
        self.alg = FCwithGRU(state_dim, action_dim, hidden_dim_a).to(device) # actor网络
        self.device = device
        self.D2LT = [0, 0] # 0: 多久没有传， 1：其他agent多久没有传
        self.id = id
        self.reward_log = []
        assert id > 0, "agent id must > 0"
        
    def reset(self):
        self.reward_log.clear()
        self.D2LT = [0, 0]
        
    @property
    def normed_d2lt(self):
        ret = [0.0, 0.0]
        sum = self.D2LT[0] + self.D2LT[1]
        ret[0] = self.D2LT[0] / (sum + 1e-9)
        ret[1] = self.D2LT[1] / (sum + 1e-9)
        return ret
    
    def take_action(self, obs, explore=False):
        ''' 传入各个agent的state总和，返回一个列表，里面有4个元素，分别表示4个agent的action数组 '''
        # 读取各个agent的state,并转变格式
        obs = torch.tensor(np.array([obs]), dtype=torch.float, device=self.device)
        action = self.alg(obs)  # actor网络将传入进来的state转换为动作
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        # detach(): 返回一个新的Tensor，但返回的结果是没有梯度的;numpy()将tensor转变为数组；[0]相当于去掉一个[]
        return action.detach().cpu().numpy()[0]
    
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
        else:
            self.D2LT[0] += 1
            self.D2LT[1] += 1
            self.reward_log.append(0)
        


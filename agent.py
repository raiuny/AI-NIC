import torch 
import numpy as np
from rl_utils import onehot_from_logits, gumbel_softmax
from model import FCwithGRU

def update_D2LT(reward,agent_number,V, V_):
    '''
    输入各个agent的reward来更新各个agent有多少个时隙没有传输了
    param reward：各个agent的传输情况
    param agent_number：多少个agent
    param V_l1：上个时隙记录的l1上的各agent的D2LT
    param V_l1_：上个时隙记录的l1上的v(-i)
    '''
    V = [x + 1 for x in V]  # 先所有D2LT加1，后续再对已经传输agent的赋0
    V_ = [x + 1 for x in V_]
    for i in range(agent_number):
        if reward[i] == 1:  # 如果i号agent在link1上成功传输
            V[i] = 0
            temp = V_.copy()
            V_ = np.zeros([agent_number, ])  # i号agent传输，则其余节点对应的v-i为0
            V_[i] = temp[i]

    return V, V_

def normalize_D2LT(V, V_):
    '''
    输入D2LT V得到标准化的di和d-i
    '''
    LEN = len(V_)
    d, d_ = np.zeros([LEN, ]),np.zeros([LEN, ]) # 初始化Di和D-i数组
    for i in range(len(V)):
        d[i] = V[i]/(V[i]+V_[i])
        d_[i] = V_[i]/(V[i]+V_[i])
    return d, d_

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
        


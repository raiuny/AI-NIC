from algorithm import DDPG
import torch 
import numpy as np
from rl_utils import onehot_from_logits, gumbel_softmax

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

# single agent actor网络
class Agent: # 训练使用MADDPG，测试时只需要使用actor而不需要critic
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim_a,hidden_dim_c,
                 actor_lr, critic_lr, device, id) -> None:
        self.actor = DDPG(state_dim, action_dim, critic_input_dim, hidden_dim_a,hidden_dim_c,
                 actor_lr, critic_lr, device).actor
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.D2LT = [0, 0] # 0: 多久没有传， 1：其他agent多久没有传
        self.id = id
        self.reward_log = []
        assert id > 0, "agent id must > 0"
        
    def reset(self):
        self.reward_log.clear()
        self.D2LT = [0, 0]
        
    @property
    def d2lt_norm(self):
        ret = [0.0, 0.0]
        sum = self.D2LT[0] + self.D2LT[1]
        ret[0] = self.D2LT[0] / (sum + 1e-9)
        ret[1] = self.D2LT[1] / (sum + 1e-9)
        return ret
    
    def take_action(self, obs, explore=False):
        ''' 传入各个agent的state总和，返回一个列表，里面有4个元素，分别表示4个agent的action数组 '''
        # 读取各个agent的state,并转变格式
        obs = torch.tensor(np.array([obs]), dtype=torch.float, device=self.device)
        return self.actor.take_action(obs, explore)
    
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
        
# multi-agent DDPG  
class MADDPG:
    def __init__(self, agent_num, device, actor_lr, critic_lr, hidden_dim_a, hidden_dim_c,
                 state_dim, action_dim, critic_input_dim, gamma, tau):
        '''

        '''
        self.agent_num = agent_num
        self.agents = []  # 存放各个agent的列表
        for i in range(agent_num):  # 每个agent执行DDPG
            self.agents.append(
                DDPG(state_dim[i], action_dim[i], critic_input_dim,
                     hidden_dim_a,hidden_dim_c, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        
    @property
    def policies(self):  # 返回各个actor网络
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):  # 返回各个target_actor网络
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        ''' 传入各个agent的state总和，返回一个列表，里面有4个元素，分别表示4个agent的action数组 '''
        states = [
            torch.tensor(np.array([states[i]]), dtype=torch.float, device=self.device)
            for i in range(self.agent_num)
        ]   # 读取各个agent的state,并转变格式
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        '''
        MADDPG更新
        :param sample: 采样的历史数据，包含五项
                 （[4个agent的观测值][4个agent的动作][4个agent的reward][4个agent的下一步观测值] # [4个agent是否结束]）
        :param i_agent:agent的序号
        :return:
        '''
        obs, act, rew, next_obs = sample  # 取出相关经验
        cur_agent = self.agents[i_agent]  # 取出对应的agent

        # 针对critic和target_critic网络
        cur_agent.critic_optimizer.zero_grad()  # 清空上一步的残余更新参数值
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]  # target actor网络得到的动作
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(target_critic_input)   # 时序差分目标，计算现实的value
        critic_input = torch.cat((*obs, *act), dim=1)   # critic网络的中心化体现在传入的是所有agent的参数
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())   # 传入估计的value和现实的value计算loss
        critic_loss.backward()  # critic网络反向传播
        cur_agent.critic_optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

        # 针对actor网络
        cur_agent.actor_optimizer.zero_grad()   # 清空上一步的残余更新参数值
        cur_actor_out = cur_agent.actor(obs[i_agent])   # 传入观测得到actor网络的输出
        cur_act_vf_in = gumbel_softmax(cur_actor_out)   # 将输出转为动作
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()  # actor网络反向传播
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        ''' 软更新各个agent的target网络  '''
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

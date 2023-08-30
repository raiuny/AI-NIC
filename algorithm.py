from model import FCwithGRU, TwoLayerFC
import torch 
from rl_utils import gumbel_softmax, onehot_from_logits
import numpy as np

class DDPG:
    ''' DDPG算法,针对单个agent '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim_a,hidden_dim_c,
                 actor_lr, critic_lr, device):
        '''
        state_dim和action_dim针对actor网络，即每个agent
        state_dim=，作为actor网络的输入
        action_dim=4，4位one-hot编码，表示该agent在两条link上的动作（四种），由actor网络输出
        critic_input_dim：汇总所有agent的信息，作为critic网络的输入
        actor_lr、critic_lr：学习率
        hidden_dim_a:64
        hidden_dim_c:critic网络的宽度略大于actor,128
        '''

        # actor-critic加目标网络，共计四个网络，都是传入状态输出动作
        # actor网络是去中心化的，每个agent各自决定
        self.actor = FCwithGRU(state_dim, action_dim, hidden_dim_a).to(device)
        self.target_actor = FCwithGRU(state_dim, action_dim,
                                       hidden_dim_a).to(device)

        # 所有智能体共享一个中心化的 Critic 网络
        # critic_input_dim是所有agent的state和action维数之和，体现中心化
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim_c).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim_c).to(device)

        # 参数同步给target_net
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):
        '''
        通过传入状态采取动作
        :param state: 状态
        :param explore: 是否进行探索
        :return: 返回的是一个维度为(4,)的数组，对应四种选择link的情况
        '''
        action = self.actor(state)  # actor网络将传入进来的state转换为动作
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        # detach(): 返回一个新的Tensor，但返回的结果是没有梯度的;numpy()将tensor转变为数组；[0]相当于去掉一个[]
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau): # learn
        ''' 软更新，即让目标网络缓慢更新 '''
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

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
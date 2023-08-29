import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm

from env_singlelink import env_SL

def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """
    从Gumbel(0,1)分布中采样
    :param shape: 动作空间维数
    :param eps: 保证log里没有0
    :param tens_type: 张量类型
    :return: gumbel结果，公式里的gi
    """
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)   # (0,1)均匀分布
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)  # 给前向的输出logits引入一个gumbel样本
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)  # one-hot化
    y = (y_hard.to(logits.device) - y).detach() + y   # 引入-y.detach()+y保证值不变的同时也有反向梯度
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以正确地反传梯度
    return y


class TwoLayerFC(torch.nn.Module):
    '''
    构建一个3层的全连接网络
    '''
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class FCwithGRU(torch.nn.Module):
    '''
    构建一个4层带有GRU结构的网络
    '''
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.rnn = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)
        self.hidden = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x, self.hidden = self.rnn(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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

    def soft_update(self, net, target_net, tau):
        ''' 软更新，即让目标网络缓慢更新 '''
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim_a, hidden_dim_c,
                 state_dim, action_dim, critic_input_dim, gamma, tau):
        '''

        '''
        self.agents = []  # 存放各个agent的列表
        for i in range(env.agent_number):  # 每个agent执行DDPG
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
            for i in range(env.agent_number)
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


def return_throughput(rewards):
    N = int(len(rewards)/50)
    temp_sum = 0
    throughput = []
    for i in range(len(rewards)):
        if i < N:
            temp_sum += rewards[i]
            throughput.append(temp_sum / (i+1))  #长度不满N时，平均值为除以i+1,LTT
        else:
            temp_sum += rewards[i] - rewards[i-N]
            throughput.append(temp_sum / N)     #长度满N了，平均值就为总和除以N,STT
    return throughput

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


hidden_dim_a = 64  # actor网络隐藏层维数
hidden_dim_c = 128  # critic网络隐藏层维数
actor_lr = 5e-4  # 学习率
critic_lr = 5e-4
gamma = 0.95
tau = 1e-2   # 软更新参数，通常比较小

device = torch.device("cpu")

state_length_M = 5
env = env_SL(state_length=state_length_M)  # 创建环境

state_dims = [6*state_length_M,6*state_length_M,6*state_length_M]

action_dims = [2,2,2]

critic_input_dim = sum(state_dims) + sum(action_dims)

MODEL_PATH = ['model/3node_SL_1.pt', 'model/3node_SL_2.pt', 'model/3node_SL_3.pt']

# 实例化, 导入训练好的actor网络模型
maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim_a, hidden_dim_c, state_dims,
                action_dims, critic_input_dim, gamma, tau)
for i in range(env.agent_number):
    maddpg.agents[i].actor.load_state_dict(torch.load(MODEL_PATH[i]))

n_episode = 1
episode_length = 40000

reward1_l1 = []
reward2_l1 = []
reward3_l1 = []
# reward5_l1, reward5_l2 = [], []
V_l1, V_l1_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])  # 记录link上的D2LT
d_l1, d_l1_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])   # 小d是归一化vi和v-i
D_l1 = np.zeros([env.agent_number, ])
for _ in range(n_episode):  # 开始迭代
    state1 = env.reset()  # 每次迭代前重置环境
    next_state1 = state1.copy()
    for t_i in tqdm(range(episode_length)):
        actions1 = maddpg.take_action(state1, explore=False)  # 返回一个列表，里面有三个元素，分别表示三个agent的action数组

        link1_obs, reward_l1, l1_collision = env.step(actions1)

        # 更新D2LT
        V_l1, V_l1_ = update_D2LT(reward_l1, env.agent_number, V_l1, V_l1_)

        # 归一化Vi V-i得到d
        d_l1, d_l1_ = normalize_D2LT(V_l1, V_l1_)
        for i in range(env.agent_number):
            next_state1[i] = np.concatenate([state1[i][6:], link1_obs[i], [d_l1[i], d_l1_[i]]])

        # 记录reward
        reward1_l1.append(reward_l1[0])
        reward2_l1.append(reward_l1[1])
        reward3_l1.append(reward_l1[2])

        state1 = next_state1.copy()

l1_throughput1 = return_throughput(reward1_l1)   #agent1吞吐量
l1_throughput2 = return_throughput(reward2_l1)   #agent2吞吐量
l1_throughput3 = return_throughput(reward3_l1)   # agent1吞吐量

l1_sum_throughput = [l1_throughput1[i] + l1_throughput2[i] + l1_throughput3[i]  for i in
						 range(episode_length)]  # l1总吞吐量

mean1 = np.mean(round(l1_sum_throughput[-1000], 2))  # 最后逼近的值

fig = plt.figure(figsize=(14, 6))
# plt.subplot(3, 1, 1)
plt.plot(l1_sum_throughput, c='r', label='Sum')
plt.plot(l1_throughput1, c='b', label='agent1')
plt.plot(l1_throughput2, c='cyan', label='agent2')
plt.plot(l1_throughput3, c='orange', label='agent3')
# plt.plot(l1_throughput4, c='green', label='agent4')
# plt.plot(l1_throughput5, c='yellow', label='agent5')
plt.ylim((0, 1))
plt.xlim(0, None)
plt.xlabel("Iterations/Slots", fontsize=14)
plt.ylabel("Throughput", fontsize=14)
plt.title("link1")
plt.text(len(l1_sum_throughput) * 0.9, mean1 - 0.1, f'sum={mean1}',
         family='Times New Roman',  # 标注文本字体
         fontsize=18,  # 文本大小
         fontweight='bold',  # 字体粗细
         color='red')  # 文本颜色
plt.legend()


plt.show()
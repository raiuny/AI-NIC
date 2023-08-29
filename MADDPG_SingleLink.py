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
    构建一个3层的全连接网络，用于critic网络，128n
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
    构建一个4层带有GRU结构的网络，用于actor网络，处理时序任务
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

def update_D2LT(reward,agent_number,V_l1, V_l1_):
    '''
    输入各个agent的reward来更新各个agent有多少个时隙没有传输了
    '''
    V_l1 = [x + 1 for x in V_l1]  # 先所有D2LT加1，后续再对已经传输agent的赋0
    V_l1_ = [x + 1 for x in V_l1_]
    for i in range(agent_number):
        if reward[i] == 1:  # 如果i号agent在link1上成功传输
            V_l1[i] = 0
            temp = V_l1_.copy()
            V_l1_ = np.zeros([agent_number, ])  # i号agent传输，则其余节点对应的v-i为0
            V_l1_[i] = temp[i]

    return V_l1, V_l1_

def normalize_D2LT(V_l1, V_l1_):
    '''
    输入D2LT V得到标准化的di和d-i
    '''
    LEN = len(V_l1_)
    d_l1, d_l1_ = np.zeros([LEN, ]),np.zeros([LEN, ]) # 初始化Di和D-i数组
    for i in range(len(V_l1)):
        d_l1[i] = V_l1[i]/(V_l1[i]+V_l1_[i])
        d_l1_[i] = V_l1_[i]/(V_l1[i]+V_l1_[i])
    return d_l1, d_l1_


def revise_reward(reward, D, if_collision, alpha):
    '''
    修正reward，得到r_individual 和 r_global引入公平性，加权得到r_total
    '''
    reward_ind, reward_other = np.zeros([len(reward, )]), np.zeros([len(reward, )])
    reward_global_temp = 0

    # 计算r_individual  r_global
    for i in range(len(reward)):
        reward_other[i] = sum(reward) - reward[i]
        if reward[i] == 1 and D[i] == np.max(D):  # 传输成功且是最应该传的
            reward_global_temp = 1
            reward_ind[i] = 1

        if reward[i] == 1 and D[i] != np.max(D):  # 传输成功但不是最该传的
            reward_global_temp = D[i]
            reward_ind[i] = -1

        if reward[i] == 0 and D[i] == np.max(D):  # 不传输，但是最该传的
            reward_ind[i] = -1/(1-D[i])
        if reward[i] == 0 and D[i] != np.max(D):  # 不传输也确实是最应该传的
            reward_ind[i] = 1

        if if_collision[i] == 1 and D[i] == np.max(D):  # 如果对应agent没有成功传输，且他是导致碰撞的一员，给予负奖励
            reward_global_temp = -1
            reward_ind[i] = 1
        if if_collision[i] == 1 and D[i] != np.max(D):  # 又是碰撞的人，又是不该传的人
            reward_global_temp = -1
            reward_ind[i] = -1

    reward_global = np.array([reward_global_temp] * len(reward))

    # 加权得到r_total
    r_total = alpha * reward_global + (1 - alpha) * reward_ind

    return reward_ind, reward_global, r_total, reward_other


num_episodes = 1
episode_length = 500000  # 每条序列的最大长度/此时要理解为时隙数
buffer_size = 100000
# buffer_size = 4000
hidden_dim_a = 64  # actor网络隐藏层维数
hidden_dim_c = 128  # critic网络隐藏层维数
actor_lr = 5e-4  # 学习率
critic_lr = 5e-4
gamma = 0.95
tau = 1e-2   # 软更新参数，通常比较小
batch_size = 64
# device = torch.de
# vice("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
update_interval = 200
learning_interval = 100
minimal_size = 4000  # 最小开始学习时经验库容量
# minimal_size = 500  # 最小开始学习时经验库容量
state_length_M = 5
alpha = 0.3
env = env_SL(state_length=state_length_M)  # 创建环境
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

# state_dims 包含4项，每一项对应一个agent的state维数，[x1,x2,x3,x4]
# state_dims = []  # 初始化状态维度
state_dims = [6*state_length_M,6*state_length_M,6*state_length_M] #3个节点版本,6*state_length_M]

action_dims = [2,2,2] #3个节点版本,2]
critic_input_dim = sum(state_dims) + sum(action_dims) # ∑ =

# 实例化
maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim_a,hidden_dim_c, state_dims,
                action_dims, critic_input_dim, gamma, tau)

return_list = []  # 记录每一轮的回报（return）
total_step = 0
for i_episode in range(num_episodes):
    state = env.reset()  # 6个0 * M * 3个agent
    # 记录各个agent在各条link上的传输情况
    reward1_l1_list, reward2_l1_list, reward3_l1_list = [], [], []
    V_l1 = np.zeros([env.agent_number, ])  # 记录link上的D2LT
    V_l1_ = np.zeros([env.agent_number, ])  # 记录link上的V(-i)
    d_l1 = np.zeros([env.agent_number, ])   # 小d是归一化vi和v-i
    d_l1_ = np.zeros([env.agent_number, ])  # 记录link上的D(-i)
    D_l1 = np.zeros([env.agent_number, ])   # 大D是对所有Vi的归一化
    next_state = state.copy()
    # ep_returns = np.zeros(len(env.agents))
    for e_i in tqdm(range(episode_length)):

        actions = maddpg.take_action(state, explore=True)  # 得到3个agent的动作合集

        ############# 以下三个操作应该打包，相当于都是与环境交互得到的信息  ########
        link_obs, reward, l_collision = env.step(actions)
        # 归一化D2LT
        if e_i != 0:
            D_l1 = V_l1 / np.sum(V_l1)
        # 修正reward
        r_ind, r_global, r_total, r_other = revise_reward(reward, D_l1, l_collision, alpha)
        ##################################################################

        # 更新D2LT
        V_l1, V_l1_ = update_D2LT(reward, env.agent_number, V_l1, V_l1_)

        # 归一化Vi V-i得到d
        d_l1, d_l1_ = normalize_D2LT(V_l1, V_l1_)
        for i in range(env.agent_number):
            next_state[i] = np.concatenate([state[i][6:],link_obs[i],[d_l1[i], d_l1_[i]]])
            # next_state[i] = np.concatenate([state[i][6:],actions[i],link_obs[i],[d_l1[i], d_l1_[i]]])

        # 记录reward
        reward1_l1_list.append(reward[0])
        reward2_l1_list.append(reward[1])
        reward3_l1_list.append(reward[2])

        replay_buffer.add(state, actions, r_total, next_state)  # 存入的action是explore=ture的action，即有梯度信息;更新还是使用r_total

        # 到下一状态
        state = next_state.copy()
        total_step += 1
        if replay_buffer.size(
        ) >= minimal_size and total_step % learning_interval == 0:   # 每隔20步学习一次
            sample = replay_buffer.sample(batch_size)

            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x]
                              for i in range(len(x[0]))]
                return [
                    torch.FloatTensor(np.vstack(aa)).to(device)
                    for aa in rearranged
                ]

            sample = [stack_array(x) for x in sample]
            for a_i in range(env.agent_number):
                maddpg.update(sample, a_i)

            if total_step % update_interval == 0:  # 延迟更新目标网络
                maddpg.update_all_targets()

    with open(f'rewards/reward1_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}_c_lr{critic_lr}'
              f'_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_v1.txt', 'w') as reward1_l1_txt:
        for i in reward1_l1_list:
            reward1_l1_txt.write(str(i) + '   ')
    with open(f'rewards/reward2_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}_c_lr{critic_lr}'
              f'_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_v1.txt', 'w') as reward2_l1_txt:
        for i in reward2_l1_list:
            reward2_l1_txt.write(str(i) + '   ')
    with open(f'rewards/reward3_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}_c_lr{critic_lr}'
              f'_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_v1.txt', 'w') as reward3_l1_txt:
        for i in reward3_l1_list:
            reward3_l1_txt.write(str(i) + '   ')


MODEL_PATH = ['model/3node_SL_1.pt', 'model/3node_SL_2.pt', 'model/3node_SL_3.pt']
for i in range(env.agent_number):
    torch.save(maddpg.agents[i].actor.state_dict(), MODEL_PATH[i])

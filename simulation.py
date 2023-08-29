
import numpy as np
from env import env_SL
from agent import Agent
import torch 


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


agent_params = {
    "state_dim": 6*5, 
    "action_dim": 2, 
    "critic_input_dim": (6*5+2)*3, 
    "hidden_dim_a": 64, 
    "hidden_dim_c": 128,
    "actor_lr": 5e-4, 
    "critic_lr": 5e-4, 
    "device": "cpu", 
    "gamma": 0.95, 
    "tau": 1e-2
}
MODEL_PATH = ['model/3node_SL_1.pt', 'model/3node_SL_2.pt', 'model/3node_SL_3.pt']

class Simulation:
    def __init__(self, agent_params, agent_number = 3, state_length_M = 5, n_episode = 1, episode_length = 40000) -> None:
        self.env = env_SL(state_length=state_length_M)
        self.agent_number = agent_number
        self.n_episode = 1
        self.episode_length = 40000
        agent_params["critic_input_dim"] = agent_number * ( agent_params["action_dim"] + agent_params["state_dim"] )
        self.agents = []
        for i in range(agent_number):
            agent = Agent(**agent_params)
            agent.actor.load_state_dict(torch.load(MODEL_PATH[i]))
            self.agents.append(agent) # 加载agents
        reward1_l1 = []
        reward2_l1 = []
        reward3_l1 = []
        # reward5_l1, reward5_l2 = [], []
        V_l1, V_l1_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])  # 记录link上的D2LT
        d_l1, d_l1_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])   # 小d是归一化vi和v-i
        D_l1 = np.zeros([env.agent_number, ])   
    def run(self):
        
        pass
    
    def summary(self):
        pass 
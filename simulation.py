import numpy as np
from env import env_SL
from agent import Agent
import torch 
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt

def get_id(reward):
    if np.sum(reward) == 0:
        return 0
    else:
        return np.where(reward == 1)


agent_params = {
    "state_dim": 6*5, 
    "action_dim": 2, 
    "critic_input_dim": (6*5+2)*3, 
    "hidden_dim_a": 64, 
    "hidden_dim_c": 128,
    "actor_lr": 5e-4, 
    "critic_lr": 5e-4, 
    "device": "cpu", 
    # "gamma": 0.95, 
    # "tau": 1e-2
}
MODEL_PATH = ['model/3node_SL_1.pt', 'model/3node_SL_2.pt', 'model/3node_SL_3.pt']

class Simulation:
    def __init__(self, agent_params, agent_number = 3, state_length_M = 5, n_episode = 1, episode_length = 40000) -> None:
        self.env = env_SL(state_length=state_length_M)
        self.agent_number = agent_number
        self.n_episode = n_episode
        self.episode_length = episode_length
        agent_params["critic_input_dim"] = agent_number * ( agent_params["action_dim"] + agent_params["state_dim"] )
        self.agents: List[Agent] = []
        for i in range(agent_number):
            agent = Agent(**agent_params)
            agent.actor.load_state_dict(torch.load(MODEL_PATH[i]))
            self.agents.append(agent) # 加载agents
 
    def run(self): # < 15:30
        for _ in range(self.n_episode):  # 开始迭代
            states = self.env.reset()  # 每次迭代前重置环境
            for ag in agents:
                ag.reset()
            next_states = states.copy()
            # agent 重新初始化D2LT
            for t_i in tqdm(range(self.episode_length)):
                actions = []
                for ag, obs in zip(self.agents, states):
                    action = ag.take_action(obs, explore=False) # 返回一个列表，里面有三个元素，分别表示三个agent的action数组
                    actions.append(action)
                link1_obs, reward_l1, l1_collision = self.env.step(actions)
                
                for i, ag in enumerate(self.agents):
                    ag.updateD2LT(reward_l1)
                    d_l1, d_l1_ = ag.d2lt_norm()
                    next_states[i] = np.concatenate([states[i][6:], link1_obs[i], [d_l1[i], d_l1_[i]]])
                states = next_states.copy()
    
    def summary(self):
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
    
    

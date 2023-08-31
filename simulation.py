import numpy as np
from env import env_SL
from agent import Agent
import torch 
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt

def get_id(reward: np.ndarray):
    if np.sum(reward) == 0:
        return 0
    else:
        return np.where(reward == 1)[0] + 1 # reward列表中只有一个为1

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


MODEL_PATH = ['model/3node_SL_1.pt', 'model/3node_SL_2.pt', 'model/3node_SL_3.pt']

class Simulation:
    def __init__(self, agent_params, agent_number = 3, state_length_M = 5, n_episode = 1, episode_length = 40000) -> None:
        self.env = env_SL()
        self.agent_number = agent_number
        self.n_episode = n_episode
        self.episode_length = episode_length
        self.agents: List[Agent] = []
        for i in range(agent_number):
            agent_params['id'] = i+1
            agent = Agent(**agent_params)
            agent.alg.load_state_dict(torch.load(MODEL_PATH[i]))
            self.agents.append(agent) # 加载agents
 
    def run(self): 
        for episode_i in range(self.n_episode):  # 开始迭代
            # 每次迭代前重置agents, 重置环境
            self.env.reset()
            for ag in self.agents:
                ag.reset()
            # agent 重新初始化D2LT
            for t_i in tqdm(range(self.episode_length)):
                actions = {} # 各个agents行为的汇总
                for ag in self.agents:
                    action = ag.take_action(explore=False) # 返回一个列表，里面有三个元素，分别表示三个agent的action数组
                    actions[ag.id] = action
                    # print(ag.states_mem, ag.last_action)
                link_ret = self.env.step(actions) # 单链路返回值
                
                # 根据链路返回值来更新各个agent的状态列表
                for ag in self.agents:
                    ag.update_states_mem(link_ret)
            self.summary(episode_i)
            
    def summary(self, k):
        l1_throughputs = []
        for ag in self.agents:
            l1_throughputs.append(return_throughput(ag.reward_log))   #agent[i] 吞吐量
        l1_sum_throughput = [l1_throughputs[0][i] + l1_throughputs[1][i] + l1_throughputs[2][i]  for i in
                                    range(self.episode_length)]  # l1总吞吐量
        
        mean1 = np.mean(round(l1_sum_throughput[-1000], 2))  # 最后逼近的值

        print(k, mean1)
        fig = plt.figure(num=k,figsize=(14, 6))
        # plt.subplot(3, 1, 1)
        plt.plot(l1_sum_throughput, c='r', label='Sum')
        plt.plot(l1_throughputs[0], c='b', label='agent1')
        plt.plot(l1_throughputs[1], c='cyan', label='agent2')
        plt.plot(l1_throughputs[2], c='orange', label='agent3')
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
        # plt.show()
        plt.savefig(f"fig/episode{k}.png")
    

if __name__ == "__main__":
    agent_params = { # actor 网络参数
    "state_dim": 6*5, 
    "action_dim": 2, 
    # "critic_input_dim": (6*5+2)*3, 
    "hidden_dim_a": 64, 
    # "hidden_dim_c": 128,
    # "actor_lr": 5e-4, 
    # "critic_lr": 5e-4, 
    "device": "cpu", 
    "memory_len": 5
    # "gamma": 0.95, 
    # "tau": 1e-2
    }
    sim = Simulation(agent_params=agent_params, agent_number=3, n_episode=3)
    sim.run()
    

from algorithm import DDPG
import torch 

class Agent:
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim_a,hidden_dim_c,
                 actor_lr, critic_lr, device, gamma, tau) -> None:
        self.alg = DDPG(state_dim, action_dim, critic_input_dim, hidden_dim_a,hidden_dim_c,
                 actor_lr, critic_lr, device)
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        
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
    
    def learn(self, env):
        pass
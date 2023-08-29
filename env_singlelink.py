import numpy as np


class env_SL(object,):
    def __init__(self,state_length = 10):
        self.agent_number = 3
        self.action_space = 2
        self.state_space = 6
        self.M = state_length

    def reset(self):  # 初始化,返回link_obs
        init_state = [np.zeros([self.state_space * self.M, ])] * self.agent_number  # agent_number个基本state
        return init_state
    
    def D2O(self, action):
        '''
        2位one-hot action转为1位action：
        [10]=>0  [01]=>1
        '''
        action_o = 0
        if action[0] == 1:
            action_o = 0
        elif action[1] == 1:
            action_o = 1

        return action_o

    def step(self, actions):
        '''
        传入actions，返回link_obs, reward, done, _
        actions是一个list，包含各个agent的action数组
        link_obs：自己成功传 1000；别人成功传0100； 碰撞0010；空闲0001
        '''
        # 初始化
        action1 = actions[0]
        action2 = actions[1]
        action3 = actions[2]
        # reward记录在link上的传输情况
        reward = np.zeros([self.agent_number, ])

        link_obs = [np.zeros([4, ])] * self.agent_number
        action1_o, action2_o, action3_o = self.D2O(action1), self.D2O(action2), self.D2O(action3)

        total_action = action1_o + action2_o + action3_o
        l_collision = np.zeros([self.agent_number, ])  # 记录碰撞
        # print(total_action)
        if total_action == 0:  # link空闲
            # link_obs
            link_obs = [[0,0,0,1]] * self.agent_number
        elif total_action == 1:  # link成功传输
            # link_obs1 = '1S'
            if action1_o == 1:
                reward[0] = 1
                link_obs[0] = [1, 0, 0, 0]
                link_obs[1], link_obs[2] = [0, 1, 0, 0],[0, 1, 0, 0]
            elif action2_o == 1:
                reward[1] = 1
                link_obs[1] = [1, 0, 0, 0]
                link_obs[0], link_obs[2] = [0, 1, 0, 0], [0, 1, 0, 0]
            elif action3_o == 1:
                reward[2] = 1
                link_obs[2] = [1, 0, 0, 0]
                link_obs[0], link_obs[1] = [0, 1, 0, 0], [0, 1, 0, 0]

        else:  # link1碰撞
            # link_obs1 = '1B'
            link_obs = [[0,0,1,0]] * self.agent_number
            if action1_o == 1:
                l_collision[0] = 1
            if action2_o == 1:
                l_collision[1] = 1
            if action3_o == 1:
                l_collision[2] = 1


        # self.update()
        return link_obs, reward, l_collision  #, done

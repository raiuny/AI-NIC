import numpy as np


class env_SL(object):
    def __init__(self):
        self.action_space = 2
        self.link_ret = {'occupy_id': 0} # 链路处于空闲状态

    def reset(self):
        self.link_ret =  {'occupy_id': 0}
        
    def D2O(self, action: np.ndarray):
        '''
        2位one-hot action转为1位action：
        [10]=>0  [01]=>1
        '''
        assert np.sum(action) == 1, "action one hot code"
        return np.where(action==1)[0]

    def step(self, actions: dict):
        '''
        actions：字典，key: id，value: action
        link_ret：字典，{"occupy_id": 1}
        '''
        # 初始化
        send_list = []
        for id, action in actions.items():
            if self.D2O(action) == 1:
                send_list.append(id) # 假设只要发送了就一定能被接收到并解析出id, 如果不能解析出id，则有可能发生了碰撞或者空闲，
                # 碰撞和空闲状态，返回均为link_ret的初始值     
        if len(send_list) == 1: # 传输成功，返回传输的id
            self.link_ret['occupy_id'] = send_list[0]
            return  self.link_ret
        elif len(send_list) > 1: # 发生碰撞
            self.link_ret['occupy_id'] = -1
            return self.link_ret
        else: # 空闲
            self.link_ret['occupy_id'] = 0
            return self.link_ret

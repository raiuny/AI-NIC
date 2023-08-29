from model import FCwithGRU, TwoLayerFC
import torch 
from rl_utils import gumbel_softmax, onehot_from_logits
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

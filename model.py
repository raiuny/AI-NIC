import torch
import torch.nn.functional as F

class FCwithGRU(torch.nn.Module):
    '''
    构建一个4层带有GRU结构的网络, actor
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
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Neural network for Q-values approximation (used in Deep Q-Network (DQN) algorithms)

    Attributes:
        seed (int): Random seed
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Second fully connected layer
        fc3 (nn.Linear): Third fully connected layer

    Methods:
        forward(state): Forward pass of model for mapping state to actions.

    """

    def __init__(self, state_dim, action_dim, fc1_dim=128, fc2_dim=64, seed=42):
        """
        Initialize parameters and build model.

        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            fc1_dim (int): Dimension of first fully connected layer
            fc2_dim (int): Dimension of second fully connected layer
            seed (int): Random seed

        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)

    def forward(self, state):
        """
        Forward pass of model for mapping state to actions.

        Args:
            state (torch.Tensor): Tensor of shape (batch_size, state_dim)

        Returns:
            x (torch.Tensor): Tensor of shape (batch_size, action_dim)

        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DuelingQNetwork(nn.Module):
    """
    Neural network for Dueling DQN algorithms.
    It has two streams of computing the Q-values:
        1. Value stream: Computes the value of the state.
        2. Advantage stream: Computes the advantage of taking the action given the state.
    The formula is as follow: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)). We substract the mean of the advantage to:
        1. Promote training stability,
        2. Reduce ambiguity: Ensure unique combination of V(s) and A(s,a) therefore Q-values are more identifiable.
        3. Promotes generalization.

    Attributes:
        seed (int): Random seed for reproducibility
        fc1 (nn.Linear): First fully connected layer for the common feature extraction.
        fc2 (nn.Linear): Second fully connected layer for the common feature extraction.
        value_stream (nn.Linear): First fully connected layer for value stream.
        advantage_stream (nn.Linear): Second fully connected layer for advantage stream.

    Methods:
        forward(state): Forward pass of model for mapping state to actions.

    """

    def __init__(self, state_dim, action_dim, fc1_dim=128, fc2_dim=64, seed=42):
        """
        Initialize parameters and build model.

        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            fc1_dim (int): Dimension of first fully connected layer
            fc2_dim (int): Dimension of second fully connected layer
            seed (int): Random seed

        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        # Value stream
        self.value_stream = nn.Linear(fc2_dim, 1)

        # Advantage stream
        self.advantage_stream = nn.Linear(fc2_dim, action_dim)

    def forward(self, state):
        """
        Forward pass of model for mapping state to actions.

        Args:
            state (torch.Tensor): Tensor of shape (batch_size, state_dim)

        Returns:
            q_values (torch.Tensor): Tensor of shape (batch_size, action_dim)

        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

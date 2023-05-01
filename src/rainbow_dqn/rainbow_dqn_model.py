import math
import torch
import torch.nn as nn


class RainbowDQNModel(nn.Module):
    """
    Rainbow Deep Q-Network Model. Approximates a state-value function in a Q-Learning
    framework with a neural network.

    Rainbow DQN is an extension of DQN.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_1_size: int = 128,
        hidden_2_size: int = 128,
        std_init: float = 0.5,
    ):
        super(RainbowDQNModel, self).__init__()
        # Feature layers.
        self.fc1 = nn.Linear(state_size, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)

        # Noisy value layer.
        self.value_hidden_stream = NoisyLinear(
            in_features=hidden_2_size // 2,
            out_features=hidden_2_size // 2,
            std_init=std_init,
        )
        self.value_stream = NoisyLinear(
            in_features=hidden_2_size // 2, out_features=1, std_init=std_init
        )

        # Noisy advantage layer.
        self.advantage_hidden_stream = NoisyLinear(
            in_features=hidden_2_size // 2,
            out_features=hidden_2_size // 2,
            std_init=std_init,
        )
        self.advantage_stream = NoisyLinear(
            in_features=hidden_2_size // 2, out_features=action_size, std_init=std_init
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Called with either one element to determine next action, or a batch
        during optimization.
        Args:
            state: The current state of the game.
        Returns:
            A tensor representing the chosen action.
        """
        action = torch.relu(self.fc1(state))
        action = torch.relu(self.fc2(action))

        # value = self.value_hidden_stream(action[:, : action.size(1) // 2])
        value = self.value_stream(value)
        # advantage = self.advantage_hidden_stream(x[:, x.size(1) // 2 :])
        advantage = self.advantage_stream(advantage)

        # Combine value and advantage streams to get Q-values.
        q_values = value.expand_as(advantage) + (
            advantage - advantage.mean(1, keepdim=True).expand_as(advantage)
        )

        return q_values

        action = torch.relu(self.fc1(state))
        action = torch.relu(self.fc2(action))

        value_hidden = torch.relu(self.value_hidden_stream(action))
        value = self.value_stream(value_hidden)

        advantage_hidden = torch.relu(self.advantage_hidden_stream(action))
        advantage = self.advantage_stream(advantage_hidden)

        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        """
        Reset all noisy layers.
        """
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()


# Noisy net
class NoisyLinear(nn.Module):
    """
    Noisy linear module for NoisyNet.

    Args:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return nn.Linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

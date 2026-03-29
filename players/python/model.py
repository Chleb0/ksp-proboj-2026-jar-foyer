from typing import List
from typing import TypedDict
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from data import ShadeID
import os

class PPOMemory(TypedDict):
    board: dict[ShadeID, List[Tensor]]
    extra: Dict[ShadeID, List[Tensor]]
    actions: Dict[ShadeID, List[int]]
    log_probs: Dict[ShadeID, List[float]]
    rewards: Dict[ShadeID, List[float]]
    dones: Dict[ShadeID, List[bool]]

class PPOActorCritic(nn.Module):
    def __init__(self, input_channels: int, extra_input_dim: int, action_dim: int) -> None:
        super().__init__()

        # CNN branch
        self.cnn: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_channels, 5, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute CNN output size dynamically
        with torch.no_grad():
            dummy: Tensor = torch.zeros(1, input_channels, 84, 84)
            cnn_out_dim: int = self.cnn(dummy).shape[1]

        # Extra input branch
        self.extra_fc: nn.Sequential = nn.Sequential(
            nn.Linear(extra_input_dim, 64),
            nn.ReLU()
        )

        # Combined
        self.combined_fc: nn.Sequential = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 256),
            nn.ReLU()
        )

        # Actor & Critic
        self.actor: nn.Linear = nn.Linear(256, action_dim)
        self.critic: nn.Linear = nn.Linear(256, 1)

    def forward(self, img: Tensor, extra: Tensor) -> Tuple[Tensor, Tensor]:
        cnn_features: Tensor = self.cnn(img)
        extra_features: Tensor = self.extra_fc(extra)

        combined: Tensor = torch.cat([cnn_features, extra_features], dim=1)
        hidden: Tensor = self.combined_fc(combined)

        logits: Tensor = self.actor(hidden)
        value: Tensor = self.critic(hidden)

        return logits, value

    def get_action(self, img: Tensor, extra: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logits, value = self.forward(img, extra)
        probs: Categorical = Categorical(logits=logits)
        action: Tensor = probs.sample()
        log_prob: Tensor = probs.log_prob(action)

        return action, log_prob, value

    def evaluate(
        self,
        img: Tensor,
        extra: Tensor,
        action: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logits, value = self.forward(img, extra)
        probs: Categorical = Categorical(logits=logits)

        log_probs: Tensor = probs.log_prob(action)
        entropy: Tensor = probs.entropy()

        return log_probs, entropy, value


from typing import Dict, List


class PPO:
    def __init__(
        self,
        model: PPOActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4
    ) -> None:
        self.model: PPOActorCritic = model
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.gamma: float = gamma
        self.eps_clip: float = eps_clip
        self.k_epochs: int = k_epochs

    def compute_returns(
        self,
        rewards: List[float],
        dones: List[bool],
        values: Tensor
    ) -> Tensor:
        returns: List[float] = []
        G: float = 0.0

        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32)

    def update(self, memory: PPOMemory, ghost: ShadeID) -> None:
        boards: Tensor = torch.stack(memory['board'][ghost])
        extras: Tensor = torch.stack(memory['extra'][ghost])
        actions: Tensor = torch.tensor(memory['actions'][ghost])
        old_log_probs: Tensor = torch.tensor(memory['log_probs'][ghost])
        rewards: List[float] = memory['rewards'][ghost]
        dones: List[bool] = memory['dones'][ghost]

        with torch.no_grad():
            _, values = self.model(boards, extras)

        returns: Tensor = self.compute_returns(rewards, dones, values)
        advantages: Tensor = returns - values.squeeze()

        for _ in range(self.k_epochs):
            log_probs, entropy, values = self.model.evaluate(boards, extras, actions)

            ratios: Tensor = torch.exp(log_probs - old_log_probs)

            surr1: Tensor = ratios * advantages
            surr2: Tensor = torch.clamp(
                ratios,
                1 - self.eps_clip,
                1 + self.eps_clip
            ) * advantages

            actor_loss: Tensor = -torch.min(surr1, surr2).mean()
            critic_loss: Tensor = F.mse_loss(values.squeeze(), returns)

            loss: Tensor = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def save_checkpoint(
    ppo: PPO,
    file_path: str
) -> None:
    temp_path = file_path

    checkpoint = {
        "model_state_dict": ppo.model.state_dict(),
        "optimizer_state_dict": ppo.optimizer.state_dict()
    }

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, file_path)

def load_checkpoint(
    ppo: PPO,
    file_path: str
) -> None:
    checkpoint = torch.load(file_path)

    ppo.model.load_state_dict(checkpoint["model_state_dict"])
    ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

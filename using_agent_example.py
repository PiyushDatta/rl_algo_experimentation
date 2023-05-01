import logging

import torch

import hydra
from omegaconf import DictConfig

from src.agent import Agent
from src.util_cls import AgentConfig
from src.util_fn import generate_agent, generate_env


@hydra.main(version_base="1.2", config_path="configs", config_name="current")
def use_agent(cfg: DictConfig):
    # Setup logging.
    logging.getLogger().setLevel(level=logging.getLevelName(str(cfg.env.logging_level)))

    # Game: Gym Environment.
    env = generate_env(str(cfg.env.env_name))

    # Number of state observations.
    state_size = env.observation_space.shape[0]

    # Number of actions from gym action space.
    action_size = env.action_space.n

    # Whether we run our model on a cpu or gpu (cuda only).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup AgentConfig.
    agent_config = AgentConfig(
        hidden_1_size=cfg.neural_net.hidden_nodes_1,
        hidden_2_size=cfg.neural_net.hidden_nodes_2,
        max_grad_norm=cfg.agent.max_grad_norm,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        lr=cfg.agent.lr,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_min=cfg.agent.epsilon_min,
        epsilon_decay=cfg.agent.epsilon_decay,
        replay_mem_size=cfg.agent.replay_mem_size,
        num_update_target=cfg.env.num_update_target,
        num_save_weights=cfg.env.num_save_weights,
        batch_size=cfg.env.batch_size,
    )

    # Setup Agent.
    agent: Agent = generate_agent(
        weights_file=cfg.agent.weights_file,
        device=device,
        agnt=cfg.agent.agent_type,
        config=agent_config,
        state_size=state_size,
        action_size=action_size,
    )

    # Test the agent on one environment move.
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action: int = agent.take_action(env, state).item()

    # Push cart to the left or push cart to the right.
    assert action == 0 or action == 1
    print("Done, using an agent works!")


if __name__ == "__main__":
    use_agent()

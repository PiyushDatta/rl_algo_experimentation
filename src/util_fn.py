import gymnasium as gym
from torch import device
from src.agent import Agent
from src.dqn.dqn_agent import DQNAgent
from src.dueling_dqn.dueling_dqn_agent import DuelingDQNAgent
from src.rainbow_dqn.rainbow_dqn_agent import RainbowDQNAgent

from src.util_cls import AgentConfig, AgentsEnum


def generate_env(env_name: str) -> gym.Env:
    """
    Generates the specified environment.
    """
    if env_name == "CartPole-v1":
        return gym.make("CartPole-v1")
    else:
        raise ValueError("Unsupported environment: {env}".format(env=env_name))


def generate_agent(
    weights_file: str,
    device: device,
    agnt: str,
    config: AgentConfig,
    state_size: int,
    action_size: int,
) -> Agent:
    """
    Generates the specified Agent.
    """
    try:
        agent_type = AgentsEnum[agnt.upper()]
    except KeyError:
        raise ValueError("Unsupported Agent: {agent}.".format(agent=agent_type))

    if agent_type == AgentsEnum.DQN_AGENT:
        return DQNAgent(
            device=device,
            state_size=state_size,
            action_size=action_size,
            weights_file=weights_file,
            config=config,
        )
    elif agent_type == AgentsEnum.DUELING_DQN_AGENT:
        return DuelingDQNAgent(
            device=device,
            state_size=state_size,
            action_size=action_size,
            weights_file=weights_file,
            config=config,
        )
    elif agent_type == AgentsEnum.RAINBOW_DQN_AGENT:
        return RainbowDQNAgent(
            device=device,
            state_size=state_size,
            action_size=action_size,
            weights_file=weights_file,
            config=config,
        )
    raise ValueError(f"No agent of type {agent_type} found")

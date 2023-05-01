import logging

import torch

import hydra
from omegaconf import DictConfig

from src.agent import Agent
from src.playground import Playground
from src.util_cls import AgentConfig
from src.util_fn import generate_agent, generate_env


def get_config_str(cfg: dict) -> str:
    """
    Get the configurations/hyperparameters
    into a nice string format.
    """
    final_str = ""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            final_str += "{k}: {v}\n".format(k=k, v=v)

    return final_str


def log_config_info(
    playground: Playground,
    agent_config: AgentConfig,
    log_name: str = "training",
    num_eps: int = 0,
) -> None:
    """
    Before training or validating, pretty print the
    configurations/hyperparameters.
    """
    logging.info(
        "Starting {log_nme} for {eps} episodes.\n\n"
        "Running the playground with:\n{pl_config}\n\n"
        "With below config:\n{config}\n\n".format(
            log_nme=log_name,
            eps=num_eps,
            pl_config=get_config_str(vars(playground)),
            config=get_config_str(vars(agent_config)),
        )
    )


@hydra.main(version_base="1.2", config_path="../configs", config_name="current")
def main(cfg: DictConfig):
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

    # Setup a Playground with the Agent.
    playground: Playground = Playground(
        gym_env=env,
        weights_file=cfg.agent.weights_file,
        device=device,
        agent=agent,
        config=agent_config,
    )

    # Train.
    if cfg.env.train:
        # Clear metrics.
        playground.clear_metrics()
        # Pretty print config.
        log_config_info(
            playground=playground,
            agent_config=agent_config,
            log_name="training",
            num_eps=cfg.env.training_episodes,
        )
        try:
            playground.train(
                episodes=cfg.env.training_episodes,
                should_plot=cfg.env.plot_training,
            )
            logging.info("Finished training!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping training.\n")

    # Validate/Test.
    if cfg.env.validate:
        # Clear metrics.
        playground.clear_metrics()
        # Pretty print config.
        log_config_info(
            playground=playground,
            agent_config=agent_config,
            log_name="validation",
            num_eps=cfg.env.validating_episodes,
        )
        try:
            playground.validate(
                episodes=cfg.env.validating_episodes,
                should_plot=cfg.env.plot_training,
            )
            logging.info("Finished validating!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping validation.\n")

    # Close our env, since we're done training/validating.
    env.close()


if __name__ == "__main__":
    main()

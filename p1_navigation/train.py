from env import BananaUnityEnv
from unityagents import UnityEnvironment
import numpy as np
from random import randint
from agent import DQNAgent, DoubleDQNAgent
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List

import sys

if len(sys.argv) != 3:
    print("Usage: python train_script.py [dueling: True/False] [double: True/False]")
    sys.exit(1)

dueling_arg = sys.argv[1]
double_arg = sys.argv[2]

if dueling_arg == "True" and double_arg == "True":
    print("Dueling and Double Enabled")
    dueling_value = True
    double_value = True
elif dueling_arg == "True" and double_arg == "False":
    print("Dueling Enabled")
    dueling_value = True
    double_value = False
elif dueling_arg == "False" and double_arg == "True":
    print("Double Enabled")
    dueling_value = False
    double_value = True

elif dueling_arg == "False" and double_arg == "False":
    print("Dueling and Double Disabled")
    dueling_value = False
    double_value = False

else:
    print("Invalid Arguments")
    sys.exit(1)


def save_agent(agent: DQNAgent, file_name: str = "checkpoint.pth", episode: int = None):
    """
    Save agent's Q-network weights and optionally episode number to a file.

    Args:
        agent (DQNAgent): The agent whose Q-network weights need to be saved.
        file_name (str): The path to the file where the data should be saved.
        episode (int, optional): The episode number at which the model is saved. Defaults to None.
    """
    checkpoint = {
        "q_network_state_dict": agent.q_network.state_dict(),
        "episode": episode,
    }
    torch.save(checkpoint, file_name)


def load_agent(agent: DQNAgent, file_name: str = "checkpoint.pth") -> int:
    """
    Load agent's Q-network weights and optionally episode number from a file.

    Args:
        agent (DQNAgent): The agent to which the Q-network weights should be loaded.
        file_name (str): The path to the file from which the data should be loaded.

    Returns:
        int or None: The episode number at which the model was saved or None if not saved.
    """
    checkpoint = torch.load(file_name)
    agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
    agent.q_network.eval()  # Set the network to evaluation mode
    return checkpoint.get("episode", None)


def plot(scores: List[float], mv_score: List[float], save_path: str, ep=None) -> None:
    """
    Plot scores given a given frequency.

    Args:
        scores (List[float]): List of accumulated scores.
        mv_score (List[float]): Moving average of the last 100 scores.
        plot_every (int): Number of episodes before updating the plot.

    """
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(mv_score)), mv_score)
    plt.ylabel("Total Reward")
    plt.xlabel("Episode #")
    if ep is None:
        plt.title(f"Training Progress")
    else:
        plt.title(f"Training Progress solved in {ep} episodes")
    plt.grid(True)
    # save figure
    plt.savefig(save_path)
    plt.close()


def train_agent(
    num_episodes,
    fc1_dim,
    fc2_dim,
    seed,
    dueling,
    double,
    chkpt_path,
    model_path,
    figure_path,
):
    with BananaUnityEnv(file_name="Banana.app", seed=seed, no_graphics=False) as env:
        if double:
            agent = DoubleDQNAgent(
                state_size=env.state_size,
                action_size=env.action_size,
                fc1_dim=fc1_dim,
                fc2_dim=fc2_dim,
                seed=seed,
                dueling=dueling,
            )
        else:
            agent = DQNAgent(
                state_size=env.state_size,
                action_size=env.action_size,
                fc1_dim=fc1_dim,
                fc2_dim=fc2_dim,
                seed=seed,
                dueling=dueling,
            )

        # Agent Training Parameters
        max_time_step = 100000
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        epsilon = epsilon_start

        scores = []
        scores_window = deque(maxlen=100)

        max_score = 13
        mv_score = []
        # Training
        for current_episode in range(1, num_episodes + 1):
            state = env.reset()
            score = 0

            for t in range(max_time_step):
                action = agent.act(state, epsilon)
                next_state, reward, done = env.step(action)

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward

                if done:
                    break

            scores.append(score)  # save all scores
            scores_window.append(score)  # save the last score 100 scores
            mv_score.append(
                np.mean(scores_window)
            )  # compute the moving average of the last 100

            epsilon = max(epsilon_end, epsilon_decay * epsilon)  # decrease epsilon

            # save model every X episodes
            if current_episode % 100 == 0:
                plot(scores, mv_score, figure_path)
                print(
                    f"Save the model in episode {current_episode} with mean score {np.mean(scores_window)}"
                )
                save_agent(agent, chkpt_path, current_episode)

            if np.mean(scores_window) >= max_score:
                print(
                    f"The Env was solved in {current_episode} episodes, with {np.mean(scores_window)} mean score"
                )
                save_agent(agent, model_path)

                break

        plot(scores, mv_score, figure_path, current_episode)
    del env
    import gc

    gc.collect()
    return current_episode


import time

if __name__ == "__main__":
    current_episode = train_agent(
        num_episodes=2000,
        fc1_dim=128,
        fc2_dim=64,
        seed=32,
        dueling=dueling_value,
        double=double_value,
        chkpt_path=f"./models/navigation_dqn_checkpoint_dueling_{dueling_value}_double_{double_value}.pth",
        model_path=f"./models/navigation_dqn_dueling_{dueling_value}_double_{double_value}.pth",
        figure_path=f"./figures/navigation_dqn_dueling_{dueling_value}_double_{double_value}.png",
    )
    print(
        f"Env solved in {current_episode} episodes for dueling={dueling_value} and double={double_value}"
    )

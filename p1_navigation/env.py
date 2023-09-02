from random import randint

import numpy as np
from unityagents import UnityEnvironment


class BananaUnityEnv:
    """
    This class is used to generate the Banana collection environment instance for the agent to interact
    with.
    """

    def __init__(self, file_name: str, seed: int, no_graphics: bool):
        """
        Initialize the BananaUnityEnv class.
        Args:
            file_name: The name of the file that contains the environment.
            seed: The seed for the environment.
            no_graphics: Whether to display the environment.
        """
        self.env = UnityEnvironment(
            file_name=file_name,
            seed=seed,
            # worker_id=randint(0, 10000),
            no_graphics=no_graphics,
        )
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]
        self.action_size = brain.vector_action_space_size
        self.state_size = brain.vector_observation_space_size
        self.num_agents = len(self.env.brain_names)

    def reset(self):
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        return state

    def step(self, action):
        brain_name = self.env.brain_names[0]
        env_info = self.env.step(np.int32([action]))[
            brain_name
        ]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        return next_state, reward, done

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

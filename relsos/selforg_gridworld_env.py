import numpy as np

import gymnasium as gym
from gymnasium import spaces

import pygame


class SelfOrgGridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_agents=1, num_targets=1):
        self.size = size
        self.window_size = 512
        self.num_agents = num_agents
        self.num_targets = num_targets

        # Observation space to include multiple agents and targets
        # Each agent and target location is encoded as an element of {0, ..., `size`}^2
        self.observation_space = spaces.Dict(
            {
                "agents": spaces.Box(
                    low=0, high=size - 1, shape=(num_agents, 2), dtype=int
                ),
                "targets": spaces.Box(
                    low=0, high=size - 1, shape=(num_targets, 2), dtype=int
                ),
            }
        )

        # We have 5 actions, corresponding to "up", "down", "right", "left", "do nothing"
        self.action_space = spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([0, 1]),  # up
            1: np.array([0, -1]),  # down
            2: np.array([1, 0]),  # right
            3: np.array([-1, 0]),  # left
            4: np.array([0, 0]),  # do nothing
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Human-rendering components
        self.window = None
        self.clock = None

        # Initialize environment state
        self.state = None
        self.reset()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize random positions for agents and targets
        positions = set()

        # Generate unique positions for each agent
        self._agent_locations = np.zeros((self.num_agents, 2), dtype=int)
        for i in range(self.num_agents):
            while True:
                pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
                if pos not in positions:
                    positions.add(pos)
                    self._agent_locations[i] = np.array(pos)
                    break

        # Generate unique positions for each target
        self._target_locations = np.zeros((self.num_targets, 2), dtype=int)
        for i in range(self.num_targets):
            while True:
                pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
                if pos not in positions:
                    positions.add(pos)
                    self._target_locations[i] = np.array(pos)
                    break

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Convert scalar actions to array actions if necessary => need it to be an array
        # TODO might want to remove this in the "final" version
        if np.isscalar(action):
            # Assuming the same action is to be applied to all agents
            action = [action] * self.num_agents

        rewards = np.zeros(self.num_agents, dtype=float)
        terminated = False

        agent_achieved_target = [False] * self.num_agents
        # Update the location of each agent based on their respective action
        for i in range(self.num_agents):
            direction = self._action_to_direction[action[i]]

            new_position = self._agent_locations[i] + direction
            if self._is_valid_position(new_position, i):
                self._agent_locations[i] = new_position

            # Check if the agent has reached any target
            for target in self._target_locations:
                if np.array_equal(self._agent_locations[i], target):
                    # TODO notice that this reward structure can easily give rise to selfish behaviour: for a single agent, the best thing to do is to somehow keep all the other agents out of their targets, and enter and exit its own target indefinitely
                    rewards[i] = 1  # Agent i has reached a target
                    agent_achieved_target[
                        i
                    ] = True  # Terminate the episode if any agent reaches a target
        terminated = all(agent_achieved_target)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # The reward could be the sum of individual agent rewards or handled differently
        total_reward = np.sum(rewards)

        return observation, total_reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _is_valid_position(self, new_position, current_agent_index):
        # Check if the new position is within the grid boundaries
        if not (0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size):
            return False

        # Check for collision with other agents
        for i, agent_position in enumerate(self._agent_locations):
            if i != current_agent_index and np.array_equal(
                new_position, agent_position
            ):
                return False

        return True

    def _get_obs(self):
        return {
            "agents": self._agent_locations,  # Array of all agent locations
            "targets": self._target_locations,  # Array of all target locations
        }

    # TODO this could probably be used for all sorts of interesting purposes
    def _get_info(self):
        min_distance = np.min(
            [
                np.linalg.norm(agent - target, ord=1)
                for agent in self._agent_locations
                for target in self._target_locations
            ]
        )

        return {"min_distance": min_distance}

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))  # Black background

        pix_square_size = (
            self.window_size / self.size
        )  # Size of a grid square in pixels

        # Draw the targets in light blue
        for target in self._target_locations:
            pygame.draw.rect(
                canvas,
                (173, 216, 230),  # Light blue color
                pygame.Rect(
                    pix_square_size * target,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw the agents in red
        for agent in self._agent_locations:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),  # Red color
                (agent + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines (maybe later)
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


# Simple way to include some tests
if __name__ == "__main__":
    # Create an instance of the environment
    env = SelfOrgGridWorld(size=10, num_agents=3, num_targets=5, render_mode="human")

    # Test 1: Reset the environment and check the initial state
    print("\n--- Testing Initial State ---")
    observation, info = env.reset()
    print("Initial Observation:", observation)
    print("Initial Info:", info)

    # Test 2: Perform random steps in the environment
    print("\n--- Sample Run ---")
    for i in range(40):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        observation, reward, done, _, info = env.step(actions)
        # print(
        #     f"Step {i+1}: Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}"
        # )
        env.render()
        # if done:
        #     print("All agents reached the targets. Resetting environment.")
        #     observation, info = env.reset()
    env.close()

    # Test 3: Check specific scenarios, like boundary conditions or agent collisions

    # Test for non-overlap of agents
    print("\n--- Testing Agent Non-Overlap ---")
    env_two_agents = SelfOrgGridWorld(size=5, num_agents=2, num_targets=2)
    observation, info = env_two_agents.reset()
    print("Initial Observation for Two Agents:", observation)

    # Manually set agent positions to be next to each other
    env_two_agents._agent_locations[0] = np.array([2, 2])
    env_two_agents._agent_locations[1] = np.array([2, 3])

    # Attempt to move one agent into the position of the other
    actions = [3, 0]  # First agent moves left, second agent does nothing
    observation, reward, done, _, info = env_two_agents.step(actions)
    # print(
    #     f"After Step: Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}"
    # )
    assert not np.array_equal(
        env_two_agents._agent_locations[0], env_two_agents._agent_locations[1]
    ), "Agents have overlapped!"

    env_two_agents.close()

    # Test for boundary conditions with a single agent
    print("\n--- Testing Boundary Conditions for Single Agent ---")
    env_one_agent = SelfOrgGridWorld(size=5, num_agents=1, num_targets=1)
    observation, info = env_one_agent.reset()
    # print("Initial Observation for One Agent:", observation)

    # Manually set agent position near the boundary
    env_one_agent._agent_locations[0] = np.array([0, 0])

    # Attempt to move the agent out of bounds
    actions = [1]  # Agent moves down (which should be blocked by the boundary)
    observation, reward, done, _, info = env_one_agent.step(actions)
    # print(
    #     f"After Step: Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}"
    # )

    assert (
        env_one_agent._agent_locations[0] == np.array([0, 0])
    ).all(), "Agent moved out of bounds!"

    env_one_agent.close()

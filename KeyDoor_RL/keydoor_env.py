import random
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal, Wall, Ball
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

from gymnasium.envs.registration import register


class KeyDoorEnvironment(MiniGridEnv):
    def __init__(self, size=10, max_steps=400, reward_shaping=True, **kwargs):
        mission_space = MissionSpace(
            mission_func=lambda: (
                "Unlock the coloured doors using their matching keys "
                "in the correct order to reach the goal"
            )
        )

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=3,
            **kwargs
        )

        self.reward_shaping = reward_shaping

    def _gen_grid(self, width, height):
        # Create empty grid and surround with walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Random internal walls
        for _ in range(5):
            x, y = self._random_empty_cell()
            self.grid.set(x, y, Wall())

        # Randomise door order
        self.colours = ["red", "blue", "green", "yellow"]
        random.shuffle(self.colours)
        self.current_door_index = 0

        self.mission = (
    "unlock doors in this order "
    + " ".join(self.colours)
)

        # Place keys and doors
        for colour in self.colours:
            kx, ky = self._random_empty_cell()
            self.grid.set(kx, ky, Key(colour))

            dx, dy = self._random_empty_cell()
            self.grid.set(dx, dy, Door(colour, is_open=False, is_locked=True))

        # Place coins (optional distraction)
        for _ in range(3):
            cx, cy = self._random_empty_cell()
            self.grid.set(cx, cy, Ball("yellow"))

        # Place goal
        gx, gy = self._random_empty_cell()
        self.grid.set(gx, gy, Goal())

        # Place agent
        self.place_agent()

    def _random_empty_cell(self):
        while True:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid.get(x, y) is None:
                return x, y

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        reward = 0.0

        # Check object in front of agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Reward correct door opening
        # Reward correct door opening (shaping)
        if self.reward_shaping and isinstance(fwd_cell, Door):
            expected_colour = self.colours[self.current_door_index]

            if fwd_cell.is_open and fwd_cell.color == expected_colour:
                reward += 0.5
                self.current_door_index += 1
                
        # Reward reaching goal
        if terminated:
            reward += 1.0

        return obs, reward, terminated, truncated, info


# REGISTER ENVIRONMENT (VERY IMPORTANT)
register(
    id="MiniGrid-KeyDoor-v0",
    entry_point="keydoor_env:KeyDoorEnvironment",
)

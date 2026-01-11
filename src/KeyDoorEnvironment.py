from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES

class KeyDoorEnvironment(MiniGridEnv):
    def __init__(self, size=8, max_steps=300, **kwargs):
        mission_space = MissionSpace(
            mission_func=lambda: "Unlock the doors in the correct order and reach the goal"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=7,
            **kwargs
        )


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_agent()
        keyA = Key("red")
        KeyB = Key("blue")
        KeyC = Key("green")
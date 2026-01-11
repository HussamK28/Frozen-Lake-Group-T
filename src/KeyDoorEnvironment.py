from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES

class KeyDoorEnvironment(MiniGridEnv):
    def __init__(
        self, 
        mission_space: MissionSpace, 
        grid_size: int | 
        None = None, width: int | None = None, height: int | None = None,
        max_steps: int = 100, 
        see_through_walls: bool = False, 
        agent_view_size: int = 7, 
        render_mode: str | None = None, 
        screen_size: int | None = 640, 
        highlight: bool = True, 
        tile_size: int = ..., 
        agent_pov: bool = False):
        super().__init__(mission_space, grid_size, width, height, max_steps, see_through_walls, agent_view_size, render_mode, screen_size, highlight, tile_size, agent_pov)
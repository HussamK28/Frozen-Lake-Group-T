import random
import time
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal, Wall, Ball
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

class KeyDoorEnvironment(MiniGridEnv):
    def __init__(self, size=10, max_steps=400, **kwargs):
        mission_space = MissionSpace(
            mission_func=lambda: "You have 4 coloured doors, unlock the doors with their corresponding keys in the correct order to reach the goal"
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
        


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        numWalls = 5
        for _ in range(numWalls):
            while True:
                x = random.randint(1, width-2)
                y = random.randint(1, height-2)
                if self.grid.get(x,y) is None:
                    self.grid.set(x,y,Wall())
                    break

        
        colours = ["red", "blue", "green", "yellow"]
        random.shuffle(colours)
        self.mission = f"Unlock doors in this order: {', '.join(colours)} to reach the goal"

        for i, colour in enumerate(colours):
            key = Key(colour)
            freeKeyCell = self.placeInRandomCell()
            self.grid.set(*freeKeyCell, key)

            door = Door(colour, is_open=False, is_locked=True)
            freeDoorCell = self.placeInRandomCell()
            
            self.grid.set(*freeDoorCell, door)

        numCoins = 3
        for _ in range(numCoins):
            coin = Ball("yellow")
            coin_cell = self.placeInRandomCell()
            self.grid.set(*coin_cell, coin)
             

        goal = Goal()
        goalCell = self.placeInRandomCell()
        self.grid.set(*goalCell, goal)
        self.place_agent()
        
    def placeInRandomCell(self):
        while True:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid.get(x, y) is None:
                return (x, y)


env = KeyDoorEnvironment(render_mode="human")
num_tests = 10

for episode in range(num_tests):
    print(f"\n=== Episode {episode+1} ===")
    obs, info = env.reset()

    obj_positions = {}
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if obj is not None:
                obj_positions[(x, y)] = obj

    # Print agent position
    print("Agent position:", env.agent_pos)

    # Print all keys
    keys = [(pos, obj.color) for pos, obj in obj_positions.items() if isinstance(obj, Key)]
    print("Keys:", keys)

    # Print all doors
    doors = [(pos, obj.color) for pos, obj in obj_positions.items() if isinstance(obj, Door)]
    print("Doors:", doors)

    # Print goal position
    goal = [(pos, "Goal") for pos, obj in obj_positions.items() if isinstance(obj, Goal)]

    print("Goal:", goal)

    # Render the environment for 2 seconds
    env.render()
    time.sleep(2)

print("\n✅ Testing completed. Check positions and randomization visually.")
env.close()
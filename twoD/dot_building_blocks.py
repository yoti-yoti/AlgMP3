import itertools
import numpy as np
from shapely.geometry import Point, LineString

class DotBuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        # return np.sqrt((prev_config[0] - next_config[0])**2 + (prev_config[1] - next_config[1])**2)
        return self.env.compute_distance(prev_config, next_config)

    def compute_path_cost(self, path):
        totat_cost = 0
        for i in range(len(path) - 1):
            totat_cost += self.compute_distance(path[i], path[i + 1])
        return totat_cost

    def sample_random_config(self, goal_prob, goal):
        if np.random.rand() < goal_prob:
            return goal

        x = np.random.rand() * (self.env.xlimit[1] + 1)
        y = np.random.rand() * (self.env.ylimit[1] + 1)
        return [x, y]

    def config_validity_checker(self, state):
        return self.env.config_validity_checker(state)

    def edge_validity_checker(self, state1, state2):
        return self.env.edge_validity_checker(state1, state2)


